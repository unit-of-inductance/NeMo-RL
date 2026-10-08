# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Apply streamed LoRA adapter deltas through vLLM's native weight loaders.

Kept free of vLLM imports (like vllm_sparse_delta.py) so the merge math and
stream validation are unit-testable on CPU-only machines. The receiver
extension in vllm_backend.py wires these into the collective refit path.
"""

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def _format_refit_key_error(label: str, keys: set[str]) -> str:
    """Format a bounded refit-key diagnostic."""
    ordered = sorted(keys)
    suffix = " ..." if len(ordered) > 8 else ""
    return f"{label} ({len(ordered)}): {ordered[:8]}{suffix}"


LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"


def _split_adapter_name(name: str) -> tuple[str, str] | None:
    """Split a streamed adapter record into (base_weight_name, "A" | "B").

    ``language_model.backbone.layers.3.self_attn.q_proj.lora_A.weight``
    becomes ``(..., "language_model.backbone.layers.3.self_attn.q_proj.weight", "A")``.
    Returns None for names that are not LoRA adapter records.
    """
    if name.endswith(LORA_A_SUFFIX):
        return name[: -len(LORA_A_SUFFIX)] + ".weight", "A"
    if name.endswith(LORA_B_SUFFIX):
        return name[: -len(LORA_B_SUFFIX)] + ".weight", "B"
    return None


class _AdapterDeltaLoadMode(TorchDispatchMode):
    """Turn native loader copies into in-place LoRA merges.

    The receiver feeds per-projection FP32 delta tensors (``scaling * (B @ A)``)
    through the model's own ``load_weights``. vLLM's weight loaders narrow those
    tensors to the rank-local slice of each parameter and end with
    ``param_data.copy_(loaded_weight)``. This mode intercepts those final copies
    and replaces them with the merge arithmetic the Megatron-Bridge offline
    merge uses: upcast the destination to FP32, add the delta, downcast to the
    parameter dtype. Keeping the GEMM result in FP32 until the single final
    rounding makes the receive path bitwise-equal to Bridge's merge.

    Every other tensor op passes through unchanged; copies whose destination is
    not a model parameter (for example a loader's intermediate device moves)
    pass through as well.
    """

    def __init__(self, param_storages: set[int]) -> None:
        super().__init__()
        self._targets = param_storages
        self.copies = 0

    def _is_target(self, destination: torch.Tensor) -> bool:
        return destination.untyped_storage()._cdata in self._targets

    def __torch_dispatch__(  # type: ignore[override]
        self,
        func: Any,
        _types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if func is not torch.ops.aten.copy_.default:
            return func(*args, **(kwargs or {}))

        destination = args[0]
        source = args[1]
        if not isinstance(destination, torch.Tensor) or not self._is_target(
            destination
        ):
            return func(*args, **(kwargs or {}))

        self.copies += 1
        # Bridge's merge math: FP32 add on the exact upcast base, then one
        # downcast. ``source`` stays FP32 here because the loaders only
        # narrow, never cast, before the copy.
        merged = destination.float().add(source.float())
        return destination.copy_(merged.to(destination.dtype))


class _AdapterDeltaApplier:
    """Pair streamed LoRA factor records and merge them onto live parameters.

    One instance per refit. ``post_unpack`` receives each unpack batch from
    ``packed_broadcast_consumer``; records may split across batches, so the
    not-yet-paired half is cloned (the unpack buffers are reused as soon as
    the next batch streams). Completed pairs are merged in stream order, one
    ``load_weights`` call per batch, under a dispatch mode that converts the
    loader's final copies into in-place merges.

    The Megatron-Bridge export re-emits a shared grouped-expert factor under
    every expert's HF name, so a name-addressed merge applies it to every
    expert slice without special casing. Packed 3D factors (all experts in
    one record) are supported through a chunked batched matmul.

    Each completed pair is merged and applied immediately, one pair at a
    time. A MoE layer streams hundreds of expert pairs per unpack batch;
    buffering their FP32 deltas for the whole batch is what OOMed the first
    nano-omni smoke runs (the deltas alone reach ~20 GiB with 512 routed
    experts, on top of the serving process's memory budget). Landing each
    pair before touching the next record keeps exactly one delta live at a
    time, which is the accumulate-into-destination behavior the design doc
    calls for.
    """

    _A = "A"
    _B = "B"

    def __init__(
        self,
        load_weights: Callable[[Iterable[tuple[str, torch.Tensor]]], Any],
        param_storages: set[int],
        scaling: float,
    ) -> None:
        self._load_weights = load_weights
        self._param_storages = param_storages
        self._scaling = scaling
        self._pending: dict[str, tuple[str, torch.Tensor]] = {}
        self._applied: set[str] = set()

    def post_unpack(self, batch: list[tuple[str, torch.Tensor]]) -> None:
        for name, tensor in batch:
            split = _split_adapter_name(name)
            if split is None:
                raise ValueError(
                    f"adapter_delta stream contains non-adapter record {name!r}; "
                    "the trainer and receiver disagree on the payload mode"
                )
            base_name, half = split
            previous = self._pending.get(base_name)
            if previous is None:
                # May be the only half of this pair in this batch; the unpack
                # buffer is reusable transport storage, so own the memory now.
                self._pending[base_name] = (half, tensor.clone())
                continue
            other_half, other_tensor = previous
            del self._pending[base_name]
            if half == self._A:
                linear_in, linear_out = tensor, other_tensor
            else:
                linear_in, linear_out = other_tensor, tensor
            # Merge and land this pair before the next record. Holding every
            # completed pair's delta until the end of the batch is what OOMed
            # the nano-omni smoke runs: hundreds of expert pairs share one
            # unpack batch, and their FP32 deltas add up to tens of GiB.
            self._apply(
                (base_name, self._merge_factors(base_name, linear_out, linear_in))
            )

    def _merge_factors(
        self, base_name: str, linear_out: torch.Tensor, linear_in: torch.Tensor
    ) -> torch.Tensor:
        """Compute ``scaling * (B @ A)`` in FP32, matching Bridge's merge.

        The scaling multiply is folded into the GEMM result in place (``mul_``)
        so no second full-size temporary is materialized; for packed experts
        the batched matmul itself is chunked along the expert axis so the
        FP32 GEMM output never exceeds a chunk's worth of memory. Chunking
        along the batch axis never changes an output element: each element's
        dot product belongs to exactly one expert slice, so the merge stays
        bitwise-equal to Bridge's.
        """
        if linear_in.dtype != torch.float32:
            linear_in = linear_in.to(torch.float32)
        if linear_out.dtype != torch.float32:
            linear_out = linear_out.to(torch.float32)
        if linear_in.dim() == 3 or linear_out.dim() == 3:
            if linear_in.dim() != linear_out.dim():
                raise ValueError(
                    f"adapter record {base_name!r} mixes packed and unpacked "
                    "factor ranks (shared-outer LoRA is not supported by this "
                    "receiver)"
                )
            # Packed experts: (E, out, r) @ (E, r, in) -> (E, out, in), chunked
            # along the expert axis to bound the FP32 GEMM output.
            delta = self._merge_packed_factors(base_name, linear_out, linear_in)
        elif linear_in.dim() == 2 and linear_out.dim() == 2:
            if linear_out.shape[1] != linear_in.shape[0]:
                raise ValueError(
                    f"adapter record {base_name!r} has mismatched factor ranks: "
                    f"B is {tuple(linear_out.shape)}, A is {tuple(linear_in.shape)}"
                )
            delta = linear_out @ linear_in
        else:
            raise ValueError(
                f"adapter record {base_name!r} has unsupported factor ranks "
                f"{tuple(linear_out.shape)} @ {tuple(linear_in.shape)}"
            )
        return delta.mul_(self._scaling)

    # Bound the FP32 packed-expert GEMM output to this many experts at a
    # time. At nano-omni scale (512 experts, one [E, h, i] record is ~21 GiB
    # in FP32) a chunk is ~400 MiB, well inside the serving process's slack.
    _PACKED_MERGE_CHUNK_EXPERTS = 8

    def _merge_packed_factors(
        self, base_name: str, linear_out: torch.Tensor, linear_in: torch.Tensor
    ) -> torch.Tensor:
        num_experts = linear_in.shape[0]
        out_shape = (num_experts, linear_out.shape[1], linear_in.shape[2])
        delta = torch.empty(out_shape, dtype=torch.float32, device=linear_in.device)
        chunk = self._PACKED_MERGE_CHUNK_EXPERTS
        for start in range(0, num_experts, chunk):
            stop = min(start + chunk, num_experts)
            torch.bmm(
                linear_out[start:stop],
                linear_in[start:stop],
                out=delta[start:stop],
            )
        return delta

    def _apply(self, delta: tuple[str, torch.Tensor]) -> None:
        # One load_weights call per name. Slower than one batched call, but
        # each call either lands its copies or the name is provably unresolved,
        # and "unresolvable name" must be a hard error, never a silent skip.
        base_name, delta_tensor = delta
        mode = _AdapterDeltaLoadMode(self._param_storages)
        with torch.no_grad(), mode:
            self._load_weights([(base_name, delta_tensor)])
        if mode.copies == 0:
            raise RuntimeError(
                f"adapter_delta merge found no parameter for {base_name!r}. "
                "A silent skip here would serve a stale model, so this is fatal."
            )
        self._applied.add(base_name)

    def finish(self) -> None:
        if self._pending:
            raise RuntimeError(
                "adapter_delta stream ended with unpaired factors: "
                f"{_format_refit_key_error('unpaired', set(self._pending))}. "
                "Every lora_A needs its lora_B sibling."
            )
        if not self._applied:
            raise RuntimeError(
                "adapter_delta refit received no adapter records; refusing to "
                "declare success on an empty stream"
            )
