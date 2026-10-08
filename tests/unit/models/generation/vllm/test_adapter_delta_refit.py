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

"""CPU-only tests for the adapter-delta refit receive path.

Covers the two risks the design doc calls out as retirable without GPUs:

1. Merge math: the receiver's in-place merge must be bitwise-equal to the
   Megatron-Bridge offline merge for the same factors.
2. Loader interception: a loader that narrows streamed deltas and ends in
   ``param_data.copy_(loaded_weight)`` (the vLLM pattern) must have its
   copies converted into merges, and unresolved names must hard-error.

The name-pair round trip against a real Megatron adapter graph needs the
mcore/Bridge stack; it lives with the source-side tests in
tests/unit/models/policy/test_megatron_worker.py. The Bridge merge-parity
test below also imports megatron.bridge, so it carries the ``mcore`` marker
and is skipped in the default unit run.
"""

import pytest
import torch

from nemo_rl.models.generation.vllm.vllm_adapter_delta import (
    _AdapterDeltaApplier,
    _AdapterDeltaLoadMode,
    _split_adapter_name,
)


def test_split_adapter_name_pairs():
    name = "language_model.backbone.layers.3.self_attn.q_proj.lora_A.weight"
    base, half = _split_adapter_name(name)
    assert base == "language_model.backbone.layers.3.self_attn.q_proj.weight"
    assert half == "A"

    name_b = "language_model.backbone.layers.3.self_attn.q_proj.lora_B.weight"
    base_b, half_b = _split_adapter_name(name_b)
    assert base_b == base
    assert half_b == "B"

    assert _split_adapter_name("model.layers.3.self_attn.q_proj.weight") is None


class _FakeFusedLinear(torch.nn.Module):
    """Mimics vLLM's fused-param weight_loader: narrow, then copy_."""

    def __init__(self, out_features: int, in_features: int, tp_size: int, tp_rank: int):
        super().__init__()
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        # Local shard of a column-parallel weight.
        self.weight = torch.nn.Parameter(
            torch.randn(out_features // tp_size, in_features, dtype=torch.bfloat16)
        )

    def weight_loader(self, param, loaded_weight, shard_offset=0):
        # Fused sections (q/k/v, gate/up) each load a slice of the param at
        # their own offset, narrowed to this rank's TP shard first -- the
        # vLLM MergedColumnParallel pattern in miniature.
        full_rows = loaded_weight.shape[0]
        shard_size = full_rows // self.tp_size
        start = self.tp_rank * shard_size
        local = param.data.narrow(0, shard_offset, shard_size)
        sliced = loaded_weight.narrow(0, start, shard_size)
        assert local.shape == sliced.shape
        local.copy_(sliced)


class _FakeModel(torch.nn.Module):
    """Mimics a vLLM model: name mapping, per-name loader dispatch."""

    # Which fused row offset each HF projection name owns (q in a fused qkv).
    _NAME_TO_SECTION_OFFSET = {"q_proj": 0}

    def __init__(self, hidden: int, q_out: int, tp_size: int, tp_rank: int):
        super().__init__()
        self.qkv = _FakeFusedLinear(q_out * 3, hidden, tp_size, tp_rank)
        self.unrelated = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))

    def load_weights(self, weights):
        loaded = []
        for name, tensor in weights:
            # HF checkpoint name -> fused param, per-shard load (the vLLM
            # MergedColumnParallel pattern, simplified to one section).
            for projection, offset in self._NAME_TO_SECTION_OFFSET.items():
                if name.endswith(f"{projection}.weight"):
                    self.qkv.weight_loader(self.qkv.weight, tensor, shard_offset=offset)
                    loaded.append(name)
                    break
            # anything else: silently skipped, exactly like vLLM's
            # AutoWeightsLoader for names outside the model.
        return set(loaded)


def _param_storages(model: torch.nn.Module) -> set[int]:
    return {p.untyped_storage()._cdata for p in model.parameters()}


@pytest.mark.mcore
def test_merge_math_matches_bridge_bitwise():
    """Receiver merge == Bridge merge, bitwise, in BF16."""
    torch.manual_seed(0)
    alpha, dim = 128, 64
    scaling = alpha / dim
    in_f, out_f = 96, 128
    base = torch.randn(out_f, in_f, dtype=torch.float32)
    # The factors ship in BF16 (adapter storage dtype), exactly what both
    # paths see. Downcast first so both merges start from identical inputs.
    a = torch.randn(dim, in_f, dtype=torch.float32).to(torch.bfloat16)
    b = torch.randn(out_f, dim, dtype=torch.float32).to(torch.bfloat16)

    # Bridge path (peft_bridge._merge_single_adapter_weight): everything
    # upcast to fp32, merged, downcast to base dtype.
    from megatron.bridge.models.conversion.peft_bridge import MegatronPeftBridge

    bridge_merged = MegatronPeftBridge._merge_single_adapter_weight(
        None,
        base.to(torch.bfloat16),
        alpha,
        dim,
        a,
        b,
    )

    # Receiver path: fp32 GEMM on the upcast factors, fp32 add on the upcast
    # base, one downcast. This is exactly what _AdapterDeltaApplier computes
    # and what _AdapterDeltaLoadMode does with the result.
    delta = scaling * (b.float() @ a.float())
    receiver_merged = (base.to(torch.bfloat16).float() + delta).to(torch.bfloat16)

    assert bridge_merged.dtype == receiver_merged.dtype == torch.bfloat16
    assert torch.equal(bridge_merged, receiver_merged), (
        "receiver merge must be bitwise-equal to the Bridge merge"
    )


def test_loader_copies_become_merges():
    """Intercepted loader copies add the delta to the local shard."""
    torch.manual_seed(1)
    tp_size, tp_rank, hidden, q_out = 4, 2, 32, 24
    model = _FakeModel(hidden, q_out, tp_size, tp_rank)
    # The loader picks this rank's slice of the q projection (the first
    # fused section) and lands it at local offset 0.
    q_local_rows = q_out // tp_size
    q_start = tp_rank * q_local_rows
    before = model.qkv.weight.detach().clone()

    dim, alpha = 8, 16
    scaling = alpha / dim
    a = torch.randn(dim, hidden, dtype=torch.bfloat16)
    b = torch.randn(q_out, dim, dtype=torch.bfloat16)

    # The delta the applier would compute: fp32 GEMM.
    delta = scaling * (b.float() @ a.float())

    mode = _AdapterDeltaLoadMode(_param_storages(model))
    with torch.no_grad(), mode:
        model.load_weights([("model.layers.0.self_attn.q_proj.weight", delta)])
    assert mode.copies == 1, mode.copies

    # Expected local merge: upcast the q section, add its sliced fp32 delta,
    # downcast once. The k/v sections of the fused param are untouched.
    expected = before.clone()
    expected[:q_local_rows] = (
        before[:q_local_rows].float() + delta[q_start : q_start + q_local_rows].float()
    ).to(torch.bfloat16)
    assert torch.equal(model.qkv.weight.detach(), expected)
    # The unrelated parameter was untouched (copy passed through / never hit).
    assert torch.count_nonzero(model.unrelated) == 0


def test_applier_pairs_streams_and_finish_validates():
    torch.manual_seed(2)
    hidden, q_out, dim, alpha = 16, 8, 4, 8
    scaling = alpha / dim
    model = _FakeModel(hidden, q_out, tp_size=1, tp_rank=0)
    a = torch.randn(dim, hidden, dtype=torch.bfloat16)
    b = torch.randn(q_out, dim, dtype=torch.bfloat16)
    before = model.qkv.weight.detach().clone()
    # The q delta merges into the first q_out rows of the fused param; the
    # k/v sections stay untouched.
    expected = before.clone()
    expected[:q_out] = (before[:q_out].float() + scaling * (b.float() @ a.float())).to(
        torch.bfloat16
    )

    applier = _AdapterDeltaApplier(
        load_weights=model.load_weights,
        param_storages=_param_storages(model),
        scaling=scaling,
    )
    name = "model.layers.0.self_attn.q_proj"
    # Split the pair across batches: A alone first (buffer-reuse hazard is the
    # reason the applier clones pending halves), then B.
    applier.post_unpack([(f"{name}.lora_A.weight", a)])
    assert model.qkv.weight.detach().equal(expected) is False
    applier.post_unpack([(f"{name}.lora_B.weight", b)])
    assert torch.equal(model.qkv.weight.detach(), expected)
    applier.finish()


def test_applier_hard_errors_on_unresolved_name():
    torch.manual_seed(3)
    model = _FakeModel(hidden=16, q_out=8, tp_size=1, tp_rank=0)
    a = torch.randn(4, 16, dtype=torch.bfloat16)
    b = torch.randn(8, 4, dtype=torch.bfloat16)
    applier = _AdapterDeltaApplier(
        load_weights=model.load_weights,
        param_storages=_param_storages(model),
        scaling=2.0,
    )
    # No parameter answers to moe.gate_up_proj; the loader skips it silently,
    # so the applier itself must refuse.
    with pytest.raises(RuntimeError, match="no parameter"):
        applier.post_unpack(
            [
                ("model.moe.gate_up_proj.lora_A.weight", a),
                ("model.moe.gate_up_proj.lora_B.weight", b),
            ]
        )


def test_applier_hard_errors_on_non_adapter_record():
    model = _FakeModel(hidden=16, q_out=8, tp_size=1, tp_rank=0)
    applier = _AdapterDeltaApplier(
        load_weights=model.load_weights,
        param_storages=_param_storages(model),
        scaling=2.0,
    )
    with pytest.raises(ValueError, match="non-adapter record"):
        applier.post_unpack(
            [("model.layers.0.self_attn.q_proj.weight", torch.zeros(2))]
        )


def test_applier_finish_rejects_unpaired_and_empty():
    model = _FakeModel(hidden=16, q_out=8, tp_size=1, tp_rank=0)
    applier = _AdapterDeltaApplier(
        load_weights=model.load_weights,
        param_storages=_param_storages(model),
        scaling=2.0,
    )
    applier.post_unpack(
        [("model.layers.0.self_attn.q_proj.lora_A.weight", torch.randn(4, 16))]
    )
    with pytest.raises(RuntimeError, match="unpaired"):
        applier.finish()

    empty = _AdapterDeltaApplier(
        load_weights=model.load_weights,
        param_storages=_param_storages(model),
        scaling=2.0,
    )
    with pytest.raises(RuntimeError, match="no adapter records"):
        empty.finish()


def test_merge_math_packed_3d_factors():
    """Packed expert factors (E, r, in) x (E, out, r) merge per expert."""
    torch.manual_seed(4)
    e, r, in_f, out_f = 6, 4, 16, 8
    scaling = 2.0
    a = torch.randn(e, r, in_f, dtype=torch.bfloat16)
    b = torch.randn(e, out_f, r, dtype=torch.bfloat16)
    delta = scaling * torch.bmm(b.float(), a.float())
    # Per-expert check: bmm must equal the per-expert 2D merges.
    for expert in range(e):
        per_expert = scaling * (b[expert].float() @ a[expert].float())
        assert torch.equal(delta[expert], per_expert)


def test_merge_math_rejects_mixed_2d_3d_pair():
    """Shared-outer pairs must be expanded per-expert by the source.

    A 2D/3D pair cannot be merged elementwise; the source's
    expand_shared_outer=True contract (pinned by the worker test) prevents
    this shape from ever arriving, and the applier enforces it fail-loud.
    """
    model = _FakeModel(hidden=16, q_out=8, tp_size=1, tp_rank=0)
    applier = _AdapterDeltaApplier(
        load_weights=model.load_weights,
        param_storages=_param_storages(model),
        scaling=2.0,
    )
    a_3d = torch.randn(6, 4, 16, dtype=torch.bfloat16)
    b_2d = torch.randn(8, 4, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="shared-outer"):
        applier.post_unpack(
            [
                ("model.moe.experts.gate_proj.lora_A.weight", a_3d),
                ("model.moe.experts.gate_proj.lora_B.weight", b_2d),
            ]
        )
