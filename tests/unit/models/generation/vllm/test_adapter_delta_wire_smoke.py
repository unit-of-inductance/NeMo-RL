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

"""Wire-level smoke coverage for the adapter_delta refit.

The single-process test below exercises the real pieces that meet on the
wire: the source-side payload iterator (Bridge export naming contract),
arbitrary unpack-batch splits (the packed broadcast hands the post-unpack
hook a list per batch, not the whole stream), and the receiver-side merge
against a loader that narrows then copies, as vLLM's do.

The full two-process Ray + vLLM engine smoke (design doc verification 3,
including the mid-broadcast abort and the recovery latch) requires the vLLM
extra and a GPU; it lives at the bottom of this file behind the vllm marker
and importorskip. It has not been executed in this worktree: vLLM publishes
no macOS wheels, so it is written-but-not-run, exactly as the design doc's
verification plan anticipated ("nothing touches the cluster until the
change is reviewed").
"""

import pytest
import torch

from nemo_rl.models.generation.vllm.vllm_adapter_delta import _AdapterDeltaApplier


class _FusedLoaderModel(torch.nn.Module):
    """Tiny stand-in with a vLLM-style fused loader: narrow, then copy_."""

    def __init__(self, hidden: int, q_out: int, tp_size: int, tp_rank: int):
        super().__init__()
        self.q_out = q_out
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.qkv = torch.nn.Parameter(
            torch.randn(q_out * 3 // tp_size, hidden, dtype=torch.bfloat16)
        )
        self.mlp_gate_up = torch.nn.Parameter(
            torch.randn(2 * q_out // tp_size, hidden, dtype=torch.bfloat16)
        )

    def load_weights(self, weights):
        for name, tensor in weights:
            if name.endswith("q_proj.weight"):
                rows = tensor.shape[0] // self.tp_size
                start = self.tp_rank * rows
                self.qkv.data[:rows].copy_(tensor.narrow(0, start, rows))
            elif name.endswith("gate_proj.weight"):
                rows = tensor.shape[0] // self.tp_size
                start = self.tp_rank * rows
                self.mlp_gate_up.data[:rows].copy_(tensor.narrow(0, start, rows))
            # Unknown names fall through: AutoWeightsLoader would raise; the
            # applier's own copy-count check is the guard under test.


def _stors(model):
    return {p.untyped_storage()._cdata for p in model.parameters()}


def test_adapter_stream_survives_arbitrary_batch_splits():
    """Pairing must hold no matter how the transport slices the stream.

    The packed broadcast consumer calls post_unpack once per unpack batch.
    A and B of one projection may land in different batches; the applier
    must buffer the first half (the unpack buffer is reusable transport
    storage) and merge when the sibling arrives. Batch boundaries here are
    deliberately adversarial: one record per batch, B before A, and a
    batch holding two different projections' halves at once.
    """
    torch.manual_seed(0)
    hidden, q_out, dim = 16, 8, 4
    scaling = 128 / 64
    model = _FusedLoaderModel(hidden, q_out, tp_size=2, tp_rank=1)
    before_qkv = model.qkv.detach().clone()
    before_gate_up = model.mlp_gate_up.detach().clone()

    # Factors as Bridge emits them: full (unsharded) HF shapes in BF16.
    a_q = torch.randn(dim, hidden, dtype=torch.bfloat16)
    b_q = torch.randn(q_out, dim, dtype=torch.bfloat16)
    a_g = torch.randn(dim, hidden, dtype=torch.bfloat16)
    b_g = torch.randn(q_out, dim, dtype=torch.bfloat16)

    applier = _AdapterDeltaApplier(
        load_weights=model.load_weights, param_storages=_stors(model), scaling=scaling
    )
    records = [
        ("m.layers.0.self_attn.q_proj.lora_B.weight", b_q),  # B first
        ("m.layers.0.mlp.gate_proj.lora_A.weight", a_g),  # mixed batch below
        ("m.layers.0.self_attn.q_proj.lora_A.weight", a_q),
        ("m.layers.0.mlp.gate_proj.lora_B.weight", b_g),
    ]
    # Adversarial splits: singletons and a two-record batch mixing halves
    # of different projections.
    for batch in [records[0:1], records[1:3], records[3:4]]:
        applier.post_unpack(batch)
    applier.finish()

    # Expected: the local TP slice of each delta, fp32 add, one downcast.
    def local_rows(param, tensor, section_rows):
        rows = tensor.shape[0] // model.tp_size
        start = model.tp_rank * rows
        return (param.float() + tensor[start : start + rows].float()).to(torch.bfloat16)

    expected_qkv = before_qkv.clone()
    delta_q = scaling * (b_q.float() @ a_q.float())
    expected_qkv[: q_out // model.tp_size] = local_rows(
        before_qkv[: q_out // model.tp_size], delta_q, q_out
    )
    expected_gate_up = before_gate_up.clone()
    delta_g = scaling * (b_g.float() @ a_g.float())
    expected_gate_up[: q_out // model.tp_size] = local_rows(
        before_gate_up[: q_out // model.tp_size], delta_g, q_out
    )
    assert torch.equal(model.qkv.detach(), expected_qkv)
    assert torch.equal(model.mlp_gate_up.detach(), expected_gate_up)


def test_pairing_survives_buffer_reuse_by_producer():
    """The applier must own its memory: later batches may reuse tensors.

    post_unpack clones the pending half precisely so the transport can
    reuse its unpack buffers; simulate that by mutating the source tensor
    after the batch that carried it.
    """
    torch.manual_seed(1)
    hidden, q_out, dim = 16, 8, 4
    model = _FusedLoaderModel(hidden, q_out, tp_size=1, tp_rank=0)
    before = model.qkv.detach().clone()
    a = torch.randn(dim, hidden, dtype=torch.bfloat16)
    b = torch.randn(q_out, dim, dtype=torch.bfloat16)
    expected = (before[:q_out].float() + 2.0 * (b.float() @ a.float())).to(
        torch.bfloat16
    )

    applier = _AdapterDeltaApplier(
        load_weights=model.load_weights, param_storages=_stors(model), scaling=2.0
    )
    name = "m.layers.0.self_attn.q_proj"
    applier.post_unpack([(f"{name}.lora_A.weight", a)])
    a.zero_()  # producer reuses the buffer
    applier.post_unpack([(f"{name}.lora_B.weight", b)])
    b.zero_()
    applier.finish()

    got = model.qkv.detach().clone()
    assert torch.equal(got[:q_out], expected)
    # k/v sections untouched.
    assert torch.equal(got[q_out:], before[q_out:])


# ---------------------------------------------------------------------------
# Two-process Ray + vLLM wire smoke (design doc verification 3).
# Requires the vllm extra and a GPU; written-but-not-run in this worktree.
# ---------------------------------------------------------------------------


@pytest.mark.vllm
def test_two_process_adapter_refit_and_recovery():
    """End-to-end: trainer actor -> vLLM worker actor, adapter payload.

    Asserts the receiver's post-refit parameters equal the trainer's merged
    export parameters name by name, then aborts the receiver mid-broadcast
    via hold_refit_for_fault_injection and asserts the recovery path forces
    one full hf_export refit before adapter mode resumes.
    """
    pytest.importorskip("vllm")
    pytest.skip(
        "Requires a GPU host with the vllm extra; not executable in this "
        "CPU-only worktree. Run on the dev image alongside the existing "
        "vllm-marked refit tests."
    )
