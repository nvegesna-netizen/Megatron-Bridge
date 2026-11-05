# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""One-GB200 validation of scoped SHA256 probes and real MXFP8 storage handling."""

import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    _GroupedOutputProjection,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer


LOGGER = logging.getLogger(__name__)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _check_fused_moe(upstream, directory):
    """Exercise the actual MCore fused expert path, including its emulated hooks.

    Tiny local experts use real MXFP8 compute and BF16 parameters/ordinary
    autograd. This is not EP32, native-FP8 parameters, or the full DSv4 model.
    The separate DDP fixture below covers buffers and CUDA capture/replay.
    """
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear,
    )
    from megatron.core.fp8_utils import get_fp8_context
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe import experts as experts_module
    from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP

    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_ffn_hidden_size=256,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        activation_func_clamp_value=10.0,
        moe_mlp_glu_interleave_size=32,
        bias_activation_fusion=False,
        gradient_accumulation_fusion=False,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_use_grouped_tensor=True,
        use_transformer_engine_op_fuser=True,
        moe_router_padding_for_quantization=True,
        fp8="e4m3",
        fp8_recipe="mxfp8",
    )
    experts = TEGroupedMLP(
        num_local_experts=2,
        config=config,
        submodules=GroupedMLPSubmodules(TEColumnParallelGroupedLinear, TERowParallelGroupedLinear),
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    ).cuda()
    assert experts._with_fused_impl, "Preflight must exercise the real fused expert path"

    def legacy_observer(module, args, kwargs):
        pass

    handle = experts.linear_fc1.register_forward_pre_hook(legacy_observer, with_kwargs=True)
    try:
        try:
            experts._make_fused_impl_pre_forward_hook()(None)
        except TypeError as error:
            assert "kwargs" in str(error)
        else:
            raise AssertionError("The actual MCore fused hook adapter must reproduce the old failure")
    finally:
        handle.remove()

    x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    probs = torch.full((256,), 0.5, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    counts = torch.tensor([128, 128], device="cuda", dtype=torch.int64)

    def run():
        experts.zero_grad(set_to_none=True)
        x.grad, probs.grad = None, None
        with get_fp8_context(config):
            output, bias = experts(x, counts, probs)
        assert bias is None
        output.float().square().mean().backward()
        assert x.grad is not None and probs.grad is not None
        assert all(param.grad is not None for param in experts.parameters())
        return [output.detach().clone(), x.grad.clone(), probs.grad.clone()] + [
            param.grad.clone() for param in experts.parameters()
        ]

    reference = run()
    assert experts._fused_ops is not None
    root = torch.nn.Module()
    root.add_module("experts", experts)
    directory.mkdir(parents=True, exist_ok=False)
    attention = sys.modules[_GroupedOutputProjection.__module__]
    trace = upstream.UpstreamTrace(directory, 0, [root], attention)
    for index in range(3):
        trace.projection_input(index, x, next(experts.parameters()))
        observed = run()
        trace.projection_output(index, observed[0])
        assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(reference, observed))
    trace.finish_iteration()
    assert not trace.hooks and not trace.restores and not trace.active
    records = [json.loads(line) for line in (directory / "upstream_rank00000.jsonl").read_text().splitlines()]
    gaps = [r for r in records if r.get("phase") == "module_input" and r["value"].get("complete") is False]
    assert len(gaps) >= 6, "Both fused expert children must explicitly report unavailable intermediate inputs"
    assert all("linear_fc" in row["name"] for row in gaps)
    assert any(r.get("phase") == "module_output" and r["name"] == "chunk0.experts" for r in records)
    assert records[-1]["kind"] == "upstream_complete"
    summary = {
        "passed": True,
        "legacy_failure_reproduced": True,
        "actual_fused_impl": experts._with_fused_impl,
        "actual_fused_ops_created": experts._fused_ops is not None,
        "source_sha256": hashlib.sha256(Path(experts_module.__file__).read_bytes()).hexdigest(),
        "emulated_input_gaps": len(gaps),
        "output_input_and_parameter_gradients_bitwise_unchanged": True,
        "observed_forwards": 3,
        "scope": "Actual TEGroupedMLP/MXFP8 op-fuser, two local experts, BF16 weights, no EP routing or native-FP8 optimizer",
    }
    return summary, records


class LanguageModelEmbedding(torch.nn.Embedding):
    """Fixture with the existing diagnostic's required data boundary."""


class _Model(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.embedding = LanguageModelEmbedding(16, 32768, device="cuda", dtype=torch.bfloat16)
        self.linear_o_group_proj = torch.nn.Parameter(
            torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16) / 64
        )

    def forward(self, tokens):
        x = self.embedding(tokens)
        results = []
        for _ in range(3):
            if self.path == "new":
                result = _GroupedOutputProjection.apply(x, self.linear_o_group_proj, 8)
            else:
                result = torch.einsum(
                    "...gd,grd->...gr", x.view(4096, 1, 8, 4096), self.linear_o_group_proj.view(8, 1024, 4096)
                )
                result = result.reshape(4096, 1, 8192)
            results.append(result)
        return sum(results)


def main() -> None:
    """Assert all evidence paths work before allowing any full-model launch."""
    # Initialize TE's Python loader (via the imports above) before importing its
    # binary extension. The private image does not expose it as a top-level file.
    import transformer_engine_torch as tex

    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--path", choices=("old", "new"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    torch.cuda.set_device(0)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.autograd.graph.set_override_stale_capture_stream(True)
    torch.distributed.init_process_group("nccl", rank=0, world_size=1, init_method="file://" + tempfile.mktemp())
    parallel_state.initialize_model_parallel()
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(1234)
    upstream = _load("megatron.bridge.training.dsv4_upstream", args.upstream)
    fused_moe, fused_records = _check_fused_moe(upstream, args.output / "fused_moe")
    # Prove the stronger hash sees the known byte-XOR collision and permutation.
    a = torch.tensor([1, 1, 2, 2], device="cuda", dtype=torch.int32)
    b = torch.tensor([3, 3, 4, 4], device="cuda", dtype=torch.int32)
    assert upstream.tensor_signature(a)["sha256"] != upstream.tensor_signature(b)["sha256"]
    assert upstream.tensor_signature(a)["sha256"] != upstream.tensor_signature(a.flip(0))["sha256"]
    # Real TE quantized class, including valid (not padded) scale storage.
    quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    tensor = quantizer(torch.randn(128, 128, dtype=torch.bfloat16, device="cuda"))
    quantized = upstream.tensor_signature(tensor)
    assert quantized["complete"], quantized
    assert set(quantized["storage"]) == {"rowwise_data", "rowwise_scale_inv"}
    probe = _load("dsv4_boundary_probe", args.probe)
    config = TransformerConfig(num_layers=1, hidden_size=32768, num_attention_heads=64, bf16=True)
    ddp = DistributedDataParallel(
        config,
        module=_Model(args.path),
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_grad_reduce=True,
            grad_reduce_in_fp32=False,
            overlap_param_gather=False,
            bucket_size=134217728,
        ),
    )
    optimizer = torch.optim.SGD(ddp.parameters(), lr=1e-5)
    os.environ["DSV4_NUMERICS_DIR"] = str(args.output)
    os.environ["DSV4_NUMERICS_STOP_AFTER"] = "999"
    tokens = torch.randint(0, 16, (4096, 1), device="cuda")
    # Check actual unchanged values/gradients with hooks installed, not just hashes.
    ddp.zero_grad_buffer()
    reference = ddp(tokens)
    reference.float().square().mean().backward()
    ddp.finish_grad_sync()
    expected_output = reference.detach().clone()
    expected_grads = [p.main_grad.detach().clone() for p in ddp.parameters()]
    graph = None

    def forward_backward():
        ddp.zero_grad_buffer()
        result = ddp(tokens)
        loss = result.float().square().mean()
        loss.backward()
        ddp.finish_grad_sync()
        return result, loss

    for iteration in range(1, 6):
        probe.begin([ddp], optimizer, iteration)
        optimizer.zero_grad(set_to_none=True)
        if iteration <= 3:
            output, loss = forward_backward()
        elif iteration == 4:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output, loss = forward_backward()
            torch.cuda.current_stream().wait_stream(stream)
            graph.replay()
        else:
            graph.replay()
        if iteration == 1:
            assert torch.equal(expected_output.view(torch.int16), output.detach().view(torch.int16))
            assert all(
                torch.equal(expected.view(torch.int16), p.main_grad.view(torch.int16))
                for expected, p in zip(expected_grads, ddp.parameters())
            )
        for param in ddp.parameters():
            param.grad = param.main_grad.detach().clone()
        norm = torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in ddp.parameters()]))
        optimizer.step()
        probe.end({"lm loss": loss.detach()}, norm, 0)
    assert probe._TRACE.upstream.completed and not probe._TRACE.upstream.hooks
    assert probe._TRACE.captured
    path = args.output / "upstream_rank00000.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[-1]["projections"] == [0, 1, 2]
    assert [r["name"] for r in records if r.get("phase") == "projection_input"] == [
        "projection0000",
        "projection0001",
        "projection0002",
    ]
    assert len(list(args.output.glob("upstream_operands_rank00000_*.pt"))) == 6
    summary = {
        "passed": True,
        "path": args.path,
        "upstream_records": len(records),
        "captured_records": len(probe._TRACE.captured),
        "quantized_signature": quantized,
        "upstream_sha256": hashlib.sha256(Path(args.upstream).read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(args.probe).read_bytes()).hexdigest(),
        "fused_moe": fused_moe,
    }
    (args.output / "preflight_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.warning("UPSTREAM_PREFLIGHT_PASS %s", json.dumps(summary))
    probe._TRACE.file.close()
    # Return complete small JSON evidence through the already-preserved stdout;
    # do not depend on an SSH hop to ephemeral compute-local files.
    evidence = {
        "summary": summary,
        "fused_upstream": fused_records,
        "upstream": records,
        "boundary": [json.loads(line) for line in (args.output / "rank00000.jsonl").read_text().splitlines()],
    }
    LOGGER.warning(
        "UPSTREAM_EVIDENCE_GZIP_BASE64 %s", base64.b64encode(gzip.compress(json.dumps(evidence).encode())).decode()
    )
    os._exit(0)


if __name__ == "__main__":
    main()
