# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""One-GPU probe preflight: actual MCore DDP, eager then captured/replayed F/B.

This verifies instrumentation, not full DSv4 numerical equivalence. Run in two
fresh processes with --path old/new; no production model or optimizer changes.
"""

import argparse
import importlib.util
import json
import logging
import os
import tempfile
from pathlib import Path

import torch
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    _GroupedOutputProjection,
)


LOGGER = logging.getLogger(__name__)


class LanguageModelEmbedding(torch.nn.Embedding):
    """Small-vocabulary embedding with the training probe's semantic boundary."""


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
        if self.path == "new":
            return _GroupedOutputProjection.apply(x, self.linear_o_group_proj, 8)
        y = torch.einsum("...gd,grd->...gr", x.view(4096, 1, 8, 4096), self.linear_o_group_proj.view(8, 1024, 4096))
        return y.reshape(4096, 1, 8192)


def main() -> None:
    """Exercise the actual wrappers and reject incomplete trace coverage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    parser.add_argument("--path", choices=("old", "new"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.cuda.set_device(0)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.autograd.graph.set_override_stale_capture_stream(True)
    torch.distributed.init_process_group(
        "nccl", rank=0, world_size=1, init_method="file://" + tempfile.mktemp(prefix="dsv4-preflight-")
    )
    parallel_state.initialize_model_parallel()
    spec = importlib.util.spec_from_file_location("dsv4_probe", args.probe)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    # Validate value-change sensitivity, including a BF16 one-bit change.
    a = torch.arange(4096, device="cuda", dtype=torch.int32)
    b = a.clone()
    b[0] += 1
    assert torch.equal(probe._digest(a), probe._digest(a.clone()))
    assert not torch.equal(probe._digest(a), probe._digest(b))
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
    os.environ["DSV4_NUMERICS_DIR"] = args.output
    os.environ["DSV4_NUMERICS_STOP_AFTER"] = "999"
    tokens = torch.randint(0, 16, (4096, 1), device="cuda")
    graph = None

    def forward_backward():
        ddp.zero_grad_buffer()
        for microbatch in range(2):
            context = ddp.no_sync() if microbatch == 0 else __import__("contextlib").nullcontext()
            with context:
                loss = ddp(tokens).float().square().mean()
                loss.backward()
        ddp.finish_grad_sync()
        return loss

    for iteration in range(1, 6):
        probe.begin([ddp], optimizer, iteration)
        optimizer.zero_grad(set_to_none=True)
        if iteration <= 3:
            loss = forward_backward()
        elif iteration == 4:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                loss = forward_backward()
            torch.cuda.current_stream().wait_stream(stream)
            graph.replay()
        else:
            tokens.copy_(torch.randint(0, 16, tokens.shape, device="cuda"))
            graph.replay()
        for param in ddp.parameters():
            param.grad = param.main_grad.detach().clone()
        norm = torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in ddp.parameters()]))
        optimizer.step()
        probe.end({"lm loss": loss.detach()}, norm, 0)
    probe._TRACE.file.close()
    rows = [json.loads(line) for line in (Path(args.output) / "rank00000.jsonl").read_text().splitlines()]
    metrics = [row for row in rows if row["kind"] == "metrics"]
    assert len(metrics) == 5
    expected = {
        "data_input",
        "projection_input",
        "projection_output",
        "projection_grad_y",
        "grad_pre_reduce",
        "grad_post_reduce",
        "optimizer_before",
        "optimizer_after",
    }
    assert all(expected.issubset(row["counts"]) for row in metrics)
    assert len(probe._TRACE.captured) > 0
    assert torch.cuda.is_current_stream_capturing() is False
    LOGGER.info(
        "PREFLIGHT_PASS %s",
        json.dumps(
            {
                "path": args.path,
                "metrics": metrics,
                "captured_probe_records": len(probe._TRACE.captured),
                "rows": len(rows),
            }
        ),
    )
    # No teardown diagnostic is being tested; close only after all evidence is written.
    os._exit(0)


if __name__ == "__main__":
    main()
