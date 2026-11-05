# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in, rank-local boundary screens for the DSv4 projection investigation.

No op-dispatch mode, extra collective, modified gradient, or deterministic flag.
GPU digests are resolved only outside graph capture. Digests are screens, not
proofs of equality: eight ordered chunks still have within-chunk XOR collisions.
"""

import collections
import functools
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path

import torch


LOGGER = logging.getLogger(__name__)
_TRACE = None


def _digest(tensor: torch.Tensor) -> torch.Tensor:
    raw = tensor.detach().contiguous().view(torch.int8).reshape(-1)
    chunks = 8 if raw.numel() >= 8 and raw.numel() % 8 == 0 else 1
    return torch.hash_tensor(raw.reshape(chunks, -1), dim=1).view(torch.int64)


class _Trace:
    def __init__(self, directory, model, optimizer):
        if not hasattr(torch, "hash_tensor"):
            raise RuntimeError("The pinned image must supply torch.hash_tensor; no silent fallback")
        self.rank = torch.distributed.get_rank()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"rank{self.rank:05d}.jsonl"
        # Refuse to mix a restarted process with an earlier partial trace.
        self.file = self.path.open("x", buffering=1)
        self.model = model
        self.optimizer = optimizer
        self.names = {id(p): f"chunk{i}.{n}" for i, m in enumerate(model) for n, p in m.named_parameters()}
        self.step = 0
        self.pending = []
        self.captured = []
        self.seen = collections.Counter()
        self.current_group = None
        self.projection_calls = 0
        self.saved_operands = {}
        self.hooks = []
        self._install()

    def _write(self, record):
        self.file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def record(self, phase, name, tensor):
        if not isinstance(tensor, torch.Tensor) or not tensor.numel():
            return
        capture = torch.cuda.is_current_stream_capturing()
        key = (capture, phase, name)
        occurrence = self.seen[key]
        self.seen[key] += 1
        signature = {
            "schema": 1,
            "execution": "captured" if capture else "eager",
            "phase": phase,
            "name": name,
            "occurrence": occurrence,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
        }
        with torch.no_grad():
            digest = _digest(tensor)
        (self.captured if capture else self.pending).append((signature, digest))

    def _projection(self, original, x, weight):
        index = self.projection_calls
        self.projection_calls += 1
        name = f"projection{index:04d}"
        if self.upstream.active:
            self.upstream.projection_input(index, x.reshape(*x.shape[:2], -1), weight.reshape(-1, weight.shape[-1]))
        # Canonical logical shapes are the same for the old einsum and new helper.
        self.record("projection_input", name + ".x", x.reshape(*x.shape[:2], -1))
        self.record("projection_input", name + ".weight", weight.reshape(-1, weight.shape[-1]))
        result = original()
        if self.upstream.active:
            self.upstream.projection_output(index, result.reshape(*x.shape[:2], -1))
        self.record("projection_output", name, result.reshape(*x.shape[:2], -1))
        # Save one real operand set in iteration 1, before any CUDA graph capture.
        save = self.rank in (0, 4, 31) and self.step == 1 and index == 0
        if save:
            self.saved_operands["x"] = x.detach().reshape(*x.shape[:2], -1).clone()
            self.saved_operands["weight"] = weight.detach().reshape(-1, weight.shape[-1]).clone()

        def incoming(gradient):
            self.record("projection_grad_y", name, gradient.reshape(*x.shape[:2], -1))
            if save:
                self.saved_operands["grad_y"] = gradient.detach().reshape(*x.shape[:2], -1).clone()
            # Returning None leaves the actual autograd gradient unchanged.

        if result.requires_grad:
            result.register_hook(incoming)
        return result

    def _install(self):
        import megatron.core.distributed.param_and_grad_buffer as buffers
        import megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention as attention

        source = Path(inspect.getfile(attention)).resolve()
        from megatron.bridge.training.dsv4_upstream import UpstreamTrace

        self.upstream = UpstreamTrace(self.directory, self.rank, self.model, attention)
        self._write(
            {
                "kind": "runtime",
                "schema": 1,
                "rank": self.rank,
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "attention_source": str(source),
                "attention_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "world_size": torch.distributed.get_world_size(),
                "hostname": os.uname().nodename,
                "digest": "8 ordered raw-byte XOR chunks where divisible; screening only",
            }
        )
        original_einsum = torch.einsum

        @functools.wraps(original_einsum)
        def einsum(equation, *operands):
            if equation == "...gd,grd->...gr":
                x, weight = operands
                return self._projection(lambda: original_einsum(equation, *operands), x, weight)
            return original_einsum(equation, *operands)

        torch.einsum = einsum
        if hasattr(attention, "_GroupedOutputProjection"):
            original_apply = attention._GroupedOutputProjection.apply

            def apply(x, weight, groups):
                return self._projection(lambda: original_apply(x, weight, groups), x, weight)

            attention._GroupedOutputProjection.apply = staticmethod(apply)

        original_reduce = buffers.dist_reduce_scatter_func

        @functools.wraps(original_reduce)
        def reduce(output, input_tensor, *args, **kwargs):
            group = self.current_group
            if group is None:
                raise RuntimeError("Reduce-scatter outside an identified bucket group")
            bucket = next(b for b in group.buckets if b.grad_data.data_ptr() == input_tensor.data_ptr())
            # Called inside the original stream context, after production readiness
            # waits/scaling, and immediately before the original collective launch.
            for param in bucket.params_list:
                name = self.names[id(param)]
                self.record("grad_pre_reduce", name, param.main_grad)
            return original_reduce(output, input_tensor, *args, **kwargs)

        buffers.dist_reduce_scatter_func = reduce
        bucket_class = buffers._ParamAndGradBucketGroup
        original_start = bucket_class.start_grad_sync
        original_finish = bucket_class.finish_grad_sync

        @functools.wraps(original_start)
        def start(group, *args, **kwargs):
            previous = self.current_group
            self.current_group = group
            try:
                return original_start(group, *args, **kwargs)
            finally:
                self.current_group = previous

        @functools.wraps(original_finish)
        def finish(group, *args, **kwargs):
            already_finished = group.grad_reduce_finished
            result = original_finish(group, *args, **kwargs)
            if not already_finished:
                for i, bucket in enumerate(group.buckets):
                    shards = group.cached_grad_buffer_shard_list[i]
                    if shards is None:
                        raise RuntimeError("Expected distributed-optimizer reduce-scatter shards")
                    shard = shards[group.intra_distributed_optimizer_instance_rank]
                    itemsize = bucket.grad_data.element_size()
                    shard_start = (shard.data_ptr() - bucket.grad_data.data_ptr()) // itemsize
                    shard_end = shard_start + shard.numel()
                    for param in bucket.params_list:
                        param_start = (param.main_grad.data_ptr() - bucket.grad_data.data_ptr()) // itemsize
                        left, right = max(param_start, shard_start), min(param_start + param.numel(), shard_end)
                        if right > left:
                            name = self.names[id(param)] + f".slice{left - param_start}:{right - param_start}"
                            self.record("grad_post_reduce", name, shard[left - shard_start : right - shard_start])
            return result

        bucket_class.start_grad_sync = start
        bucket_class.finish_grad_sync = finish
        for chunk_index, chunk in enumerate(self.model):
            for name, module in chunk.named_modules():
                if type(module).__name__ == "LanguageModelEmbedding":
                    label = f"chunk{chunk_index}.{name}"

                    def embedding_input(module, args, kwargs, label=label):
                        for i, tensor in enumerate(args):
                            self.record("data_input", f"{label}.arg{i}", tensor)
                        for key, tensor in kwargs.items():
                            self.record("data_input", f"{label}.{key}", tensor)

                    self.hooks.append(module.register_forward_pre_hook(embedding_input, with_kwargs=True))

        original_step = self.optimizer.step

        @functools.wraps(original_step)
        def optimizer_step(*args, **kwargs):
            self._optimizer_parameters("optimizer_before")
            result = original_step(*args, **kwargs)
            self._optimizer_parameters("optimizer_after")
            return result

        self.optimizer.step = optimizer_step

    def _optimizer_parameters(self, phase):
        optimizers = getattr(self.optimizer, "chained_optimizers", [self.optimizer])
        for i, optimizer in enumerate(optimizers):
            base = getattr(optimizer, "optimizer", optimizer)
            for j, group in enumerate(base.param_groups):
                for k, param in enumerate(group["params"]):
                    self.record(phase, f"optim{i}.group{j}.param{k}", param)

    def begin(self, step):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Step lifecycle must stay outside graph capture")
        self.step = step
        self.pending = []
        self.seen = collections.Counter()
        self.projection_calls = 0

    def end(self, loss, grad_norm, skipped):
        # Capture-created digest tensors are replay outputs at the same addresses.
        # Resolve only after the complete forward/backward/optimizer step, never
        # while the collective or the graph is still writing them.
        torch.cuda.synchronize()
        if self.step == 1:
            self.upstream.finish_iteration()
        events = self.pending + self.captured
        for seq, (signature, digest) in enumerate(events):
            self._write(
                {
                    "kind": "boundary",
                    "rank": self.rank,
                    "iteration": self.step,
                    "seq": seq,
                    **signature,
                    "digest": [f"{v & 0xFFFFFFFFFFFFFFFF:016x}" for v in digest.cpu().tolist()],
                }
            )
        metrics = {key: float(value.detach().float().cpu()) for key, value in loss.items()}
        norm = float(grad_norm) if grad_norm is not None else None
        counts = dict(collections.Counter(signature["phase"] for signature, _ in events))
        self._write(
            {
                "kind": "metrics",
                "iteration": self.step,
                "rank": self.rank,
                "loss": metrics,
                "grad_norm": norm,
                "skipped": bool(skipped),
                "counts": counts,
            }
        )
        if self.saved_operands:
            torch.save(
                {key: value.cpu() for key, value in self.saved_operands.items()},
                self.directory / f"projection_operands_rank{self.rank:05d}_iter000001.pt",
            )
            self.saved_operands.clear()
        self.file.flush()
        os.fsync(self.file.fileno())
        if self.step == int(os.environ["DSV4_NUMERICS_STOP_AFTER"]):
            required = {
                "projection_input",
                "projection_output",
                "projection_grad_y",
                "grad_pre_reduce",
                "grad_post_reduce",
                "optimizer_before",
                "optimizer_after",
                "data_input",
            }
            if not required.issubset(counts):
                raise RuntimeError(f"Incomplete diagnostic coverage: {required - set(counts)}")
            self._write({"kind": "diagnostic_complete", "rank": self.rank, "iterations": self.step})
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()
            LOGGER.warning(
                "DSv4 numerical diagnostic completed %s iterations on rank %s; exiting before teardown",
                self.step,
                self.rank,
            )
            # All measurements are complete. Avoid the separately proven process-
            # group destructor hang; no exception or training error is suppressed.
            os._exit(0)


def begin(model: list, optimizer: object, step: int) -> None:
    """Arm the explicitly requested diagnostic, otherwise do nothing."""
    global _TRACE
    directory = os.environ.get("DSV4_NUMERICS_DIR")
    if not directory:
        return
    if _TRACE is None:
        _TRACE = _Trace(directory, model, optimizer)
    _TRACE.begin(step)


def end(loss: dict, grad_norm: float | None, skipped: int) -> None:
    """Flush completed-step evidence without changing numerical results."""
    if _TRACE is not None:
        _TRACE.end(loss, grad_norm, skipped)
