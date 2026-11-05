# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Bounded first-forward SHA256 observations; intentionally synchronization-heavy.

This is diagnostic-only, not a performance measurement or a transparent observer.
Original values/gradients are returned unchanged. No added distributed collective.
"""

import collections
import dataclasses
import functools
import hashlib
import inspect
import json
import os
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import torch


def tensor_signature(tensor: torch.Tensor) -> dict[str, object]:
    """Hash logical raw bytes, including initialized rowwise MXFP8 payload/scales.

    Non-rowwise quantized storage is explicitly outside this forward observation.
    No dequantization is used as a substitute for raw quantized bytes.
    """
    result = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "tensor_type": f"{type(tensor).__module__}.{type(tensor).__name__}",
    }
    if hasattr(tensor, "get_metadata"):
        metadata = tensor.get_metadata()
        result["storage"] = {}
        payload, scales = metadata.get("rowwise_data"), metadata.get("rowwise_scale_inv")
        if (
            "MXFP8" in type(tensor).__name__
            and isinstance(payload, torch.Tensor)
            and payload.numel() == tensor.numel()
        ):
            result["storage"]["rowwise_data"] = tensor_signature(payload)
            # MXFP8 scales are padded in both dimensions. Padding bytes are not
            # numerical operands and must not manufacture first differences.
            rows = tensor.numel() // tensor.shape[-1]
            blocks = (tensor.shape[-1] + 31) // 32
            if (
                isinstance(scales, torch.Tensor)
                and scales.ndim == 2
                and scales.shape[0] >= rows
                and scales.shape[1] >= blocks
            ):
                result["storage"]["rowwise_scale_inv"] = tensor_signature(scales[:rows, :blocks])
        result["complete"] = set(result["storage"]) == {"rowwise_data", "rowwise_scale_inv"}
        result["scope"] = "MXFP8 rowwise payload/valid scales only; columnwise cache and padding excluded"
        return result
    if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
        result.update(complete=False, reason="Unsupported tensor subclass; no silent dequantization")
        return result
    with torch.no_grad():
        value = tensor.detach().resolve_conj().resolve_neg().contiguous()
        raw = value.reshape(-1).view(torch.uint8).cpu().numpy()
        result.update(complete=True, nbytes=raw.nbytes, sha256=hashlib.sha256(memoryview(raw)).hexdigest())
    return result


def signature(value: object) -> object:
    """Describe a boundary without treating unobserved opaque inputs as matches."""
    if isinstance(value, torch.Tensor):
        return tensor_signature(value)
    if isinstance(value, (tuple, list)):
        return [signature(item) for item in value]
    if isinstance(value, dict):
        return {str(key): signature(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: signature(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"opaque_type": f"{type(value).__module__}.{type(value).__name__}", "complete": False}


class UpstreamTrace:
    """Record named boundary SHA256s only through the first three projections."""

    def __init__(self, directory: Path, rank: int, model: list[torch.nn.Module], attention: ModuleType) -> None:
        self.directory = Path(directory)
        self.rank = rank
        self.active = True
        self.completed = False
        self.seq = 0
        self.seen: collections.Counter[tuple[str, str]] = collections.Counter()
        self.projections: list[int] = []
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self.restores: list[tuple[ModuleType, str, Callable[..., object]]] = []
        self.file = (self.directory / f"upstream_rank{rank:05d}.jsonl").open("x", buffering=1)
        self._write(
            {
                "kind": "upstream_runtime",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "attention_sha256": hashlib.sha256(Path(inspect.getfile(attention)).read_bytes()).hexdigest(),
                "observer": "synchronous logical raw-byte SHA256; timing/race perturbation",
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU fixture",
                "torch": torch.__version__,
            }
        )
        for chunk, root in enumerate(model):
            for name, module in root.named_modules():
                if not name:
                    continue
                label = f"chunk{chunk}.{name}"
                self.hooks.append(module.register_forward_pre_hook(self._pre(label), with_kwargs=True))
                self.hooks.append(module.register_forward_hook(self._post(label), with_kwargs=True))
        for name in ("apply_rotary_pos_emb", "fused_mla_rope_out_of_place"):
            original = getattr(attention, name, None)
            if original is not None:
                self.restores.append((attention, name, original))
                setattr(attention, name, self._wrap(name, original))

    def _write(self, record: dict[str, object]) -> None:
        self.file.write(
            json.dumps({"schema": "dsv4-upstream-sha256-v1", "rank": self.rank, **record}, allow_nan=False) + "\n"
        )

    def _record(self, phase: str, name: str, value: object) -> None:
        if not self.active:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Synchronous upstream observations must never enter CUDA graph capture")
        identity = (phase, name)
        occurrence = self.seen[identity]
        self.seen[identity] += 1
        self._write(
            {
                "kind": "upstream_boundary",
                "iteration": 1,
                "execution": "eager",
                "seq": self.seq,
                "phase": phase,
                "name": name,
                "occurrence": occurrence,
                "value": signature(value),
            }
        )
        self.seq += 1

    def _pre(self, name: str) -> Callable[..., None]:
        def hook(
            module: torch.nn.Module,
            args: tuple[object, ...] | None,
            kwargs: dict[str, object] | None = None,
        ) -> None:
            if self.active:
                if args is None or kwargs is None:
                    # TEGroupedMLP emulates child pre-hooks as hook(child, None).
                    # No intermediate input/output is materialized on that path.
                    # Accept the callback without inventing an observed input.
                    inputs: object = {
                        "complete": False,
                        "reason": "Emulated pre-hook has no materialized input/kwargs or child output boundary",
                    }
                else:
                    inputs = {"args": args, "kwargs": kwargs}
                self._record("module_input", name, inputs)
                self._record("module_parameters", name, dict(module.named_parameters(recurse=False)))

        return hook

    def _post(self, name: str) -> Callable[..., None]:
        def hook(module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object], output: object) -> None:
            self._record("module_output", name, output)

        return hook

    def _wrap(self, name: str, original: Callable[..., object]) -> Callable[..., object]:
        @functools.wraps(original)
        def call(*args: object, **kwargs: object) -> object:
            self._record("function_input", name, {"args": args, "kwargs": kwargs})
            result = original(*args, **kwargs)
            self._record("function_output", name, result)
            return result

        return call

    def projection_input(self, index: int, x: torch.Tensor, weight: torch.Tensor) -> None:
        """Observe canonical inputs while retaining no extra GPU tensors."""
        if not self.active:
            return
        if index != len(self.projections) or index > 2:
            raise RuntimeError("Unexpected first-forward projection ordering")
        self.projections.append(index)
        self._record("projection_input", f"projection{index:04d}", {"x": x, "weight": weight})
        if self.rank in (0, 4, 31):
            self._save(index, "inputs", {"x": x, "weight": weight})

    def projection_output(self, index: int, output: torch.Tensor) -> None:
        """Close the bounded observation after the third projection result."""
        if not self.active:
            return
        self._record("projection_output", f"projection{index:04d}", output)
        if self.rank in (0, 4, 31):
            self._save(index, "output", {"output": output})
        if index == 2:
            self._write(
                {
                    "kind": "upstream_complete",
                    "iteration": 1,
                    "projections": self.projections,
                    "boundary_records": self.seq,
                }
            )
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()
            self.active = False
            self.completed = True

    def _save(self, index: int, phase: str, tensors: dict[str, torch.Tensor]) -> None:
        path = self.directory / f"upstream_operands_rank{self.rank:05d}_projection{index:04d}_{phase}.pt"
        if path.exists():
            raise FileExistsError(path)
        # These are own tensor dictionaries, never uploaded to the GitLab archive.
        torch.save({name: tensor.detach().cpu() for name, tensor in tensors.items()}, path)

    def finish_iteration(self) -> None:
        """Require complete first-forward evidence, then remove observer hooks."""
        if not self.completed:
            raise RuntimeError("The first forward did not reach three projections")
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        for module, name, original in self.restores:
            setattr(module, name, original)
        self.restores.clear()
