"""NumPy and optional PyTorch execution; no custom GPU extensions."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import cast

import numpy as np

from ._math import bin_average_plan
from ._typing import FloatArray


def torch_module():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install 'sol-metric[torch]' and a suitable PyTorch build.") from exc
    return torch


def resolve_device(device):
    if device == "cpu":
        return "cpu"
    if device == "auto":
        try:
            torch = torch_module()
        except ImportError:
            return "cpu"
        return f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    torch = torch_module()
    parsed = torch.device(device)
    if parsed.type != "cuda":
        raise ValueError("Supported devices are 'auto', 'cpu', 'cuda', and 'cuda:N'")
    if not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    index = torch.cuda.current_device() if parsed.index is None else parsed.index
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is unavailable")
    return f"cuda:{index}"


def available_memory(
    proc: Path = Path("/proc"), cgroup_root: Path = Path("/sys/fs/cgroup")
) -> int | None:
    """Bytes of RAM this process can still use, or None when unknown.

    Linux reports MemAvailable, capped by the memory limits of the process's
    cgroup and its ancestors, which SLURM jobs and containers set. Inactive
    file cache is reclaimable and does not count as used. Other systems: None.
    """
    try:
        meminfo = (proc / "meminfo").read_text()
    except OSError:
        return None
    match = re.search(r"^MemAvailable:\s+(\d+) kB$", meminfo, re.MULTILINE)
    if match is None:
        return None
    available = int(match[1]) * 1024
    for limit, usage in _cgroup_memory(proc, cgroup_root):
        available = min(available, max(0, limit - usage))
    return available


def _cgroup_memory(proc, root):
    """Yield (limit, usage) bytes for each memory-limited cgroup level of this process."""
    try:
        lines = (proc / "self/cgroup").read_text().splitlines()
    except OSError:
        return
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if controllers == "":
            base, names = root, ("memory.max", "memory.current", "inactive_file")
        elif "memory" in controllers.split(","):
            names = ("memory.limit_in_bytes", "memory.usage_in_bytes", "total_inactive_file")
            base = root / "memory"
        else:
            continue
        level = base / path.strip("/")
        while True:
            reading = _cgroup_level(level, *names)
            if reading is not None:
                yield reading
            if level == base or level == level.parent:
                break
            level = level.parent


def _cgroup_level(level, limit_name, usage_name, inactive_name):
    try:
        limit = (level / limit_name).read_text().strip()
        if limit == "max":
            return None
        result = int(limit), int((level / usage_name).read_text())
    except (OSError, ValueError):
        return None
    try:
        stat = (level / "memory.stat").read_text()
    except OSError:
        return result
    match = re.search(rf"^{inactive_name} (\d+)$", stat, re.MULTILINE)
    return result[0], max(0, result[1] - (int(match[1]) if match else 0))


def ram_backed(path: str | os.PathLike[str], mounts: Path = Path("/proc/self/mounts")) -> bool:
    """Whether path is on a file system held in RAM (tmpfs or ramfs); Linux only."""
    try:
        lines = mounts.read_text().splitlines()
    except OSError:
        return False
    target, best, kind = os.path.realpath(path), "", ""
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        # /proc/mounts escapes spaces and other separators as octal triplets.
        point = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[1])
        inside = target == point or target.startswith(point.rstrip("/") + "/")
        if inside and len(point) >= len(best):
            best, kind = point, fields[2]
    return kind in ("tmpfs", "ramfs")


class Backend:
    def __init__(self, device, dtype):
        self.device = resolve_device(device)
        self.dtype_name = dtype
        if dtype not in ("float32", "float64", "float16", "bfloat16"):
            raise ValueError("dtype must be float32, float64, float16, or bfloat16")
        self.torch = None if self.device == "cpu" else torch_module()
        if self.torch is None:
            if dtype not in ("float32", "float64"):
                raise ValueError("CPU computation supports float32 and float64")
            self.dtype = np.dtype(dtype)
        else:
            self.dtype = getattr(self.torch, dtype)
            if dtype == "bfloat16":
                with self.torch.cuda.device(self.device):
                    if not self.torch.cuda.is_bf16_supported():
                        raise ValueError("This GPU does not support bfloat16; use float32")
        self.storage_dtype = np.float64 if dtype == "float64" else np.float32
        self.itemsize = np.dtype(self.storage_dtype).itemsize

    def array(self, value, *, full_precision=False):
        if self.torch is None:
            if not isinstance(value, np.ndarray) and hasattr(value, "detach"):
                # NumPy cannot represent Torch bfloat16. Convert before crossing
                # the boundary, preserving float64 when it was requested.
                torch = torch_module()
                dtype = getattr(torch, np.dtype(self.storage_dtype).name)
                value = value.detach().to(device="cpu", dtype=dtype).numpy()
            return np.asarray(value, dtype=self.storage_dtype)
        dtype = (
            getattr(self.torch, np.dtype(self.storage_dtype).name) if full_precision else self.dtype
        )
        if isinstance(value, np.ndarray):
            # Memmaps may be read-only, and NumPy views may have negative strides.
            if not value.flags.writeable or any(stride < 0 for stride in value.strides):
                value = np.array(value, copy=True, order="C")
            value = self.torch.from_numpy(value)
        return value.detach().to(device=self.device, dtype=dtype, non_blocking=True)

    def empty(self, shape):
        if self.torch is None:
            return np.empty(shape, dtype=self.storage_dtype)
        return self.torch.empty(
            shape, device=self.device, dtype=getattr(self.torch, np.dtype(self.storage_dtype).name)
        )

    def host(self, value):
        if self.torch is None:
            return np.asarray(value)
        return value.detach().cpu().numpy()

    def quantiles(self, tokens, offsets, directions, count):
        """Only one document batch and one direction block are materialized."""
        lens = np.diff(offsets)
        n, longest, width = len(lens), int(lens.max()), len(directions)
        values = self.array(tokens)
        u = self.array(directions)
        aligned = np.all(lens == longest) and longest % count == 0
        if self.torch is None:
            projected = values @ u.T
            padded = np.full((n, width, longest), np.inf, dtype=self.storage_dtype)
            for i, length in enumerate(lens):
                padded[i, :, :length] = projected[offsets[i] : offsets[i + 1]].T
            padded.sort(axis=-1)
            if aligned:
                return padded.reshape(n, width, count, longest // count).mean(axis=-1)
            indices, weights = bin_average_plan(lens, count)
            selected = cast(
                FloatArray, np.take_along_axis(padded, indices.reshape(n, 1, -1), axis=-1)
            )
            selected = selected.reshape(n, width, *indices.shape[1:])
            selected *= weights.astype(self.storage_dtype)[:, None]
            return selected.sum(axis=-1)
        torch = self.torch
        projected = (values @ u.T).to(getattr(torch, np.dtype(self.storage_dtype).name))
        if np.all(lens == longest):
            padded = projected.reshape(n, longest, width).transpose(1, 2).contiguous()
        else:
            padded = torch.full(
                (n, width, longest), float("inf"), device=self.device, dtype=projected.dtype
            )
            rows_np = np.repeat(np.arange(n), lens)
            positions_np = np.arange(len(tokens)) - offsets[:-1][rows_np]
            rows = torch.from_numpy(rows_np).to(self.device, non_blocking=True)
            positions_t = torch.from_numpy(positions_np).to(self.device, non_blocking=True)
            padded[rows, :, positions_t] = projected
        ordered = torch.sort(padded, dim=-1).values
        if aligned:
            return ordered.reshape(n, width, count, longest // count).mean(dim=-1)
        indices, weights = bin_average_plan(lens, count)
        index_t = torch.from_numpy(indices.reshape(n, 1, -1)).to(self.device)
        selected = torch.gather(ordered, 2, index_t.expand(n, width, -1))
        selected = selected.reshape(n, width, *indices.shape[1:])
        selected.mul_(self.array(weights, full_precision=True)[:, None])
        return selected.sum(dim=-1)

    def slice_sum(self, qx, qy, gp, transport):
        """Sort on the contiguous document axis, retaining all documents per slice."""
        m = qx.shape[-1]
        if self.torch is None:
            g = np.asarray(gp, dtype=self.storage_dtype)
            x = np.matmul(g, np.asarray(qx).transpose(1, 2, 0)).reshape(-1, len(qx)) / m
            y = np.matmul(g, np.asarray(qy).transpose(1, 2, 0)).reshape(-1, len(qy)) / m
            x.sort(axis=-1)
            y.sort(axis=-1)
            if transport is None:
                delta = x.astype(np.float64) - y
                return float(np.sum(delta * delta, dtype=np.float64) / len(qx))
            ix, iy, weights = transport
            delta = x[:, ix].astype(np.float64) - y[:, iy]
            return float(np.sum((delta * delta) * weights, dtype=np.float64))
        torch = self.torch
        g = self.array(gp)
        x = torch.matmul(g, qx.to(self.dtype).permute(1, 2, 0)).reshape(-1, len(qx))
        y = torch.matmul(g, qy.to(self.dtype).permute(1, 2, 0)).reshape(-1, len(qy))
        # Low precision is restricted to matrix multiplication. Quantiles, sorting,
        # subtraction, and squaring retain at least float32 precision.
        x = torch.sort(x.to(qx.dtype) / m, dim=-1).values
        y = torch.sort(y.to(qy.dtype) / m, dim=-1).values
        if transport is None:
            delta = x - y
            return delta.square().sum(dtype=torch.float64) / len(qx)
        ix, iy, weights = transport
        ix = torch.from_numpy(ix).to(self.device, non_blocking=True)
        iy = torch.from_numpy(iy).to(self.device, non_blocking=True)
        weights = torch.from_numpy(weights).to(self.device, non_blocking=True)
        delta = x[:, ix] - y[:, iy]
        return (delta.square() * weights).sum(dtype=torch.float64)
