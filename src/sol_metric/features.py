"""Reusable quantile features, with optional disk-backed storage."""

from __future__ import annotations

import errno
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from ._typing import CachePath, FloatArray

SCHEMA = 2
PAGE_BYTES = 4096


def direction_block(n_quantiles: int, itemsize: int) -> int:
    """Directions per storage block: one memory page per document and block."""
    return max(1, PAGE_BYTES // (n_quantiles * itemsize))


class Features:
    """Empirical quantile bin averages and their provenance.

    The logical shape is (documents, directions, quantile bins). Features built
    by ``SOL`` are stored in direction blocks, (blocks, documents, block width,
    bins), because scoring reads a range of directions for every document: such
    a range is then one contiguous region in RAM or on disk, on any storage.

    Use ``SOL.featurize`` to construct these; ``load`` memory-maps the data.
    Cache directories are explicit immutable snapshots, never silently reused.
    Use as a context manager or call ``close`` before deleting a cache directory,
    particularly on Windows. Arrays must not be used after closing a cache.
    """

    def __init__(
        self, values: FloatArray, metadata: dict[str, Any], *, _owns_mapping: bool = False
    ) -> None:
        """Wrap a caller-owned (documents, directions, bins) array and copy metadata.

        Prefer ``SOL.featurize`` so direction and encoder metadata are complete.
        The constructor does not validate scientific provenance or array shape.
        """
        self._data: FloatArray | None = values
        self._width: int | None = None
        self._n_directions = int(values.shape[1]) if values.ndim == 3 else 0
        self._owns_mapping = _owns_mapping
        self.metadata: dict[str, Any] = json.loads(json.dumps(metadata, allow_nan=False))

    @classmethod
    def _blocked(
        cls,
        blocks: FloatArray,
        n_directions: int,
        metadata: dict[str, Any],
        *,
        owns_mapping: bool = False,
    ) -> Features:
        features = cls(blocks, metadata, _owns_mapping=owns_mapping)
        features._width, features._n_directions = int(blocks.shape[2]), n_directions
        return features

    @classmethod
    def _create(
        cls,
        shape: tuple[int, int, int],
        dtype: Any,
        metadata: dict[str, Any],
        path: CachePath | None,
        fill: Callable[[Callable[[int, int, int, FloatArray], None]], None],
    ) -> Features:
        """Allocate direction-block storage, fill it, and return the features.

        ``fill`` receives ``write(first_row, last_row, first_direction, values)``,
        where ``values`` has shape (rows, directions, bins). With ``path``, the
        result is an atomic disk cache that is reopened read-only.
        """
        n, n_directions, n_quantiles = shape
        width = direction_block(n_quantiles, np.dtype(dtype).itemsize)
        blocked = (-(-n_directions // width), n, width, n_quantiles)
        if path is None:
            blocks = np.zeros(blocked, dtype=dtype)
            fill(_writer(blocks))
            return cls._blocked(blocks, n_directions, metadata)
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Writing a memory map onto a full disk crashes the process, so check first.
        needed = math.prod(blocked) * np.dtype(dtype).itemsize
        free = shutil.disk_usage(path.parent).free
        if needed > free:
            raise OSError(
                errno.ENOSPC,
                f"Features need {needed / 1024**2:.0f} MiB, but {free / 1024**2:.0f} MiB are free",
                str(path.parent),
            )
        with tempfile.TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as tmp:
            stage = Path(tmp) / "cache"
            stage.mkdir()
            blocks = np.lib.format.open_memmap(
                stage / "values.npy", mode="w+", dtype=dtype, shape=blocked
            )
            try:
                fill(_writer(blocks))
                blocks.flush()
                cls._blocked(blocks, n_directions, metadata)._write_metadata(stage)
            finally:
                getattr(blocks, "_mmap").close()
                del blocks
            os.rename(stage, path)
        return cls.load(path)

    def _array(self) -> FloatArray:
        if self._data is None:
            raise ValueError("Features are closed")
        return self._data

    @property
    def shape(self) -> tuple[int, int, int]:
        """Logical shape (documents, directions, quantile bins)."""
        data = self._array()
        if self._width is None:
            return (int(data.shape[0]), int(data.shape[1]), int(data.shape[2]))
        return (int(data.shape[1]), self._n_directions, int(data.shape[3]))

    @property
    def dtype(self) -> np.dtype[Any]:
        """Storage dtype: float64 for float64 execution, float32 otherwise."""
        return self._array().dtype

    def __len__(self) -> int:
        return self.shape[0]

    def block(self, start: int, stop: int) -> FloatArray:
        """Read directions ``start:stop`` for every document.

        Returns an array shaped (documents, stop - start, bins). For features
        built by ``SOL``, this reads one contiguous region of the storage.
        """
        data = self._array()
        if not 0 <= start <= stop <= self.shape[1]:
            raise ValueError("Direction range is outside the features")
        if self._width is None:
            return np.asarray(data[:, start:stop])
        first, last = start // self._width, -(-stop // self._width)
        chunk = np.array(data[first:last])  # One sequential read.
        n, bins = chunk.shape[1], chunk.shape[3]
        offset = first * self._width
        logical = chunk.transpose(1, 0, 2, 3).reshape(n, -1, bins)
        return logical[:, start - offset : stop - offset]

    @property
    def values(self) -> FloatArray:
        """All features as one (documents, directions, quantile bins) array.

        For features built by ``SOL`` this assembles a copy in host RAM; use
        ``block`` to read direction ranges of large caches. Raises ValueError
        after closing.
        """
        data = self._array()
        return data if self._width is None else self.block(0, self._n_directions)

    @property
    def closed(self) -> bool:
        """Whether this object has released its array and owned file mapping."""
        return self._data is None

    def close(self) -> None:
        """Release this object's data and any file mapping it opened.

        Repeated calls are harmless. Arrays passed directly to the constructor
        remain owned by the caller; their underlying mappings are not closed.
        """
        if self.closed:
            return
        if self._owns_mapping:
            getattr(self._data, "_mmap").close()
        self._data = None

    def __enter__(self) -> Features:
        if self.closed:
            raise ValueError("Features are closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def save(self, path: CachePath) -> Features:
        """Save this object's values and metadata as a new cache directory.

        Args:
            path: A destination that does not already exist; parents are created.

        Returns:
            A new read-only, disk-backed Features object. The source stays open.

        Raises:
            FileExistsError: The destination exists.
            ValueError: The source is closed or metadata is invalid JSON.
            TypeError: Metadata contains objects JSON cannot serialize.
            OSError: Cache creation, writing, or reopening fails.
        """
        n, n_directions, _ = self.shape
        step = 1024

        def fill(write):
            for start in range(0, n_directions, step):
                write(0, n, start, self.block(start, min(n_directions, start + step)))

        return Features._create(self.shape, self.dtype, self.metadata, path, fill)

    def _write_metadata(self, path):
        data = self._array()
        payload = {
            "schema": SCHEMA,
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "layout": {"direction_block": self._width, "n_directions": self._n_directions},
            "metadata": self.metadata,
        }
        (Path(path) / "metadata.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: CachePath) -> Features:
        """Open an existing cache read-only without loading it into RAM.

        Args:
            path: Directory containing ``metadata.json`` and ``values.npy``.

        Returns:
            A Features object that owns its mapping. Close it before removing
            the directory; arrays must not be used after closing.

        Raises:
            ValueError: Invalid JSON, unsupported schema, or shape/dtype mismatch.
            KeyError: Required cache metadata is missing.
            OSError: The cache files cannot be read.
        """
        path = Path(path)
        payload = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA:
            raise ValueError("Unsupported SOL feature-cache schema; recompute the features")
        layout = payload["layout"]
        values = np.load(path / "values.npy", mmap_mode="r", allow_pickle=False)
        try:
            width, n_directions = layout["direction_block"], layout["n_directions"]
            if (
                values.ndim != 4
                or list(values.shape) != payload["shape"]
                or str(values.dtype) != payload["dtype"]
                or values.dtype.kind != "f"
                or values.shape[2] != width
                or values.shape[0] != -(-n_directions // width)
            ):
                raise ValueError("Feature cache shape or dtype does not match its metadata")
            return cls._blocked(values, n_directions, payload["metadata"], owns_mapping=True)
        except BaseException:
            values._mmap.close()
            raise


def _writer(blocks: FloatArray) -> Callable[[int, int, int, FloatArray], None]:
    """Write (rows, directions, bins) arrays into (blocks, documents, width, bins)."""
    width = blocks.shape[2]

    def write(first_row: int, last_row: int, start: int, values: FloatArray) -> None:
        stop, direction = start + values.shape[1], start
        while direction < stop:
            index, offset = divmod(direction, width)
            take = min(width - offset, stop - direction)
            chunk = values[:, direction - start : direction - start + take]
            blocks[index, first_row:last_row, offset : offset + take] = chunk
            direction += take

    return write
