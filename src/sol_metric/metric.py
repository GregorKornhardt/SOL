"""Public SOL metric and bounded-block execution."""

from __future__ import annotations

import math
import tempfile
import warnings
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from functools import cache, partial
from itertools import chain
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ._math import GP_NORMALIZATION, QUANTILE_CONVENTION, directions, gp_directions, transport_grid
from ._runtime import Backend, available_memory, ram_backed
from ._typing import CachePath, Encoder, FloatArray, Offsets, Precision, TokenMatrix
from ._version import __version__
from .features import Features


class _Unset:
    """Distinguish omitted options from the meaningful value None."""


_UNSET = _Unset()
# feature_memory_mb="auto" keeps features in RAM up to this share of the memory
# available when they are created, or the fallback where that is unknown.
_AUTO_FEATURE_SHARE = 0.5
_AUTO_FEATURE_FALLBACK_MB = 4096.0
# Directions binned per call while featurizing; wider blocks stop helping here.
_FEATURE_DIRECTIONS = 512
# Directions read per scoring step; GP draws per step fill slice_batch_size.
_SCORE_DIRECTIONS = 64


@contextmanager
def _out_of_memory(advice):
    """Turn NumPy and CUDA out-of-memory errors into MemoryError with advice."""
    try:
        yield
    except MemoryError as exc:
        raise MemoryError(f"Out of memory while {advice}") from exc
    except RuntimeError as exc:
        # torch.OutOfMemoryError, matched by name so Torch stays optional.
        if type(exc).__name__ != "OutOfMemoryError":
            raise
        raise MemoryError(f"Out of GPU memory while {advice}") from exc


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class SOLConfig:
    """Metric parameters, independent of memory/performance settings.

    ``scale=None`` uses the hidden dimension, as in the paper. An explicit
    positive scale multiplies the squared distance before the square root.
    Empirical transport and quantile-bin integration are always exact.
    First-level directions are unit vectors; second-level GP draws are raw,
    without per-function centering or norm normalization.
    """

    n_directions: int = 1024
    n_projections: int = 1024
    n_quantiles: int = 64
    lengthscale: float | None = 0.1
    seed: int = 0
    gp_seed: int = 0
    scale: float | None = None

    def __post_init__(self) -> None:
        for name in ("n_directions", "n_projections", "n_quantiles"):
            _positive_int(name, getattr(self, name))
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in ("seed", "gp_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        if self.lengthscale is not None and (
            not math.isfinite(self.lengthscale) or self.lengthscale <= 0
        ):
            raise ValueError("lengthscale must be positive and finite, or None")
        if self.scale is not None and (not math.isfinite(self.scale) or self.scale <= 0):
            raise ValueError("scale must be positive and finite")
        if self.scale is not None:
            object.__setattr__(self, "scale", float(self.scale))
        if self.lengthscale is not None:
            object.__setattr__(self, "lengthscale", float(self.lengthscale))


class SOL:
    """Compute the square-root DSW distance between corpora of token measures.

    Metric settings can be passed directly as keyword arguments. Omitted
    ``n_directions`` and ``n_projections`` use 8192 for Dream and 1024 for other
    encoders, resolved from encoder provenance when scoring or featurizing.
    Inputs without encoder provenance use 1024. Other defaults are
    ``n_quantiles=64``, ``lengthscale=0.1``, ``seed=0``, ``gp_seed=0`` and
    ``scale=None``.
    ``scale=None`` uses the hidden dimension; ``lengthscale=None`` uses
    independent Gaussian draws. An optional ``SOLConfig`` is also accepted;
    explicit metric keywords override its fields. Supplied configurations and
    explicit slice counts disable automatic defaults for those counts.

    CPU uses NumPy; CUDA uses optional PyTorch. Memory follows the batch sizes,
    which change execution only, never scores: ``batch_size`` and
    ``max_tokens_per_batch`` bound the texts per step, both for the default
    encoder's forward passes and for binning, and ``slice_batch_size`` the
    slices scored per step, about 25 bytes per document and slice. Lower them
    after an out-of-memory error. A ``TextEncoder`` passed explicitly keeps its
    own batch settings.
    ``feature_memory_mb`` bounds features kept in host RAM: ``from_texts`` keeps
    both corpora's features in RAM up to this size and writes larger ones to
    temporary files under ``temp_dir``. The default ``"auto"`` resolves to half
    of the memory available when features are created, respecting cgroup limits
    such as SLURM allocations, or 4096 MiB where that is unknown (non-Linux).
    Storage location never changes scores.

    Args:
        config: Optional immutable SOLConfig; fixes both automatic slice counts.
        n_directions: Positive token-direction count; inferred when omitted.
        n_projections: Positive GP-draw count; inferred when omitted.
        n_quantiles: Positive quantile-bin count, default 64.
        lengthscale: Positive finite RBF scale, default 0.1; None for white noise.
        seed: Nonnegative integer token-direction seed, default 0.
        gp_seed: Nonnegative integer GP-draw seed, default 0.
        scale: Positive finite squared-distance multiplier, or None for dimension.
        device: "auto", "cpu", "cuda", or an indexed CUDA device.
        dtype: Matrix-product precision, default float32; CPU supports float64
            too, CUDA additionally supports float16 and compatible bfloat16.
        batch_size: Positive maximum texts per step, default 8: per forward pass
            of the default encoder and per binning step.
        max_tokens_per_batch: Positive padded tokens per such step, default 8192.
            Longer documents are encoded and binned on their own.
        slice_batch_size: Positive slices (direction and GP pairs) scored per
            step, default 1024; memory is about 25 bytes per document and slice.
        feature_memory_mb: Host RAM in MiB for in-memory features, or "auto"
            (default) for half of the available memory; 0 always writes
            ``from_texts`` features to temporary files.
        temp_dir: Existing directory for temporary text features, or system default.

    Raises:
        TypeError: config is not a SOLConfig.
        ValueError: Invalid metric/execution setting, precision or device index.
        ImportError: Torch is missing when CUDA is explicitly requested.
        RuntimeError: CUDA is requested but unavailable.
    """

    def __init__(
        self,
        config: SOLConfig | None = None,
        *,
        n_directions: int | _Unset = _UNSET,
        n_projections: int | _Unset = _UNSET,
        n_quantiles: int | _Unset = _UNSET,
        lengthscale: float | None | _Unset = _UNSET,
        seed: int | _Unset = _UNSET,
        gp_seed: int | _Unset = _UNSET,
        scale: float | None | _Unset = _UNSET,
        device: str = "auto",
        dtype: Precision = "float32",
        batch_size: int = 8,
        max_tokens_per_batch: int = 8192,
        slice_batch_size: int = 1024,
        feature_memory_mb: float | Literal["auto"] = "auto",
        temp_dir: CachePath | None = None,
    ) -> None:
        if config is not None and not isinstance(config, SOLConfig):
            raise TypeError("config must be a SOLConfig")
        # None is meaningful for lengthscale and scale, so distinguish it from
        # an omitted keyword when applying overrides to an existing config.
        settings: dict[str, Any] = dict(
            n_directions=n_directions,
            n_projections=n_projections,
            n_quantiles=n_quantiles,
            lengthscale=lengthscale,
            seed=seed,
            gp_seed=gp_seed,
            scale=scale,
        )
        settings = {name: value for name, value in settings.items() if value is not _UNSET}
        self._auto_slices: tuple[str, ...] = ()
        self.config = SOLConfig(**settings) if config is None else replace(config, **settings)
        self._auto_slices = tuple(
            name
            for name in ("n_directions", "n_projections")
            if config is None and name not in settings
        )
        for name, value in (
            ("batch_size", batch_size),
            ("max_tokens_per_batch", max_tokens_per_batch),
            ("slice_batch_size", slice_batch_size),
        ):
            _positive_int(name, value)
        if feature_memory_mb != "auto" and (
            isinstance(feature_memory_mb, str)
            or not math.isfinite(feature_memory_mb)
            or feature_memory_mb < 0
        ):
            raise ValueError('feature_memory_mb must be "auto" or nonnegative and finite')
        self.backend = Backend(device, dtype)
        self.batch_size = batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.slice_batch_size = slice_batch_size
        self.feature_memory_mb = feature_memory_mb
        self.temp_dir = temp_dir
        self.last_run: dict[str, Any] | None = None
        self._default_encoder: Encoder | None = None

    @property
    def device(self) -> str:
        """Resolved execution device: ``cpu`` or an indexed CUDA device."""
        return self.backend.device

    @property
    def config(self) -> SOLConfig:
        """Effective settings, resolved for the most recent input encoder."""
        return self._config

    @config.setter
    def config(self, value: SOLConfig) -> None:
        if not isinstance(value, SOLConfig):
            raise TypeError("config must be a SOLConfig")
        self._config = value
        self._auto_slices = ()

    def _resolve_slices(self, provenance=None):
        if not self._auto_slices:
            return
        provenance = provenance if isinstance(provenance, dict) else {}
        model_type = str(provenance.get("model_type") or "").lower()
        model = str(provenance.get("model") or "").lower()
        # Model type also identifies local checkpoints; the ID fallback covers
        # older caches and custom encoders exposing only the Hugging Face ID.
        dream = model_type == "dream" if model_type else model.startswith("dream-org/dream-")
        count = 8192 if dream else 1024
        self._config = replace(self.config, **{name: count for name in self._auto_slices})

    def _tokens(self, tokens, offsets):
        if not hasattr(tokens, "shape") or len(tokens.shape) != 2 or min(tokens.shape) < 1:
            raise ValueError("tokens must be a nonempty (tokens, dimension) array or tensor")
        if isinstance(tokens, np.ndarray):
            if tokens.dtype.kind != "f":
                raise ValueError("tokens must have floating-point dtype")
        elif not hasattr(tokens, "is_floating_point") or not tokens.is_floating_point():
            raise ValueError("tokens must be a NumPy array or floating-point torch tensor")
        offsets = np.asarray(offsets)
        if (
            offsets.ndim != 1
            or len(offsets) < 2
            or offsets.dtype.kind not in "iu"
            or offsets[0] != 0
            or offsets[-1] != len(tokens)
            or np.any(offsets[1:] <= offsets[:-1])
        ):
            raise ValueError(
                "offsets must start at 0, end at len(tokens), and delimit nonempty documents"
            )
        for start in range(0, len(tokens), self.max_tokens_per_batch):
            block = tokens[start : start + self.max_tokens_per_batch]
            valid = (
                np.isfinite(block).all()
                if isinstance(block, np.ndarray)
                else block.isfinite().all().item()
            )
            if not valid:
                raise ValueError("tokens contain NaN or infinity")
        return offsets.astype(np.int64, copy=False)

    def _execution(self):
        """Block sizes for this run, from the batch settings; recorded in last_run."""
        cfg = self.config
        read = min(_SCORE_DIRECTIONS, cfg.n_directions, self.slice_batch_size)
        return {
            "batch_size": self.batch_size,
            "max_tokens_per_batch": self.max_tokens_per_batch,
            "slice_batch_size": self.slice_batch_size,
            "feature_directions": min(_FEATURE_DIRECTIONS, cfg.n_directions),
            "score_directions": read,
            "score_projections": max(1, min(cfg.n_projections, self.slice_batch_size // read)),
        }

    @staticmethod
    def _batches(offsets, plan):
        lengths = np.diff(offsets)
        start = 0
        while start < len(lengths):
            end, longest = start, 0
            while end < min(len(lengths), start + plan["batch_size"]):
                candidate = max(longest, int(lengths[end]))
                if (end - start + 1) * candidate > plan["max_tokens_per_batch"]:
                    break
                longest = candidate
                end += 1
            # A document beyond the token budget is binned on its own.
            end = max(end, start + 1)
            yield start, end
            start = end

    def _finish(self, total, plan, metadata=None, dimension=None):
        if hasattr(total, "item"):
            total = total.item()
        if not math.isfinite(total) or total < 0:
            raise FloatingPointError("Nonfinite SOL score; check input scale or use float64")
        scale = self.config.scale
        if scale is None:
            scale = dimension if dimension is not None else (metadata or {}).get("dimension")
            if scale is None:
                raise ValueError(
                    "Raw feature arrays need an explicit scale; use Features to infer dimension"
                )
        score = math.sqrt(scale * total / (self.config.n_directions * self.config.n_projections))
        if not math.isfinite(score):
            raise FloatingPointError("SOL scaling overflowed; reduce scale or input magnitude")
        self.last_run = {
            "sol_version": __version__,
            "score": score,
            "squared": False,
            "scale": scale,
            "quantile_convention": QUANTILE_CONVENTION,
            "gp_normalization": GP_NORMALIZATION,
            "config": asdict(self.config),
            "device": self.device,
            "dtype": self.backend.dtype_name,
            "execution": plan,
            "features": metadata,
        }
        self.last_run["numpy_version"] = np.__version__
        if self.backend.torch is not None:
            torch = self.backend.torch
            self.last_run["torch"] = {
                "version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(self.device),
                "matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            }
        return score

    def from_tokens(
        self,
        tokens_x: TokenMatrix,
        offsets_x: Offsets,
        tokens_y: TokenMatrix,
        offsets_y: Offsets,
        *,
        provenance: dict[str, Any] | None = None,
    ) -> float:
        """Score packed token matrices and document offsets.

        Optional encoder ``provenance`` selects automatic slice counts, as for
        ``featurize``. Without it, omitted counts use 1024. Each corpus is
        binned once; its features stay in RAM within ``feature_memory_mb`` or go
        to temporary files under ``temp_dir``, as for ``from_texts``.

        Args:
            tokens_x, tokens_y: Finite floating NumPy arrays or Torch tensors,
                shaped (total tokens, hidden dimension). Dimensions must match.
            offsets_x, offsets_y: Integer document boundaries, starting at zero
                and ending at the corresponding token count, strictly increasing.
            provenance: JSON-serializable encoder settings used for defaults.

        Returns:
            A finite, nonnegative distance. Populates ``last_run`` metadata.

        Raises:
            ValueError: Invalid tokens, offsets or dimensions.
            OSError: Temporary feature files do not fit on disk.
            FloatingPointError: Arithmetic or final scaling becomes nonfinite.
        """
        self.last_run = None
        ox, oy = self._tokens(tokens_x, offsets_x), self._tokens(tokens_y, offsets_y)
        if tokens_x.shape[1] != tokens_y.shape[1]:
            raise ValueError("Token embedding dimensions differ")
        self._resolve_slices(provenance)
        n = len(ox) + len(oy) - 2
        limit, _ = self._feature_limit()
        with tempfile.TemporaryDirectory(prefix="sol-features-", dir=self.temp_dir) as tmp:
            disk = self._feature_mib(n) > limit
            paths = [self._spill(n, Path(tmp) / side) if disk else None for side in "xy"]
            with (
                self._featurize(tokens_x, ox, paths[0], provenance) as fx,
                self._featurize(tokens_y, oy, paths[1], provenance) as fy,
            ):
                result = self.from_features(fx, fy)
        assert self.last_run is not None
        self.last_run["feature_storage"] = "disk" if disk else "memory"
        self.last_run["feature_memory_mb"] = float(limit)
        return result

    def _feature_metadata(self, dimension, provenance):
        return {
            "schema": 1,
            "dimension": dimension,
            "n_directions": self.config.n_directions,
            "n_quantiles": self.config.n_quantiles,
            "seed": self.config.seed,
            "quantile_convention": QUANTILE_CONVENTION,
            "dtype": self.backend.dtype_name,
            "provenance": {} if provenance is None else provenance,
        }

    def featurize(
        self,
        tokens: TokenMatrix,
        offsets: Offsets,
        *,
        path: CachePath | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> Features:
        """Prepare reusable reference or generated features.

        ``path`` writes an atomic disk-backed cache and returns a read-only
        ``Features`` object. Omit it to keep the full result in host RAM.
        Encoder ``provenance`` selects automatic slice counts.

        Args:
            tokens: Finite floating NumPy array or Torch tensor shaped
                (total tokens, hidden dimension).
            offsets: Strictly increasing integer document boundaries from zero
                to the total token count.
            path: New cache directory, or None for a host-RAM result.
            provenance: JSON-serializable encoder settings stored in the cache.

        Returns:
            Features shaped (documents, n_directions, n_quantiles), with
            float64 storage for float64 execution and float32 otherwise.

        Raises:
            ValueError: Invalid input or non-serializable/nonfinite provenance.
            TypeError: Provenance contains values JSON cannot serialize.
            FileExistsError: The cache destination already exists.
            MemoryError: In-memory features exceed feature_memory_mb, or binning
                runs out of memory; lower batch_size or max_tokens_per_batch.
            OSError: Cache creation or disk storage fails.
        """
        bounds = self._tokens(tokens, offsets)
        self._resolve_slices(provenance)
        if path is None:
            self._check_resident(len(bounds) - 1)
        return self._featurize(tokens, bounds, path, provenance)

    def _featurize(self, tokens, bounds, path, provenance):
        """Bin validated token states batch by batch into Features."""
        cfg = self.config
        n, d = len(bounds) - 1, tokens.shape[1]
        plan = self._execution()
        u = self.backend.array(directions(cfg.n_directions, d, cfg.seed))
        metadata = self._feature_metadata(d, provenance)

        def fill(write):
            for first_doc, last_doc in self._batches(bounds, plan):
                first, last = int(bounds[first_doc]), int(bounds[last_doc])
                batch_offsets = bounds[first_doc : last_doc + 1] - first
                self._bin_batch(tokens[first:last], batch_offsets, u, plan, write, first_doc)

        shape = (n, cfg.n_directions, cfg.n_quantiles)
        return Features._create(shape, self.backend.storage_dtype, metadata, path, fill)

    def _bin_batch(self, tokens, offsets, u, plan, write, row):
        """Move one document batch to the device once, then bin every direction block."""
        with _out_of_memory("binning; lower batch_size or max_tokens_per_batch"):
            self._bin_batch_unchecked(tokens, offsets, u, plan, write, row)

    def _bin_batch_unchecked(self, tokens, offsets, u, plan, write, row):
        values = self.backend.array(tokens)
        for start in range(0, len(u), plan["feature_directions"]):
            block = u[start : start + plan["feature_directions"]]
            q = self.backend.quantiles(values, offsets, block, self.config.n_quantiles)
            write(row, row + len(offsets) - 1, start, self.backend.host(q))

    def _feature_mib(self, n):
        cfg = self.config
        return n * cfg.n_directions * cfg.n_quantiles * self.backend.itemsize / 1024**2

    def _feature_limit(self):
        """Resolve feature_memory_mb now: (MiB, description for messages)."""
        if self.feature_memory_mb != "auto":
            return self.feature_memory_mb, f"feature_memory_mb={self.feature_memory_mb:g}"
        available = available_memory()
        if available is None:
            mib = _AUTO_FEATURE_FALLBACK_MB
            return mib, f'feature_memory_mb="auto" ({mib:.0f} MiB; available memory unknown)'
        mib = _AUTO_FEATURE_SHARE * available / 1024**2
        return mib, f'feature_memory_mb="auto" ({mib:.0f} MiB, half the available memory)'

    def _spill(self, n, path):
        """Return the temporary feature path for n documents, warning about tmpfs."""
        if ram_backed(Path(path).parent):
            warnings.warn(
                f"Writing {self._feature_mib(n):.0f} MiB of features to "
                f"{Path(path).parent}, a RAM-backed file system (tmpfs), does not "
                "free RAM; pass temp_dir=... or set TMPDIR to a directory on disk",
                RuntimeWarning,
                stacklevel=4,
            )
        return path

    def _check_resident(self, n, limit=None):
        """Raise before computing when n documents' features exceed feature_memory_mb."""
        mib, setting = (limit or self._feature_limit)()
        if self._feature_mib(n) > mib:
            raise MemoryError(
                f"In-memory features need {self._feature_mib(n):.0f} MiB, above "
                f"{setting}; pass path=... for a disk cache or raise feature_memory_mb"
            )

    def from_features(
        self, features_x: Features | FloatArray, features_y: Features | FloatArray
    ) -> float:
        """Score reusable features or raw (documents, directions, quantiles) arrays.

        Raw arrays must contain empirical quantile bin averages and use the same
        encoder and inner directions;
        provenance checks are possible only for ``Features`` objects.

        Args:
            features_x, features_y: Two open Features objects with identical
                metadata, or two finite floating NumPy arrays shaped
                (documents, n_directions, n_quantiles). Raw arrays require an
                explicit ``scale`` because they do not carry a hidden dimension.

        Returns:
            A finite, nonnegative distance. Populates ``last_run`` metadata.

        Raises:
            TypeError: Mixing Features objects and raw arrays.
            ValueError: Closed caches, incompatible metadata, invalid arrays,
                or raw arrays without an explicit scale.
            MemoryError: Scoring runs out of memory; lower slice_batch_size.
            FloatingPointError: Arithmetic or final scaling becomes nonfinite.
        """
        self.last_run = None
        metadata = None
        if isinstance(features_x, Features) or isinstance(features_y, Features):
            if not isinstance(features_x, Features) or not isinstance(features_y, Features):
                raise TypeError("Pass two Features objects or two raw arrays")
            if features_x.metadata != features_y.metadata:
                raise ValueError(
                    "Feature provenance differs (encoder, precision, directions, or quantiles)"
                )
            metadata = features_x.metadata
            if metadata.get("quantile_convention") != QUANTILE_CONVENTION:
                raise ValueError(
                    "Feature quantile convention is not empirical bin averages; "
                    "recompute features from token states"
                )
            self._resolve_slices(metadata.get("provenance"))
            for key in ("n_directions", "n_quantiles", "seed"):
                if metadata.get(key) != getattr(self.config, key):
                    raise ValueError(f"Feature {key} does not match the metric configuration")
        else:
            self._resolve_slices()
        cfg = self.config
        for q in (features_x, features_y):
            if (
                not isinstance(q, (Features, np.ndarray))
                or len(q.shape) != 3
                or q.shape[0] < 1
                or tuple(q.shape[1:]) != (cfg.n_directions, cfg.n_quantiles)
                or q.dtype.kind != "f"
            ):
                raise ValueError(
                    "Features must be floating arrays shaped (documents, n_directions, n_quantiles)"
                )
        # The default scale is the hidden dimension, which only Features metadata
        # records. Check it before sampling directions or scoring any slice.
        if cfg.scale is None and not (metadata or {}).get("dimension"):
            raise ValueError(
                "Raw feature arrays need an explicit scale; use Features to infer dimension"
            )
        nx, ny = features_x.shape[0], features_y.shape[0]
        plan = self._execution()
        g = gp_directions(cfg.n_projections, cfg.n_quantiles, cfg.lengthscale, cfg.gp_seed)
        transport = transport_grid(nx, ny)

        def read(q, start, stop):
            # Features read each direction range as one contiguous storage region.
            return q.block(start, stop) if isinstance(q, Features) else q[:, start:stop]

        total = 0.0
        for start in range(0, cfg.n_directions, plan["score_directions"]):
            stop = min(start + plan["score_directions"], cfg.n_directions)
            bx, by = read(features_x, start, stop), read(features_y, start, stop)
            if not np.isfinite(bx).all() or not np.isfinite(by).all():
                raise ValueError("Features contain NaN or infinity")
            qx, qy = (
                self.backend.array(bx, full_precision=True),
                self.backend.array(by, full_precision=True),
            )
            with _out_of_memory("scoring; lower slice_batch_size"):
                for k in range(0, cfg.n_projections, plan["score_projections"]):
                    total += self.backend.slice_sum(
                        qx, qy, g[k : k + plan["score_projections"]], transport
                    )
            del qx, qy
        return self._finish(total, plan, metadata)

    def featurize_texts(
        self,
        texts: Iterable[str],
        *,
        encoder: Encoder | None = None,
        path: CachePath | None = None,
    ) -> Features:
        """Encode a corpus once, then build reusable quantile features.

        Args:
            texts: Nonempty iterable of nonempty documents; generators work.
            encoder: TextEncoder or a custom streaming Encoder implementation;
                None uses a default Dream 7B TextEncoder with the metric's device and batch sizes,
                created on first use and reused by this metric.
            path: New cache directory, or None to retain features in host RAM.

        Returns:
            Features with encoder provenance and shape
            (documents, n_directions, n_quantiles).

        Raises:
            TypeError: A single string is passed instead of a corpus.
            ValueError: The corpus or yielded embeddings are invalid.
            ImportError: Optional dependencies required by the encoder are absent.
            FileExistsError: The cache destination already exists.
            MemoryError: In-memory features exceed feature_memory_mb, or binning
                runs out of memory; lower batch_size or max_tokens_per_batch.
            OSError: Model loading or temporary/cache disk storage fails.

        Token states are binned as the encoder yields each batch and are never
        stored. Encoder errors propagate. See ``featurize`` for cache details.
        """
        return self._stream_features(texts, self._text_encoder(encoder), path)

    def _text_encoder(self, encoder):
        """Return ``encoder``, or a default Dream TextEncoder created once per metric.

        The default encoder uses this metric's ``batch_size`` and
        ``max_tokens_per_batch`` for its forward passes.
        """
        if encoder is not None:
            return encoder
        if self._default_encoder is None:
            from .text import TextEncoder

            self._default_encoder = TextEncoder(
                device=self.device,
                batch_size=self.batch_size,
                max_tokens_per_batch=self.max_tokens_per_batch,
            )
        return self._default_encoder

    def _stream_features(self, texts, encoder, path, *, spill=None, others=0, limit=None):
        """Turn each batch of encoded documents into features before encoding more.

        With ``spill``, features stay in host RAM when this corpus plus ``others``
        documents fit ``feature_memory_mb``; otherwise they go to a cache at ``spill``.
        ``limit`` resolves that size, by default when the encoder has loaded.
        """
        from .text import encoded_states

        if path is not None and Path(path).exists():
            raise FileExistsError(path)
        # A TextEncoder on the metric's GPU hands over states without a host copy.
        n, states = encoded_states(texts, encoder, self.device if self.backend.torch else None)
        # The first document loads the encoder, which completes its provenance.
        first = next(states)
        provenance = encoder.provenance
        self._resolve_slices(provenance)
        limit = limit or self._feature_limit
        if path is None and spill is None:
            self._check_resident(n, limit)
        elif path is None and self._feature_mib(n + others) > limit()[0]:
            path = self._spill(n + others, spill)
        cfg, d = self.config, first.shape[1]
        budget = self.max_tokens_per_batch
        plan = self._execution()
        u = self.backend.array(directions(cfg.n_directions, d, cfg.seed))
        torch = self.backend.torch

        def fill(write):
            batch: list[Any] = []
            row = longest = 0

            def flush():
                offsets = np.r_[0, np.cumsum([len(x) for x in batch])]
                if torch is not None and torch.is_tensor(batch[0]):
                    tokens = torch.cat(batch)
                else:
                    tokens = np.concatenate(batch)
                self._bin_batch(tokens, offsets, u, plan, write, row)

            for document in chain([first], states):
                length = len(document)
                size = (len(batch) + 1) * max(longest, length)
                if batch and (len(batch) == plan["batch_size"] or size > budget):
                    flush()
                    row, batch, longest = row + len(batch), [], 0
                if length > budget:
                    # A document beyond the token budget is binned on its own.
                    self._bin_batch(document, np.array([0, length]), u, plan, write, row)
                    row += 1
                    continue
                batch.append(document)
                longest = max(longest, length)
            if batch:
                flush()

        shape = (n, cfg.n_directions, cfg.n_quantiles)
        metadata = self._feature_metadata(d, provenance)
        return Features._create(shape, self.backend.storage_dtype, metadata, path, fill)

    def from_texts(
        self, texts_x: Iterable[str], texts_y: Iterable[str], *, encoder: Encoder | None = None
    ) -> float:
        """Encode and compare two corpora, using one encoder for both.

        Args:
            texts_x, texts_y: Nonempty iterables of nonempty documents. Corpus
                sizes may differ; generators are consumed once.
            encoder: TextEncoder or a custom streaming Encoder implementation;
                None uses a default Dream 7B TextEncoder with the metric's device and batch sizes,
                created on first use and reused by this metric.

        Returns:
            A finite, nonnegative SOL distance and encoder metadata in
            ``last_run``. With the default scale this is sqrt(d) times DSW.

        Raises:
            TypeError: A single string is passed instead of a corpus.
            ValueError: Empty/invalid documents, embeddings or encoder settings.
            ImportError: Optional encoder dependencies are absent.
            MemoryError: Binning or scoring runs out of memory; lower the batch
                sizes named in the message.
            FloatingPointError: Arithmetic or final scaling becomes nonfinite.
            OSError: Model loading or temporary disk storage fails.

        Token states are binned batch by batch and never stored. Both corpora's
        features stay in host RAM when they fit ``feature_memory_mb``, resolved
        once after the encoder has loaded; larger ones go to temporary files
        under ``temp_dir``, removed on exit, including exceptions.
        ``last_run["feature_storage"]`` records which, and
        ``last_run["feature_memory_mb"]`` the resolved limit.
        Truncation, padding and precision follow the encoder settings.
        """
        from .text import as_corpus

        self.last_run = None
        corpus_x, corpus_y = as_corpus(texts_x), as_corpus(texts_y)
        encoder = self._text_encoder(encoder)
        # Both corpora share one decision, made once the encoder occupies memory.
        limit = cache(self._feature_limit)
        with tempfile.TemporaryDirectory(prefix="sol-features-", dir=self.temp_dir) as tmp:
            stream = partial(self._stream_features, encoder=encoder, path=None, limit=limit)
            with (
                stream(corpus_x, spill=Path(tmp) / "x", others=len(corpus_y)) as fx,
                stream(corpus_y, spill=Path(tmp) / "y", others=len(corpus_x)) as fy,
            ):
                result = self.from_features(fx, fy)
                on_disk = any(isinstance(f._data, np.memmap) for f in (fx, fy))
        assert self.last_run is not None
        self.last_run["encoder"] = encoder.provenance
        self.last_run["feature_storage"] = "disk" if on_disk else "memory"
        self.last_run["feature_memory_mb"] = float(limit()[0])
        return result
