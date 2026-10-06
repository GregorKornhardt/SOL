"""Optional Hugging Face token encoder; model dependencies are imported lazily."""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Iterable, Iterator
from typing import Any, Literal

import numpy as np

from ._runtime import resolve_device, torch_module
from ._typing import Encoder, FloatArray, Precision

DEFAULT_MODEL = "Dream-org/Dream-v0-Base-7B"
# Checkpoint commits used for the SOL paper. These are applied when no revision
# is given, so default runs stay reproducible if a model repository changes.
PAPER_REVISIONS = {
    "Dream-org/Dream-v0-Base-7B": "6572adb5535263e4d1a337b56942ba48b6dee2a9",
    "gpt2-large": "32b71b12589c2f8d625668d2335a01cac3249519",
    "openai-community/gpt2-large": "32b71b12589c2f8d625668d2335a01cac3249519",
    "allenai/OLMo-2-0425-1B": "a1847dff35000b4271fa70afc5db10fd29fedbdf",
    "Qwen/Qwen3-0.6B-Base": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
}
# Context lengths documented by model authors where configs state more: Dream's
# config inherits Qwen2.5's 131,072 positions, but its README gives 2048.
DOCUMENTED_CONTEXT = {"dream": 2048}


def position_limit(config: Any, tokenizer: Any) -> int | None:
    """Longest supported input in tokens, or None when unknown.

    The smallest of the model's positions, the tokenizer's limit and any
    documented context length; tokenizers without a limit report about 1e30.
    """
    values = (
        getattr(config, "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
        DOCUMENTED_CONTEXT.get(str(getattr(config, "model_type", "")).lower()),
    )
    limits = [value for value in values if isinstance(value, int) and 0 < value < 10**9]
    return min(limits) if limits else None


class TextEncoder:
    """Frozen final-layer token states, excluding padding.

    Both ``batch_size`` and ``max_tokens_per_batch`` limit padded encoder batches.
    ``max_length`` is a truncation policy and affects the metric. The default
    ``"auto"`` keeps each document whole up to the model's position limit and
    truncates longer ones to it with a warning; ``None`` encodes each document
    to its end or raises.
    The default model is Dream 7B, the SOL paper's recommended backbone. The
    paper's encoders are pinned to the paper's checkpoint commits unless another
    ``revision`` is given; the resolved revision is recorded when available.
    Model weights are loaded on first use and reused by this object.

    Args:
        model_name: Hugging Face model identifier or local checkpoint path,
            default ``Dream-org/Dream-v0-Base-7B``.
        revision: Model/tokenizer revision. None uses the paper's commit for the
            paper's encoders and the repository's latest version otherwise;
            pass "main" for the latest version of a paper encoder.
        device: "auto", "cpu", or an indexed CUDA device such as "cuda:0".
        dtype: Weight/activation precision. "auto" uses float16 on CUDA, as in
            the SOL paper, and float32 on CPU. CPU supports float32 and float64;
            CUDA also accepts float16 and supported bfloat16.
        batch_size: Maximum documents per padded encoder batch.
        max_tokens_per_batch: Maximum padded tokens per encoder batch. With
            ``max_length="auto"``, a longer document is encoded on its own.
        max_length: Truncation horizon in tokens, including special tokens.
            "auto" (default) keeps every token up to the model's position limit
            (the smallest of the model's, the tokenizer's and a documented
            context length, 2048 for Dream) and truncates
            longer documents to it with a RuntimeWarning per corpus. None
            encodes every token; a document beyond the limit then raises
            ValueError.
        text_prefix: Literal prefix prepended to every document.
        local_files_only: Disallow model/tokenizer downloads.
        trust_remote_code: Permit the checkpoint's custom model implementation.
            None enables it only for Dream at the paper's commit, whose model
            code Dream requires; other models need an explicit True.

    Raises:
        ValueError: Invalid identifier, precision, prefix, or positive-size option.
        ImportError: Optional dependencies are missing or incompatible, on use.
        OSError: Checkpoint loading fails, on use.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        revision: str | None = None,
        device: str = "auto",
        dtype: Precision | Literal["auto"] = "auto",
        batch_size: int = 8,
        max_tokens_per_batch: int = 2048,
        max_length: int | Literal["auto"] | None = "auto",
        text_prefix: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool | None = None,
    ) -> None:
        from .metric import _positive_int

        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a model identifier or local path")
        if not isinstance(text_prefix, str):
            raise ValueError("text_prefix must be a string")
        for name, value in (
            ("batch_size", batch_size),
            ("max_tokens_per_batch", max_tokens_per_batch),
        ):
            _positive_int(name, value)
        if max_length is not None and max_length != "auto":
            _positive_int("max_length", max_length)
        if dtype not in ("auto", "float32", "float64", "float16", "bfloat16"):
            raise ValueError("Unsupported encoder dtype")
        if revision is None:
            revision = PAPER_REVISIONS.get(model_name)
        if trust_remote_code is None:
            # Dream needs its own model code; trust it only at the audited paper commit.
            trust_remote_code = revision == PAPER_REVISIONS[DEFAULT_MODEL] and (
                model_name == DEFAULT_MODEL
            )
        self.model_name, self.revision = model_name, revision
        self.device, self.dtype = device, dtype
        self.batch_size, self.max_tokens_per_batch = batch_size, max_tokens_per_batch
        self.max_length, self.text_prefix = max_length, text_prefix
        self.local_files_only, self.trust_remote_code = local_files_only, trust_remote_code
        self.model: Any = None
        self.tokenizer: Any = None
        self._resolved_revision = revision
        self._tokenizer_revision = revision
        self._position_limit: int | None = None

    @property
    def provenance(self) -> dict[str, Any]:
        """JSON-serializable encoding settings and resolved revisions when loaded.

        No model is loaded by reading this property. ``model_type`` and the
        model's ``position_limit`` are added and an "auto" ``dtype`` is resolved
        after loading. This metadata accompanies text features and scores.
        """
        result = {
            "model": self.model_name,
            "revision": self._resolved_revision,
            "tokenizer_revision": self._tokenizer_revision,
            "layer": "last_hidden_state",
            "max_length": self.max_length,
            "text_prefix": self.text_prefix,
            "dtype": self.dtype,
            "padding": "excluded",
            "special_tokens": "included",
        }
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        if model_type:
            result["model_type"] = model_type
        if self._position_limit is not None:
            result["position_limit"] = self._position_limit
        return result

    def _load(self):
        if self.model is not None:
            return
        torch = torch_module()
        try:
            from packaging.version import Version
            from transformers import AutoModel, AutoTokenizer
            from transformers import __version__ as transformers_version
        except ImportError as exc:
            raise ImportError("Text encoding requires the 'sol-metric[text]' extra") from exc
        version = Version(transformers_version)
        if not Version("4.51.3") <= version < Version("5"):
            raise ImportError(
                "Text encoding requires transformers>=4.51.3,<5; "
                "install 'sol-metric[text]' to update compatible dependencies"
            )
        self.device = resolve_device(self.device)
        if self.dtype == "auto":
            # The paper encodes in FP16 on GPUs; CPUs lack FP16 support here.
            self.dtype = "float32" if self.device == "cpu" else "float16"
        if self.device == "cpu" and self.dtype not in ("float32", "float64"):
            raise ValueError("Use float32 or float64 for CPU text encoding")
        if self.dtype == "bfloat16":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("This GPU does not support bfloat16 encoding")
        options = {
            "revision": self.revision,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **options)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(
                    "The tokenizer needs a pad token or EOS token for batched encoding"
                )
            tokenizer.pad_token = tokenizer.eos_token
        # Transformers 4.56 renamed this option; retain support for the minimum
        # dependency version without a deprecation warning on current versions.
        dtype_option = "dtype" if version >= Version("4.56") else "torch_dtype"
        options[dtype_option] = getattr(torch, self.dtype)
        if self.device != "cpu" and importlib.util.find_spec("accelerate") is not None:
            # Materialize weights directly on the GPU. Without Accelerate, Transformers
            # first loads them into host RAM (about 30 GB for FP32 Dream 7B).
            model = AutoModel.from_pretrained(
                self.model_name, device_map={"": self.device}, **options
            )
        else:
            model = AutoModel.from_pretrained(self.model_name, **options).to(self.device)
        model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        self._resolved_revision = getattr(model.config, "_commit_hash", None) or self.revision
        self._tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash") or self.revision
        self._position_limit = position_limit(model.config, tokenizer)
        self.tokenizer, self.model = tokenizer, model

    def __call__(self, texts: Iterable[str]) -> Iterator[FloatArray]:
        """Yield final-layer token states in corpus order, excluding padding.

        Args:
            texts: Nonempty iterable of nonempty strings; consumed lazily.

        Yields:
            One NumPy array shaped (retained tokens, hidden dimension) per
            document. Storage is float64 for float64 encoding, float32 otherwise.

        Raises:
            TypeError: ``texts`` is a single string or bytes object.
            ValueError: Empty corpus/document, invalid item, insufficient token
                budget, a document beyond the model's positions with
                ``max_length=None``, unsupported precision/device, or missing
                final states.
            ImportError: Text dependencies are absent or incompatible.
            OSError: Model or tokenizer loading fails.

        Loading and validation occur when the iterator is consumed. Inference
        is frozen, with gradients disabled and the model in evaluation mode.
        """
        return self._states(texts, None)

    def _states(self, texts: Iterable[str], device: str | None) -> Iterator[Any]:
        """Yield states as NumPy arrays, or as tensors when already on ``device``."""
        if isinstance(texts, (str, bytes)):
            raise TypeError("Pass an iterable of documents, not a single string")
        self._load()
        resident = device is not None and device == self.device
        auto, limit = self.max_length == "auto", self._position_limit
        pending: list[dict[str, Any]] = []
        longest, documents, truncated, overlong = 0, 0, 0, 0
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Every document must be a nonempty string")
            item = self._tokenize(text, None if auto else self.max_length)
            length = len(item["input_ids"])
            if not length:
                raise ValueError("The encoder produced an empty token sequence")
            documents += 1
            if auto and limit is not None and length > limit:
                truncated, overlong = truncated + 1, max(overlong, length)
                item = self._tokenize(text, limit)
                length = len(item["input_ids"])
            if self.max_length is None and limit is not None and length > limit:
                raise ValueError(
                    f"A document has {length} tokens, but the model supports {limit} "
                    "positions; set max_length to truncate"
                )
            # In auto mode, a document beyond the token budget is encoded on its own.
            if length > self.max_tokens_per_batch and not auto:
                raise ValueError(
                    "One encoded document exceeds max_tokens_per_batch; increase it explicitly"
                )
            if pending and (
                len(pending) == self.batch_size
                or (len(pending) + 1) * max(longest, length) > self.max_tokens_per_batch
            ):
                yield from self._encode(pending, resident)
                pending, longest = [], 0
            pending.append(item)
            longest = max(longest, length)
        if pending:
            yield from self._encode(pending, resident)
        if not documents:
            raise ValueError("A corpus must contain at least one document")
        if truncated:
            warnings.warn(
                f"{truncated} of {documents} documents exceed the {limit} positions of "
                f"{self.model_name} and were truncated to them (longest: {overlong} tokens); "
                "pass max_length to set a horizon explicitly",
                RuntimeWarning,
                stacklevel=2,
            )

    def _tokenize(self, text, max_length):
        """Tokenize one document, truncated to max_length unless it is None."""
        return self.tokenizer(
            self.text_prefix + text,
            truncation=max_length is not None,
            max_length=max_length,
            return_attention_mask=True,
            verbose=False,
        )

    def _encode(self, items, resident=False):
        torch = torch_module()
        encoded = self.tokenizer.pad(items, padding=True, return_tensors="pt").to(self.device)
        mask = encoded["attention_mask"].bool()
        with torch.inference_mode():
            if str(getattr(self.model.config, "model_type", "")).lower() == "dream":
                # Dream's AutoModel wraps a masked LM; the base model avoids vocabulary logits.
                kwargs = dict(encoded)
                kwargs["attention_mask"] = mask[:, None, None, :]
                output = self.model.model(**kwargs, use_cache=False)
            else:
                output = self.model(**encoded)
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                raise ValueError("The encoder must expose final token states as last_hidden_state")
            dtype = torch.float64 if self.dtype == "float64" else torch.float32
            for row, keep in zip(hidden, mask):
                states = row[keep].to(dtype)
                # Resident states skip the host round trip; clone leaves inference mode.
                yield states.clone() if resident else states.cpu().numpy()


def as_corpus(texts: Iterable[str]) -> list[str]:
    """Materialize a nonempty corpus; reject a single string."""
    if isinstance(texts, (str, bytes)):
        raise TypeError("Pass an iterable of documents, not a single string")
    corpus = list(texts)
    if not corpus:
        raise ValueError("A corpus must contain at least one document")
    return corpus


def encoded_states(
    texts: Iterable[str], encoder: Encoder, device: str | None = None
) -> tuple[int, Iterator[Any]]:
    """Return the corpus size and a generator of validated token-state matrices.

    The texts are materialized so the feature array can be sized before
    encoding; token states are yielded one document at a time and never stored.
    A ``TextEncoder`` on ``device`` yields tensors there, avoiding host copies.
    """
    corpus = as_corpus(texts)
    if device is not None and isinstance(encoder, TextEncoder):
        produced: Iterable[Any] = encoder._states(corpus, device)
    else:
        produced = encoder(corpus)

    def states() -> Iterator[Any]:
        dimension, dtype, count = None, None, 0
        for document in produced:
            values: Any = document
            tensor = hasattr(values, "is_floating_point")
            if tensor:
                floating = bool(values.is_floating_point())
            else:
                values = np.asarray(values)
                floating = values.dtype.kind == "f"
            if values.ndim != 2 or min(values.shape) < 1 or not floating:
                raise ValueError("An encoder must yield nonempty floating token matrices")
            if dimension is None:
                dimension, dtype = values.shape[1], values.dtype
            if values.shape[1] != dimension or values.dtype != dtype:
                raise ValueError("Encoder dimensions and dtype must stay constant")
            finite = values.isfinite().all() if tensor else np.isfinite(values).all()
            if not bool(finite):
                raise ValueError("Encoder returned nonfinite token states")
            count += 1
            yield values
        if count != len(corpus):
            raise ValueError(f"The encoder yielded {count} documents for {len(corpus)} texts")

    return len(corpus), states()
