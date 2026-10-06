# API reference

Import `SOL`, `SOLConfig`, `TextEncoder`, `Features` and the optional custom
encoder typing contract `Encoder` from `sol_metric`. Only NumPy is required to import
the package. The wheel includes a `py.typed` marker for editor/type-checker use.

## Metric configuration

`SOL(config=None, *, ..., device="auto", dtype="float32", ...)` returns a metric
instance. `config` must be a `SOLConfig`; explicit metric keywords override it.
The following settings also form the fields of the immutable `SOLConfig`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `n_directions` | Automatic in `SOL`; 1024 in `SOLConfig` | Positive number of unit token-space directions. |
| `n_projections` | Automatic in `SOL`; 1024 in `SOLConfig` | Positive number of second-level GP draws. |
| `n_quantiles` | 64 | Positive number of empirical quantile bins. |
| `lengthscale` | 0.1 | Positive, finite RBF length scale; `None` uses independent Gaussian grid values. |
| `seed` | 0 | Nonnegative integer seed for token directions. |
| `gp_seed` | 0 | Nonnegative integer seed for GP draws. |
| `scale` | `None` | Positive, finite multiplier of the squared distance; `None` uses the hidden dimension. |

When omitted in `SOL`, each slice count resolves independently to 8192 for Dream
and 1024 otherwise. A supplied `SOLConfig` or explicit count fixes that count.
Encoder provenance determines automatic counts on each call. `metric.config`
shows the most recently resolved configuration; assigning a `SOLConfig` to it
disables automatic counts. Instances record mutable execution state and should
not be shared between concurrent calls.

Execution settings belong to `SOL`, and do not change the metric definition.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `"auto"` | CUDA if available, CPU otherwise. Explicit `"cpu"`, `"cuda"` and `"cuda:N"` are accepted. |
| `dtype` | `"float32"` | CPU: float32/float64. CUDA: also float16 and supported bfloat16. Low precision applies to matrix products; quantiles and scoring retain at least float32. |
| `batch_size` | 8 | Positive maximum texts per step: per forward pass of the default encoder and per binning step. A `TextEncoder` passed explicitly keeps its own batch settings. |
| `max_tokens_per_batch` | 8192 | Positive maximum padded tokens per such step. A longer document is encoded and binned on its own. |
| `slice_batch_size` | 1024 | Positive slices (token direction and GP draw pairs) scored per step; about 25 bytes per document and slice. |
| `feature_memory_mb` | `"auto"` | Host RAM in MiB for in-memory features. `"auto"` is half of the memory available when features are created, capped by cgroup limits (SLURM, containers), or 4096 where unknown (non-Linux). `from_texts` writes larger feature sets to `temp_dir`; `0` always does. |
| `temp_dir` | `None` | Existing parent directory for temporary text features; `None` uses the system temporary directory. |

Invalid sizes/configuration raise `ValueError` or `TypeError`. A missing optional
Torch installation raises `ImportError` when required. An unavailable CUDA
index raises `ValueError`; explicitly requesting unavailable CUDA raises
`RuntimeError`. Memory follows the batch sizes, which never change scores; an
out-of-memory error during binning or scoring is raised as `MemoryError` naming
the batch size to lower. Model weights, caller inputs and runtime overhead come
on top.

## Score token states

```python
score = metric.from_tokens(tokens_x, offsets_x, tokens_y, offsets_y,
                           provenance=None)
```

Token inputs are finite floating NumPy arrays or Torch tensors shaped
`(total_tokens, hidden_dimension)`. The dimensions must match. CPU bfloat16
Torch inputs are converted before crossing into NumPy; gradients are detached.
Offsets are one-dimensional integer boundaries starting at 0, ending at the
token count and increasing strictly. `[0, 2, 5]` denotes two documents with two
and three token states. Documents and corpus sizes may differ between arms.

The result is a finite, nonnegative Python `float`. `provenance` is an optional
JSON-serializable dictionary used to select automatic slice counts.
`ValueError` indicates invalid shapes, boundaries or nonfinite states.
`MemoryError` indicates that binning or scoring ran out of memory, naming the
batch size to lower; `FloatingPointError` indicates numerical
overflow/nonfinite scoring.

## Build and compare features

```python
left = metric.featurize(tokens_x, offsets_x, path=None, provenance=None)
right = metric.featurize(tokens_y, offsets_y, path=None, provenance=None)
score = metric.from_features(left, right)
```

`featurize` returns `Features` shaped `(documents, n_directions, n_quantiles)`.
Storage is float64 for float64 metric execution and float32 otherwise. Omitting
`path` retains the output in host RAM; specifying a new directory writes an
atomic disk cache and returns a read-only mapping. Existing destinations raise
`FileExistsError`. Metadata must be JSON-serializable and finite. Token-input
validation and memory errors match `from_tokens`; disk errors raise `OSError`.

`from_features` accepts either two open `Features` objects or two finite floating
NumPy arrays of the shape above. Mixing them raises `TypeError`. `Features`
metadata must match exactly, including encoder, precision, hidden dimension,
token-direction seed/count and quantile convention. It must also match the
metric's token-direction and quantile settings. GP seed/count/length scale may
change without re-encoding. Raw arrays have no provenance checks and require an
explicit positive `scale`, such as the original hidden dimension. Invalid or
closed caches, mismatched settings and invalid arrays raise `ValueError`.
Scoring returns a `float`; memory and numerical errors match `from_tokens`.

## Encode and compare texts

```python
score = metric.from_texts(reference, generated, encoder=None)
features = metric.featurize_texts(reference, encoder=None, path=None)
```

Each corpus is a nonempty iterable of nonempty strings. Generators are consumed
once and corpus sizes may differ. Passing a single string raises `TypeError`.
`encoder` is a `TextEncoder` or a custom implementation of the `Encoder`
contract below. `None`, the default, uses a Dream 7B `TextEncoder` on the
metric's device, created on first use and reused by the metric. Each corpus is held as a list of strings; its token states are
turned into features batch by batch as the encoder yields them and are never
stored. A `TextEncoder` on the metric's GPU keeps its states on the GPU. Features
need documents × `n_directions` × `n_quantiles` × 4 bytes. `from_texts` keeps
both corpora's features in RAM up to `feature_memory_mb` and otherwise writes
them to temporary files under `temp_dir`, removed on success or exception;
`last_run["feature_storage"]` records `"memory"` or `"disk"` and
`last_run["feature_memory_mb"]` the limit, resolved once after the encoder has
loaded. Disk caches check free space before writing, and spilling into a
RAM-backed `temp_dir` (tmpfs) raises a `RuntimeWarning`. A custom encoder
must yield exactly one matrix per text.

`from_texts` returns a distance `float` and records the encoder in `last_run`.
`featurize_texts` returns `Features` carrying encoder provenance; cache behavior
matches `featurize`. Invalid documents/embeddings raise `ValueError`, unavailable
optional packages raise `ImportError`, running out of memory raises
`MemoryError` naming the batch size to lower, and disk/model-loading errors can
raise `OSError`. Errors from
custom encoders propagate. Precision, truncation and special-token behavior
come from the encoder; see [model setup](MODELS.md).

## TextEncoder

```python
encoder = TextEncoder(model_name="Dream-org/Dream-v0-Base-7B", revision=None,
                      device="auto", dtype="auto", batch_size=8,
                      max_tokens_per_batch=2048, max_length="auto", text_prefix="",
                      local_files_only=False, trust_remote_code=None)
```

| Parameter | Meaning |
| --- | --- |
| `model_name` | Hugging Face identifier or local checkpoint path as a string; Dream 7B by default. |
| `revision` | Shared model/tokenizer revision. `None` uses the paper's commit for Dream 7B, GPT-2 Large, OLMo 2 1B and Qwen3 0.6B, and the latest version otherwise; `"main"` selects the latest version explicitly. |
| `device`, `dtype` | Device and encoder weight/activation precision. `dtype="auto"` uses float16 on CUDA, as in the paper, and float32 on CPU; explicit choices match the metric. Provenance records the resolved precision. |
| `batch_size` | Positive maximum documents per encoder batch. |
| `max_tokens_per_batch` | Positive maximum padded tokens per encoder batch. With `max_length="auto"`, a longer document is encoded on its own; otherwise each document must fit. |
| `max_length` | Truncation horizon in tokens, including any special tokens. `"auto"` (default) keeps each document whole up to the model's position limit, the smallest of the model's `max_position_embeddings`, the tokenizer's `model_max_length` and a documented context length (2,048 for Dream), and truncates longer documents to it with one `RuntimeWarning` per corpus; `provenance["position_limit"]` records the limit. A positive integer is a fixed horizon. `None` encodes every token; a document beyond the limit raises `ValueError`, and `max_tokens_per_batch` must hold the longest document. |
| `text_prefix` | Literal string prepended before tokenization. |
| `local_files_only` | Load only local/cached assets. |
| `trust_remote_code` | Enable the checkpoint's upstream custom implementation, needed for Dream. `None` enables it only for Dream at the paper's commit; other models and revisions need an explicit `True`. |

`encoder(texts)` returns an iterator yielding one NumPy matrix
`(retained_tokens, hidden_dimension)` per document in order. Padding is excluded;
tokenizer-added special tokens are included. Output storage is float64 for
float64 encoding, float32 otherwise. Model loading and input validation happen
when the iterator is consumed. On CUDA with `accelerate` installed, weights load
directly onto the GPU. Loaded weights are reused and remain frozen in
evaluation/inference mode. Reading `encoder.provenance` does not load a model;
it returns a fresh metadata dictionary, with resolved revisions and model type
after loading when available.

Invalid settings/documents, unsupported CPU/GPU precision, missing padding/EOS
or absent final hidden states raise `ValueError`. A single string raises
`TypeError`; missing/incompatible optional packages raise `ImportError`; model
access failures can raise `OSError`.

## Custom encoders

`Encoder` is a structural typing protocol. Subclassing is unnecessary. Yield
one finite floating NumPy matrix per text, in order, with constant dimension
and dtype. Provide JSON-serializable provenance that identifies the encoding
policy. SOL validates yielded matrices and rejects an empty corpus.

```python
from collections.abc import Iterable, Iterator
import numpy as np
from sol_metric import Encoder, SOL

class LengthEncoder:
    provenance = {"model": "length-example", "revision": "1"}

    def __call__(self, texts: Iterable[str]) -> Iterator[np.ndarray]:
        for text in texts:
            yield np.array([[len(text), 1.0]], dtype=np.float32)

encoder: Encoder = LengthEncoder()
metric = SOL(n_directions=16, n_projections=16, device="cpu")
score = metric.from_texts(["one", "two"], ["longer"], encoder=encoder)
```

## Features lifecycle

`Features(values, metadata)` wraps a caller-owned `(documents, directions,
bins)` array and copies metadata; prefer metric featurization for complete
scientific provenance. The constructor does not validate array shape/provenance.
`shape` and `dtype` describe the logical array, `metadata` is a dictionary and
`closed` is a Boolean.

Features built by `SOL` are stored in direction blocks: `(blocks, documents,
width, bins)`, where `width × bins` values fill one 4 KB page per document.
Scoring reads a range of directions for every document, which is then one
contiguous region, so disk caches and temporary files read quickly on local
disks, network file systems and in RAM alike. `features.block(start, stop)`
returns directions `start:stop` as a `(documents, stop - start, bins)` array.
`values` returns the whole logical array; for block storage this assembles a
copy in host RAM, so use `block` for large caches. Both raise `ValueError` after
closing.

`Features.load(path)` returns a read-only mapped cache, checking schema, shape,
dtype and block layout. Caches from earlier schema versions must be recomputed. It raises `ValueError` for invalid JSON/schema/array metadata,
`KeyError` for missing required fields and `OSError` for unreadable files.
`features.save(path)` returns a new mapped cache; it preserves the source and
never overwrites an existing destination. Invalid metadata raises
`ValueError`/`TypeError`; existing destinations raise `FileExistsError`.

`close()` returns `None`, is idempotent, and closes only mappings owned by the
object. Context managers close on exit without suppressing exceptions. Close
caches before deleting directories, especially on Windows. Do not retain/use
array views after closing. Cache data is read-only; public metadata should be
treated as immutable to preserve provenance checks.

## Run metadata

`metric.last_run` is `None` before successful scoring and a JSON-serializable
dictionary afterward. It includes score, package/dependency versions, resolved
configuration, scale, numerical conventions, device/dtype and memory/block
plan. CUDA adds GPU and matmul/TF32 settings; text scoring adds encoder
provenance. Scoring resets it before validating token/feature inputs or encoding
text, so a failed call cannot leave a stale score from an earlier call.
Featurization does not replace the previous score record. Persist it with the
input hashes when reproducing a comparison.
