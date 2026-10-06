# Model setup

Install `.[text]` from the checkout, or `sol-metric[text]` after PyPI publication.
Set `HF_HOME` before starting Python to choose the model download cache.
Weights are loaded on first encoding and reused by the `TextEncoder` instance.

The text extra requires Torch >=2.2.2 and Transformers >=4.51.3,<5. The
Transformers floor covers Qwen3; older versions do not recognize that model
architecture ([upstream requirements](https://huggingface.co/Qwen/Qwen3-0.6B-Base#requirements)).
The 4.51.3 patch also fixes the earlier loader's dependency on an optional
Accelerate import, so ordinary encoding does not require that extra package.
The encoder checks this supported range before accessing a checkpoint. Both
the older `torch_dtype` and current `dtype` loading options are handled.

| Encoder | Hugging Face identifier | Hidden dimension | Default slices per level | Paper revision |
| --- | --- | ---: | ---: | --- |
| Dream 7B (default) | `Dream-org/Dream-v0-Base-7B` | 3584 | 8192 | `6572adb5535263e4d1a337b56942ba48b6dee2a9` |
| GPT-2 Large | `gpt2-large` | 1280 | 1024 | `32b71b12589c2f8d625668d2335a01cac3249519` |
| OLMo 2 1B | `allenai/OLMo-2-0425-1B` | 2048 | 1024 | `a1847dff35000b4271fa70afc5db10fd29fedbdf` |
| Qwen3 0.6B Base | `Qwen/Qwen3-0.6B-Base` | 1024 | 1024 | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |

`TextEncoder()` without a model name uses Dream 7B, and `SOL.from_texts` uses
such an encoder when none is passed. A revision pins the exact checkpoint
commit on the Hugging Face Hub, because updated weights, tokenizer files or
model code change the resulting distances. The four encoders above use the
paper's commits unless another `revision` is given; pass `revision='main'` for
a repository's latest version. Other models load their latest version unless a
revision is given. Encoder precision, truncation and text prefixes are also part
of the feature provenance and affect the resulting distance.

```python
from sol_metric import TextEncoder

encoder = TextEncoder(
    'gpt2-large', device='cpu', dtype='float32', max_length=256,
    batch_size=4, max_tokens_per_batch=1024,
)
```

## Precision

The encoder default `dtype="auto"` uses FP16 on CUDA, the setting used for the
paper's results, and FP32 on CPU, which does not support FP16 here. Explicit
choices are `float32`/`float64` on CPU and `float16`, `float32` or, on
compatible GPUs, `bfloat16` on CUDA. Token states are returned as float32
(float64 for float64 encoding), and the metric computes in FP32 regardless of
the encoder precision. Provenance records the resolved precision.

Measured with Dream 7B on an RTX 5090, at a 1,024-token horizon:

| Encoder precision | Speed | 5,000 OWT documents | Score vs FP32 (32-document sample) |
| --- | ---: | ---: | ---: |
| `float16` (CUDA default) | 12,400 tokens/s | about 7 min | −7e-5 relative |
| `float32` | 3,700 tokens/s | about 22 min | reference |
| `bfloat16` | not measured | not measured | −3e-3 relative |

FP32 matrix products do not use tensor cores on consumer GPUs, hence the gap.
Batching changes FP32 token states only at rounding level (about 1e-5 relative).
In FP16, the batch composition changes Dream token states by up to about 2% per
token and the score by about 3e-6 relative between batch budgets; bfloat16
changes token states by up to 14%. Record `batch_size` and
`max_tokens_per_batch` with FP16 scores, and use FP32 when results must not
depend on batching.

## Memory and batching

Model weights and activations come on top of the metric's own batches.
GPT-2 Large needs several GB of RAM. Dream weights need about 15 GB in FP16 and
28 GB in FP32; on a 32 GB GPU, FP32 Dream needs `max_tokens_per_batch` of at
most 4096. On CUDA, install `accelerate` (`python -m pip install accelerate`) to
load weights directly onto the GPU. Without it, Transformers first materializes
them in host RAM, so FP32 Dream also needs about 30 GB of free host memory.

`max_length` sets the truncation horizon. The default `"auto"` keeps every token
up to the model's position limit (1,024 for GPT-2 Large, 2,048 for Dream 7B,
4,096 for OLMo 2 1B and 32,768 for Qwen3 0.6B) and truncates longer documents
to that limit, with one warning per corpus giving their number and the longest
length. The limit is the smallest of the model's position embeddings, the
tokenizer's maximum and a context length documented by the model authors:
Dream's configuration inherits 131,072 positions from Qwen2.5, but its
[README](https://github.com/DreamLM/Dream) gives a context of 2,048 tokens.
`max_length=None` keeps every token and raises for a document beyond the limit.
Batches pad only to their longest document, so a horizon above the longest
document gives the same result as `None`. `batch_size` and
`max_tokens_per_batch` limit padded batches; with `"auto"`, a document longer
than `max_tokens_per_batch` is encoded on its own. All final-layer non-padding
states are retained. Models must
expose `last_hidden_state`; tokenizers need a pad token or an EOS token that
can serve as padding. Unsupported model architectures fail explicitly.

## Dream

Dream requires its upstream custom model code. `trust_remote_code=None`, the
default, enables that code only for Dream at the paper's pinned commit; any
other revision or model needs an explicit `trust_remote_code=True`, best with a
pinned revision. Its adapter extracts the base
transformer's final states without vocabulary logits, with bidirectional
attention and padding excluded. No model implementation or weights are bundled.

Dream's automatic slice defaults are recognized from the loaded `model_type`,
including local checkpoints, or its `Dream-org/Dream-*` identifier. Explicit
slice counts override these defaults. Saved features preserve the provenance.

## Offline use

For offline encoding, set `local_files_only=True` and use a cached model ID or
local checkpoint path. For the bundled examples, pass
`--model PATH --local-files-only`. Models and tokenizers retain their upstream
licenses.

## Test coverage

Ordinary tests create tiny local BERT, GPT-2, OLMo2 and Qwen3 checkpoints and
verify batching, truncation, EOS padding, final-state extraction and agreement
between direct/cached scoring. CI tests the minimum and current dependency
combinations without model downloads. Dream's special base-model/mask contract
is tested with a small interface stub; its complete custom 7B checkpoint is
not loaded in ordinary CI. The bundled LM1B/OpenWebText full 5k baselines were
computed with the pinned GPT-2 Large checkpoint and can be rerun manually.
