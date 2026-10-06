# Contributing

Keep this repository focused on the installable SOL library, its tests,
documentation and the bundled LM1B/OpenWebText examples. Research experiments,
model weights, hidden states and local result files belong outside the package.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev,text]'
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m mypy
cffconvert --validate
python -m unittest discover -s tests/sol_metric -v
python -m unittest discover -s tests/examples -v
python -m build
python -m twine check --strict dist/*
```

Run `python -m ruff format src tests examples` to apply formatting. Mypy uses
the current Python/NumPy typing definitions; CI checks the minimum Python/NumPy
combination and a current combination. Public APIs have annotations and the
wheel includes `py.typed`. Torch/Transformers imports stay lazy.

Library tests cover an independent numerical reference, metric invariances,
cache compatibility/lifecycle, memory limits, precision and input validation.
Text tests create tiny local BERT, GPT-2, OLMo2 and Qwen3 checkpoints without
downloads. Dream's special wrapper interface has an adapter-contract test;
the full upstream 7B checkpoint is not loaded in ordinary CI. CUDA tests run
when a GPU and Torch are available. NumPy-only environments intentionally skip
Torch/text/CUDA tests. Offline runs can set `HF_HUB_OFFLINE=1`.
`test_docs.py` runs every README example in order, and each guide example that
starts with an import, with a tiny local model standing in for any requested
checkpoint; keep those examples runnable. Real-checkpoint and frozen 5k checks
are manual release steps in the [release guide](docs/RELEASING.md#gpu-validation).

## Minimum dependencies

Use a separate Python 3.10 or 3.11 environment to avoid changing a working GPU
installation. The tested floors are NumPy 1.23.5, Torch 2.2.2 and Transformers
4.51.3. Newer Transformers 4.x are allowed; 5.x is not currently supported.

```bash
python -m pip install 'numpy==1.23.5' 'transformers==4.51.3'
python -m pip install 'torch==2.2.2' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,text]'
python -m unittest discover -s tests/sol_metric -v
python -m unittest discover -s tests/examples -v
python -m mypy
```

## Frozen 5k validation

Example tests verify all input hashes, counts, preview prefixes and score-check
behavior without model inference. Full validation re-encodes both 10,000-text
comparisons with the pinned GPT-2 Large checkpoint. Run the commands in the
[README](README.md#validate-the-frozen-5k-pairs) when encoder/numerical changes
could affect the metric. A GPU runner can enable `gpu` and `full_validation`
through the manual workflow; it needs the pinned checkpoint cached and enough
RAM/VRAM/scratch space. Full validation is optional in ordinary CI.

The FP16 expected scores come from the paper's independent evaluation code and
must not be replaced by package measurements; the FP32 values are package
regression baselines. Do not regenerate expected scores merely to make a test pass. Investigate a
change in input selection, tokenization, numerical conventions or execution
precision, and document any intentional baseline/protocol change.

## Changes and reports

Preserve the independent numerical reference when changing execution code.
Add meaningful regression coverage for numerical, compatibility and resource
handling changes. Update the [API reference](docs/API.md), model instructions
and [changelog](CHANGELOG.md) when public behavior changes. Report issues through
the [GitHub issue tracker](https://github.com/GregorKornhardt/SOL/issues), with
the Python/dependency versions, input shapes, configuration and `last_run`
metadata where available. Remove private text/local paths from shared reports.

Release steps are in [docs/RELEASING.md](docs/RELEASING.md).
