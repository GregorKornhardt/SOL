# Releasing sol-metric

The distribution is `sol-metric`, its import package is `sol_metric`, and the source
repository is [GregorKornhardt/SOL](https://github.com/GregorKornhardt/SOL).
The version lives in `src/sol_metric/_version.py`; keep `CITATION.cff` and the changelog
consistent with it. The prepared 0.1.1 version has not been published here.

## Prepare artifacts

Use a clean release checkout and a separate environment. Install a suitable
Torch build first if CUDA checks are needed.

```bash
python -m pip install '.[dev,text]'
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m mypy
cffconvert --validate
python -m unittest discover -s tests/sol_metric -v
python -m unittest discover -s tests/examples -v
python -m build --outdir release-dist
python -m twine check --strict release-dist/*
```

Use an empty artifact directory for each release. The wheel should contain only
`sol_metric` code, `py.typed`, package metadata and licenses. The source archive should
also contain docs, tests, the example script, the 32-text example inputs and the
5k validation manifest and provenance (`examples/validation_data/*.json`, `*.md`).
The four 5k text archives (~20 MB) stay in the Git repository and its release
tags only; `MANIFEST.in` excludes them. The source archive must contain no
experiments, model weights, local feature arrays or results. Check
author/maintainer and software/paper citation details before publishing. GitHub
CI must pass on the intended release revision; local Linux checks alone do not
verify macOS/Windows.

Install the wheel in a fresh NumPy-only environment and import outside the
checkout to verify that Torch/Transformers are optional. Then extract the source
archive and run its tests/examples against the installed wheel; the test that
reads the 5k archives skips there. `tests/sol_metric/test_docs.py` runs the
README and guide examples with a tiny local model, so keep examples runnable.

## GPU validation

CI has no GPU runner, so run these checks by hand on a CUDA machine for every
release candidate, from a checkout of the release revision with `.[text]`
installed. Each run must end with `validation.passed: true`:

```bash
python examples/compare_corpora.py owt --validate --device cuda:0 --output results/owt.json
python examples/compare_corpora.py lm1b --validate --device cuda:0 --output results/lm1b.json
python examples/compare_corpora.py owt --validate --device cuda:0 --dtype float32 --output results/owt_fp32.json
python examples/compare_corpora.py lm1b --validate --device cuda:0 --dtype float32 --output results/lm1b_fp32.json
```

These re-encode both frozen 5k pairs with pinned GPT-2 Large; the
[README](../README.md#validate-the-frozen-5k-pairs) records the protocol and
resources. For a cached checkpoint, set `HF_HOME` and `HF_HUB_OFFLINE=1`, or
pass `--model PATH --local-files-only`. On an RTX 5090 the FP16 runs took about
2.5 minutes (OpenWebText) and 35 seconds (LM1B), with about 5 GB of host RAM.
Preserve expected scores unless a scientific change has been reviewed and
documented. The repository's manual `gpu` workflow runs the same commands on a
self-hosted runner once one is registered.

Then load each documented encoder once on the GPU and score a small example.
Each must print a finite positive score, 8192 slices for Dream and 1024
otherwise, and the paper revision listed in [model setup](MODELS.md):

```bash
for model in Dream-org/Dream-v0-Base-7B gpt2-large allenai/OLMo-2-0425-1B Qwen/Qwen3-0.6B-Base; do
python - "$model" <<'EOF'
import sys
from sol_metric import SOL, TextEncoder

reference = ["The train arrived at noon.", "It rained throughout the night."]
generated = ["The afternoon train was late.", "Heavy rain fell overnight."]
metric = SOL(device="cuda:0")
encoder = TextEncoder(sys.argv[1], device="cuda:0")
score = metric.from_texts(reference, generated, encoder=encoder)
print(sys.argv[1], score, metric.config.n_directions, metric.last_run["encoder"]["revision"])
EOF
done
```

Dream needs about 15 GB of GPU memory in FP16. With `HF_HUB_OFFLINE=1`,
Transformers 4.57 fails to resolve OLMo by its Hub ID; pass the local snapshot
directory instead.

## TestPyPI rehearsal

Check that the intended project name is available or controlled by the
maintainer. PyPI and TestPyPI are separate services/accounts. For manual uploads,
use a scoped API token and Twine's interactive credential prompt or a credential
store; keep credentials out of the repository.

The following commands upload externally and are separate from local builds:

```bash
python -m twine upload --repository testpypi release-dist/*
```

For a fresh smoke-test environment, install NumPy from PyPI first, then fetch
only the release itself from TestPyPI. Replace `0.1.1` with the release version.

```bash
python -m pip install numpy
python -m pip install --no-deps --index-url https://test.pypi.org/simple/ 'sol-metric==0.1.1'
python -c "import sys, sol_metric; from importlib.metadata import version; from importlib.resources import files; assert version('sol-metric') == sol_metric.__version__; assert files('sol_metric').joinpath('py.typed').is_file(); assert not {'torch', 'transformers'} & set(sys.modules)"
```

Also test the text extra using dependencies from PyPI and a suitable Torch
installation. Source examples require the source archive/checkout, because
datasets and scripts are deliberately excluded from the wheel.

## PyPI publication

After the rehearsal and review, upload the same checked wheel/source artifacts:

```bash
python -m twine upload release-dist/*
```

Release files/version numbers cannot be reused to replace existing PyPI files.
Use a new version for fixes. Install the published version in a fresh environment
and verify its import/version and examples. Record the final version/date in
the changelog and citation, and associate the matching source revision with the
release when GitHub publication is authorized.

There is no automatic publishing workflow in this repository. A future workflow
can use PyPI Trusted Publishing: configure the GitHub owner `GregorKornhardt`,
repository `SOL`, exact workflow filename and any chosen environment separately
on PyPI and TestPyPI. Pending publishers support the first upload of a new
project. Configure that account integration before enabling uploads.

See the official [packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
and [PyPI publisher setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
