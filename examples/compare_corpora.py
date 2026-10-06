"""Compare the bundled LM1B or OpenWebText reference and generated examples.

Install the checkout with ``python -m pip install '.[text]'`` first.
Use --validate for the frozen 5,000-text pairs and their expected scores.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from importlib.metadata import version
from itertools import islice
from pathlib import Path

from sol_metric import SOL, TextEncoder

ROOT = Path(__file__).resolve().parents[1]
CORPORA = {
    "lm1b": ("lm1b_reference", "lm1b_generated"),
    "owt": ("owt_reference", "owt_generated"),
}
MODEL = "gpt2-large"
REVISION = "32b71b12589c2f8d625668d2335a01cac3249519"
# Encoder batches follow the paper's worker: eight documents per padded batch.
# FP16 token states depend slightly on batch composition, so this is fixed.
ENCODER_BATCH_SIZE = 8
ENCODER_MAX_TOKENS = 8192


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def read_texts(data_dir, name, count):
    """Verify a frozen shard, then read only the requested text prefix."""
    if count < 1:
        raise ValueError("The requested text count must be positive")
    root = Path(data_dir).resolve()
    index = json.loads((root / "corpora.json").read_text(encoding="utf-8"))
    if name not in index:
        raise ValueError(f"{name} is missing from the example data")
    entry = index[name]
    if count > entry["count"]:
        raise ValueError(f"Requested {count} rows, but {name} contains {entry['count']}")
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Corpus path escapes the example data directory")
    if not path.is_file():
        raise FileNotFoundError(
            f"{entry['path']} is missing from {root}; the frozen 5k archives are in the "
            "Git repository https://github.com/GregorKornhardt/SOL, not the source distribution"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"Corpus checksum mismatch: {name}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        texts = [json.loads(line)["text"] for line in islice(handle, count)]
    if len(texts) != count or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError(f"Expected {count} nonempty texts in {name}")
    return texts, {"corpus": name, "sha256": entry["sha256"], "rows": [0, count]}


def resolve_dtype(dtype, device):
    """Match TextEncoder's "auto": FP16 on CUDA, as in the paper, and FP32 on CPU."""
    if dtype == "auto":
        return "float32" if device == "cpu" else "float16"
    return dtype


def validation_expectation(data_dir, dataset, dtype):
    """Load the recorded numerical contract for one frozen 5k pair and precision."""
    manifest = json.loads((Path(data_dir) / "validation.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 2:
        raise ValueError("Unsupported validation manifest schema")
    entry = manifest["datasets"][dataset]
    if dtype not in entry["scores"]:
        raise ValueError(f"No expected {dtype} score; recorded: {sorted(entry['scores'])}")
    score = entry["scores"][dtype]
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or score < 0
    ):
        raise ValueError("Validation manifest needs a finite nonnegative expected score")
    return {
        "protocol": manifest["protocol"],
        "rtol": manifest["rtol"],
        "atol": manifest["atol"],
        "reference": entry["reference"],
        "generated": entry["generated"],
        "encoder_dtype": dtype,
        "score": score,
        "score_source": entry["score_sources"][dtype],
    }


def check_result(result, expected):
    """Check inputs and scientific settings before comparing the SOL score."""
    for name, value in expected["protocol"].items():
        actual = result.get("run", {}).get("config") if name == "config" else result.get(name)
        if actual != value:
            raise ValueError(f"Validation protocol mismatch: {name}")
    for side in ("reference", "generated"):
        if result.get(side) != expected[side]:
            raise ValueError(f"Validation input mismatch: {side}")
    run = result.get("run", {})
    if (
        run.get("gp_normalization") != "none"
        or run.get("quantile_convention") != "empirical_bin_average_v1"
        or run.get("squared") is not False
        or run.get("scale") != 1280
        or run.get("dtype") != "float32"
    ):
        raise ValueError("Validation numerical convention mismatch")
    if (
        str(run.get("device", "")).startswith("cuda")
        and run.get("torch", {}).get("allow_tf32") is not False
    ):
        raise ValueError("Validation requires TF32 to be disabled")
    encoder = run.get("encoder", {})
    if encoder.get("revision") != REVISION or encoder.get("tokenizer_revision") != REVISION:
        raise ValueError("Validation encoder revision mismatch")
    if (
        result.get("encoder_dtype") != expected["encoder_dtype"]
        or encoder.get("dtype") != expected["encoder_dtype"]
        or encoder.get("max_length") != expected["protocol"]["max_length"]
    ):
        raise ValueError("Validation encoder precision or truncation mismatch")
    score = result.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("Validation needs a finite SOL score")
    if not math.isclose(
        score, expected["score"], rel_tol=expected["rtol"], abs_tol=expected["atol"]
    ):
        raise ValueError(
            f"Validation score mismatch: got {score}, expected {expected['score']} "
            f"(rtol={expected['rtol']}, atol={expected['atol']})"
        )
    return {
        "passed": True,
        "encoder_dtype": expected["encoder_dtype"],
        "expected_score": expected["score"],
        "expected_score_source": expected["score_source"],
        "absolute_error": abs(score - expected["score"]),
        "relative_error": abs(score - expected["score"]) / expected["score"],
        "rtol": expected["rtol"],
        "atol": expected["atol"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=CORPORA)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Default: examples/data, or examples/validation_data with --validate",
    )
    parser.add_argument(
        "--samples", type=positive_int, help="Default: 32, or the recorded 5000 with --validate"
    )
    parser.add_argument("--max-length", type=positive_int, help="Default: 1024")
    parser.add_argument(
        "--slices", type=positive_int, help="Default: 128, or the recorded 1024 with --validate"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "float32"),
        default="auto",
        help="Encoder precision; auto uses float16 on CUDA and float32 on CPU",
    )
    parser.add_argument("--model", help="Optional local GPT-2 Large checkpoint path")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--scratch-dir", type=Path, help="Temporary feature storage")
    parser.add_argument(
        "--dry-run", action="store_true", help="Verify inputs without loading a model"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the frozen 5k pair and check its recorded score",
    )
    parser.add_argument(
        "--check-result",
        type=Path,
        help="Check a saved 5k result without encoding again; implies --validate",
    )
    parser.add_argument("--output", type=Path, help="Also save the result as JSON")
    args = parser.parse_args(argv)
    args.validate = args.validate or args.check_result is not None
    if args.check_result is not None and args.dry_run:
        parser.error("--check-result cannot be combined with --dry-run")
    args.data_dir = args.data_dir or ROOT / (
        "examples/validation_data" if args.validate else "examples/data"
    )
    saved = None
    if args.check_result is not None:
        try:
            saved = json.loads(args.check_result.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot read saved result: {exc}")
        dtype = saved.get("encoder_dtype")
    else:
        dtype = resolve_dtype(args.dtype, args.device)
        if args.device == "cpu" and dtype == "float16":
            parser.error("float16 encoding needs a CUDA device; use --dtype float32 on CPU")
    expected = None
    if args.validate:
        try:
            expected = validation_expectation(args.data_dir, args.dataset, dtype)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"Cannot read validation manifest: {exc}")
    for option, field, default in (
        ("samples", "samples_per_corpus", 32),
        ("max_length", "max_length", 1024),
        ("slices", "slices_per_level", 128),
    ):
        value = getattr(args, option)
        required = expected["protocol"][field] if expected else default
        if expected and value is not None and value != required:
            parser.error(f"Validation requires --{option.replace('_', '-')} {required}")
        setattr(args, option, required if value is None else value)
    try:
        reference, reference_info = read_texts(
            args.data_dir, CORPORA[args.dataset][0], args.samples
        )
        generated, generated_info = read_texts(
            args.data_dir, CORPORA[args.dataset][1], args.samples
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(
            f"Cannot read example inputs: {exc}. Use --data-dir PATH to the bundled example data."
        )
    result = {
        "dataset": args.dataset,
        "reference": reference_info,
        "generated": generated_info,
        "samples_per_corpus": args.samples,
        "max_length": args.max_length,
        "slices_per_level": args.slices,
        "encoder": MODEL,
        "revision": REVISION,
        "encoder_dtype": dtype,
        "encoder_batch_size": ENCODER_BATCH_SIZE,
        "encoder_max_tokens_per_batch": ENCODER_MAX_TOKENS,
        "input_mode": "retokenized_text",
        "example_only": not args.validate,
    }
    if saved is not None:
        try:
            if saved.get("dataset") != args.dataset:
                raise ValueError("Validation dataset mismatch")
            result = saved
            result["validation"] = check_result(result, expected)
        except (ValueError, KeyError) as exc:
            parser.error(str(exc))
    elif not args.dry_run:
        if args.scratch_dir is not None:
            args.scratch_dir.mkdir(parents=True, exist_ok=True)
        import torch

        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            encoder = TextEncoder(
                args.model or MODEL,
                revision=REVISION,
                device=args.device,
                dtype=dtype,
                max_length=args.max_length,
                batch_size=ENCODER_BATCH_SIZE,
                max_tokens_per_batch=max(ENCODER_MAX_TOKENS, args.max_length),
                local_files_only=args.local_files_only,
            )
            metric = SOL(
                n_directions=args.slices,
                n_projections=args.slices,
                device=args.device,
                max_tokens_per_batch=max(8192, args.max_length),
                temp_dir=args.scratch_dir,
            )
            result["score"] = metric.from_texts(reference, generated, encoder=encoder)
            result["run"] = metric.last_run
            result["transformers_version"] = version("transformers")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        if expected:
            try:
                result["validation"] = check_result(result, expected)
            except ValueError as exc:
                parser.error(str(exc))
    elif expected:
        result["validation"] = {
            "encoder_dtype": dtype,
            "expected_score": expected["score"],
            "expected_score_source": expected["score_source"],
            "score_checked": False,
            "protocol": expected["protocol"],
        }
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
