"""The dataset examples preserve frozen input selection and check integrity."""

import copy
import gzip
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

path = Path(__file__).resolve().parents[2] / "examples/compare_corpora.py"
spec = importlib.util.spec_from_file_location("compare_corpora", path)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)

# The frozen 5k archives are in the Git repository but not the source distribution.
HAS_5K_ARCHIVES = all(
    (example.ROOT / "examples/validation_data" / f"{name}.jsonl.gz").is_file()
    for names in example.CORPORA.values()
    for name in names
)


class ExampleInputsTests(unittest.TestCase):
    def test_prefix_selection_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "texts.jsonl.gz"
            texts = ["First document.", "Second document.", "Third document."]
            with gzip.open(archive, "wt", encoding="utf-8") as handle:
                for text in texts:
                    handle.write(json.dumps({"text": text}) + "\n")
            checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
            index = {"test": {"path": archive.name, "count": 3, "sha256": checksum}}
            (root / "corpora.json").write_text(json.dumps(index), encoding="utf-8")
            actual, provenance = example.read_texts(root, "test", 2)
            self.assertEqual(actual, texts[:2])
            self.assertEqual(provenance, {"corpus": "test", "sha256": checksum, "rows": [0, 2]})
            with self.assertRaisesRegex(ValueError, "contains 3"):
                example.read_texts(root, "test", 4)
            with self.assertRaisesRegex(ValueError, "example data"):
                example.read_texts(root, "missing", 2)
            archive.write_bytes(archive.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                example.read_texts(root, "test", 2)

    def test_rejects_archive_path_outside_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = {"test": {"path": "../elsewhere.jsonl.gz", "count": 3, "sha256": ""}}
            (root / "corpora.json").write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                example.read_texts(root, "test", 2)

    def test_both_bundled_examples_run_without_loading_an_encoder(self):
        for dataset in ("lm1b", "owt"):
            with self.subTest(dataset=dataset), patch.object(example, "TextEncoder") as encoder:
                output = io.StringIO()
                with redirect_stdout(output):
                    example.main([dataset, "--dry-run"])
                encoder.assert_not_called()
                result = json.loads(output.getvalue())
                self.assertEqual(result["dataset"], dataset)
                self.assertEqual(result["samples_per_corpus"], 32)
                self.assertEqual(result["reference"]["rows"], [0, 32])
                self.assertNotIn("score", result)

    @unittest.skipUnless(HAS_5K_ARCHIVES, "frozen 5k archives are only in the Git repository")
    def test_full_validation_pairs_preserve_preview_prefixes_and_run_offline(self):
        for dataset, names in example.CORPORA.items():
            with self.subTest(dataset=dataset):
                for name in names:
                    texts, info = example.read_texts(
                        example.ROOT / "examples/validation_data", name, 5000
                    )
                    preview, _ = example.read_texts(example.ROOT / "examples/data", name, 32)
                    self.assertEqual(len(texts), 5000)
                    self.assertEqual(texts[:32], preview)
                    self.assertEqual(info["rows"], [0, 5000])
                with (
                    patch.object(example, "TextEncoder") as encoder,
                    redirect_stdout(io.StringIO()) as output,
                ):
                    example.main([dataset, "--validate", "--dry-run"])
                    encoder.assert_not_called()
                    result = json.loads(output.getvalue())
                    self.assertEqual(result["samples_per_corpus"], 5000)
                    self.assertEqual(result["slices_per_level"], 1024)
                    self.assertFalse(result["validation"]["score_checked"])


class ValidationTests(unittest.TestCase):
    def setUp(self):
        config = {
            "n_directions": 1024,
            "n_projections": 1024,
            "n_quantiles": 64,
            "lengthscale": 0.1,
            "seed": 0,
            "gp_seed": 0,
            "scale": None,
        }
        protocol = {
            "samples_per_corpus": 5000,
            "max_length": 1024,
            "slices_per_level": 1024,
            "encoder": example.MODEL,
            "revision": example.REVISION,
            "encoder_batch_size": 8,
            "encoder_max_tokens_per_batch": 8192,
            "input_mode": "retokenized_text",
            "config": config,
        }
        left = {"corpus": "lm1b_reference", "sha256": "reference-hash", "rows": [0, 5000]}
        right = {"corpus": "lm1b_generated", "sha256": "generated-hash", "rows": [0, 5000]}
        self.expected = {
            "protocol": protocol,
            "reference": left,
            "generated": right,
            "encoder_dtype": "float32",
            "score": 0.75,
            "score_source": "test",
            "rtol": 1e-4,
            "atol": 2e-6,
        }
        self.result = {name: value for name, value in protocol.items() if name != "config"}
        self.result.update(
            dataset="lm1b",
            reference=left,
            generated=right,
            encoder_dtype="float32",
            score=0.75,
            run={
                "config": config,
                "gp_normalization": "none",
                "quantile_convention": "empirical_bin_average_v1",
                "squared": False,
                "scale": 1280,
                "device": "cpu",
                "dtype": "float32",
                "encoder": {
                    "revision": example.REVISION,
                    "tokenizer_revision": example.REVISION,
                    "dtype": "float32",
                    "max_length": 1024,
                },
            },
        )

    def test_accepts_matching_score_and_small_rounding_differences(self):
        for score in (0.75, 0.750001):
            self.result["score"] = score
            self.assertTrue(example.check_result(self.result, self.expected)["passed"])

    def test_rejects_changed_inputs_protocol_and_score(self):
        for field in (
            "hash",
            "sample_count",
            "seed",
            "precision",
            "encoder_precision",
            "batching",
            "revision",
            "score",
            "nan",
            "tf32",
        ):
            result = copy.deepcopy(self.result)
            if field == "hash":
                result["generated"]["sha256"] = "different"
            elif field == "sample_count":
                result["samples_per_corpus"] = 32
            elif field == "seed":
                result["run"]["config"]["seed"] = 1
            elif field == "precision":
                result["run"]["dtype"] = "float16"
            elif field == "encoder_precision":
                result["encoder_dtype"] = result["run"]["encoder"]["dtype"] = "float16"
            elif field == "batching":
                result["encoder_max_tokens_per_batch"] = 2048
            elif field == "revision":
                result["run"]["encoder"]["revision"] = "different"
            elif field == "score":
                result["score"] = 0.8
            elif field == "nan":
                result["score"] = float("nan")
            else:
                result["run"].update(device="cuda:0", torch={"allow_tf32": True})
            with self.subTest(field=field), self.assertRaises(ValueError):
                example.check_result(result, self.expected)

    def manifest(self, root, scores):
        manifest = {
            "schema": 2,
            "protocol": self.expected["protocol"],
            "rtol": 1e-4,
            "atol": 2e-6,
            "datasets": {
                "lm1b": {
                    "reference": self.expected["reference"],
                    "generated": self.expected["generated"],
                    "scores": scores,
                    "score_sources": {name: "test" for name in scores},
                }
            },
        }
        (root / "validation.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_validation_disallows_small_demo_overrides_before_loading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.manifest(root, {"float32": 0.75})
            for options in (["--samples", "32"], ["--dtype", "float16"]):
                with (
                    self.subTest(options=options),
                    patch.object(example, "read_texts") as load,
                    patch("sys.stderr", new=io.StringIO()),
                ):
                    with self.assertRaises(SystemExit) as error:
                        example.main(["lm1b", "--validate", "--data-dir", str(root), *options])
                    self.assertEqual(error.exception.code, 2)
                    load.assert_not_called()

    def test_saved_result_is_checked_against_its_own_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.manifest(root, {"float16": 0.7, "float32": 0.75})
            saved = copy.deepcopy(self.result)
            saved.update(encoder_dtype="float16", score=0.7)
            saved["run"]["encoder"]["dtype"] = "float16"
            path = root / "result.json"
            path.write_text(json.dumps(saved), encoding="utf-8")
            inputs = [(["text"], self.expected[side]) for side in ("reference", "generated")]
            with (
                patch.object(example, "read_texts", side_effect=inputs),
                redirect_stdout(io.StringIO()) as output,
            ):
                example.main(["lm1b", "--check-result", str(path), "--data-dir", str(root)])
            validation = json.loads(output.getvalue())["validation"]
            self.assertTrue(validation["passed"])
            self.assertEqual(validation["encoder_dtype"], "float16")
            self.assertEqual(validation["expected_score"], 0.7)


if __name__ == "__main__":
    unittest.main()
