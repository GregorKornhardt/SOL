"""Encoder-dependent slice defaults without model downloads."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sol_metric import SOL, Features, SOLConfig, TextEncoder


class TinyEncoder(TextEncoder):
    """Use the real provenance property and expose model type after encoding."""

    def __init__(self, model_name, model_type=None):
        super().__init__(model_name, device="cpu")
        self.model_type = model_type

    def __call__(self, texts):
        if self.model_type is not None:
            self.model = SimpleNamespace(config=SimpleNamespace(model_type=self.model_type))
        for text in texts:
            yield np.array([[float(len(text))]], dtype=np.float32)


class EncoderDefaultTests(unittest.TestCase):
    def metric(self, **kwargs):
        return SOL(
            device="cpu",
            n_quantiles=1,
            slice_batch_size=512 * 512,
            **kwargs,
        )

    def check_score(self, metric, encoder, counts):
        score = metric.from_texts(["a"], ["aaa"], encoder=encoder)
        cfg = metric.last_run["config"]
        self.assertEqual((cfg["n_directions"], cfg["n_projections"]), counts)
        # In one dimension, unit token directions are +/-1. For singleton
        # corpora and one quantile bin, SOL is the point gap times GP RMS.
        gp = np.random.default_rng(cfg["gp_seed"]).standard_normal(counts[1])
        expected = 2 * np.sqrt(np.mean(gp**2) * (1 + 1e-6))
        self.assertAlmostEqual(score, expected, places=6)
        return score

    def test_local_dream_defaults_and_saved_features(self):
        encoder = TinyEncoder("/local/checkpoint", "dream")
        self.assertNotIn("model_type", encoder.provenance)
        metric = self.metric()
        score = self.check_score(metric, encoder, (8192, 8192))
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp) / "left", Path(tmp) / "right"
            with metric.featurize_texts(["a"], encoder=encoder, path=left) as features:
                self.assertEqual(features.values.shape, (1, 8192, 1))
                self.assertEqual(features.metadata["provenance"]["model_type"], "dream")
            with metric.featurize_texts(["aaa"], encoder=encoder, path=right):
                pass
            reader = self.metric()
            with Features.load(left) as fx, Features.load(right) as fy:
                self.assertAlmostEqual(reader.from_features(fx, fy), score, places=6)
                self.assertEqual(reader.config.n_projections, 8192)
                with self.assertRaisesRegex(ValueError, "n_directions"):
                    self.metric(n_directions=1024).from_features(fx, fy)

    def test_model_detection_and_reusing_metric(self):
        metric = self.metric(n_directions=2, scale=1)
        for name, model_type, count in (
            ("Dream-org/Dream-v0-Base-7B", None, 8192),
            ("gpt2-large", "gpt2", 1024),
            ("/local/checkpoint", "dream", 8192),
            ("Dream-org/Dream-v0-Base-7B", "gpt2", 1024),
            ("someone/dream-text-encoder", None, 1024),
        ):
            with self.subTest(model=name, model_type=model_type):
                self.check_score(metric, TinyEncoder(name, model_type), (2, count))

        encoder = TinyEncoder("Dream-org/Dream-v0-Base-7B")
        self.check_score(metric, encoder, (2, 8192))
        x, y = np.array([[1.0]]), np.array([[3.0]])
        metric.from_tokens(x, [0, 1], y, [0, 1])
        self.assertEqual(metric.last_run["config"]["n_projections"], 1024)
        self.check_score(metric, encoder, (2, 8192))
        metric.from_features(np.ones((1, 2, 1)), np.full((1, 2, 1), 3.0))
        self.assertEqual(metric.last_run["config"]["n_projections"], 1024)
        self.check_score(metric, encoder, (2, 8192))
        with metric.featurize(x, [0, 1]) as fx, metric.featurize(y, [0, 1]) as fy:
            metric.from_features(fx, fy)
            self.assertEqual(metric.last_run["config"]["n_projections"], 1024)

    def test_explicit_counts_and_config_take_precedence(self):
        encoder = TinyEncoder("Dream-org/Dream-v0-Base-7B")
        for kwargs, counts in (
            ({"n_directions": 1024, "n_projections": 3}, (1024, 3)),
            ({"n_directions": 2}, (2, 8192)),
            ({"n_projections": 3}, (8192, 3)),
            ({"config": SOLConfig(n_directions=3, n_projections=5)}, (3, 5)),
        ):
            with self.subTest(kwargs=kwargs):
                self.check_score(self.metric(**kwargs), encoder, counts)
        metric = self.metric()
        metric.config = replace(metric.config, n_directions=3, n_projections=5)
        self.check_score(metric, encoder, (3, 5))

    def test_token_provenance_and_explicit_direction_defaults(self):
        metric = self.metric(n_projections=3)
        x, y = np.array([[1.0]]), np.array([[3.0]])
        provenance = {"model": "Dream-org/Dream-v0-Base-7B"}
        metric.from_tokens(x, [0, 1], y, [0, 1], provenance=provenance)
        self.assertEqual(metric.config.n_directions, 8192)
        with metric.featurize(x, [0, 1], provenance=provenance) as fx:
            self.assertEqual(fx.values.shape, (1, 8192, 1))
        with metric.featurize(x, [0, 1]) as fx:
            self.assertEqual(fx.values.shape, (1, 1024, 1))
        # Even the ordinary default counts stay explicit when supplied as config.
        metric = self.metric(config=SOLConfig(), n_projections=3)
        self.check_score(metric, TinyEncoder(provenance["model"]), (1024, 3))


if __name__ == "__main__":
    unittest.main()
