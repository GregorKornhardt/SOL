"""Offline encoder integration: tiny local transformer, no model downloads."""

import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sol_metric import SOL, Features, TextEncoder


class ImportTests(unittest.TestCase):
    def test_custom_encoder_failure_cleans_temporary_storage(self):
        class FailingEncoder:
            provenance = {"model": "failing-encoder"}

            def __call__(self, texts):
                yield np.ones((2, 3), dtype=np.float32)
                raise RuntimeError("Encoding interrupted")

        with tempfile.TemporaryDirectory() as directory:
            metric = SOL(n_directions=2, n_projections=2, device="cpu", temp_dir=directory)
            metric.from_tokens(np.ones((2, 3)), [0, 1, 2], np.zeros((1, 3)), [0, 1])
            self.assertIsNotNone(metric.last_run)
            with self.assertRaisesRegex(RuntimeError, "Encoding interrupted"):
                metric.from_texts(["a", "b"], ["c"], encoder=FailingEncoder())
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertIsNone(metric.last_run)

    def test_defaults_to_dream_and_paper_revisions(self):
        dream = TextEncoder()
        self.assertEqual(dream.model_name, "Dream-org/Dream-v0-Base-7B")
        self.assertEqual(dream.revision, "6572adb5535263e4d1a337b56942ba48b6dee2a9")
        self.assertEqual(dream.provenance["revision"], dream.revision)
        self.assertTrue(dream.trust_remote_code)
        gpt2 = TextEncoder("gpt2-large")
        self.assertEqual(gpt2.revision, "32b71b12589c2f8d625668d2335a01cac3249519")
        self.assertFalse(gpt2.trust_remote_code)
        # Other revisions or models never trust remote code implicitly.
        latest = TextEncoder(revision="main")
        self.assertEqual(latest.revision, "main")
        self.assertFalse(latest.trust_remote_code)
        self.assertFalse(TextEncoder(trust_remote_code=False).trust_remote_code)
        other = TextEncoder("someone/encoder")
        self.assertIsNone(other.revision)
        self.assertFalse(other.trust_remote_code)

    def test_import_does_not_load_optional_dependencies(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import sol_metric; "
                "assert not {'torch', 'transformers', 'dsw'} & set(sys.modules)",
            ],
            check=True,
        )

    def test_custom_encoder_streaming_and_cleanup(self):
        class Encoder:
            provenance = {"model": "test-fixed-encoder", "revision": "1"}

            def __call__(self, texts):
                for text in texts:
                    yield np.array([[float(len(text)), 1]], dtype=np.float32)

        encoder = Encoder()
        with tempfile.TemporaryDirectory() as tmp:
            metric = SOL(n_directions=3, n_projections=4, n_quantiles=4, device="cpu", temp_dir=tmp)
            score = metric.from_texts(iter(["a", "bb"]), iter(["ccc"]), encoder=encoder)
            self.assertGreater(score, 0)
            self.assertEqual(list(Path(tmp).iterdir()), [])
            self.assertEqual(metric.last_run["encoder"], encoder.provenance)
            qx = metric.featurize_texts(["a", "bb"], encoder=encoder)
            qy = metric.featurize_texts(["ccc"], encoder=encoder)
            self.assertAlmostEqual(metric.from_features(qx, qy), score)
            with self.assertRaisesRegex(ValueError, "at least one document"):
                metric.featurize_texts([], encoder=encoder)
            self.assertEqual(list(Path(tmp).iterdir()), [])


class StreamingTests(unittest.TestCase):
    """Text features are binned per encoder batch and match the token-array path."""

    class Encoder:
        provenance = {"model": "test-random-encoder", "revision": "1"}

        def __init__(self):
            self.yielded = 0

        def __call__(self, texts):
            for text in texts:
                rng = np.random.default_rng(len(text))
                self.yielded += 1
                yield rng.normal(size=(len(text), 5)).astype(np.float32)

    texts_x = ["a", "bbbbbb", "ccc", "dddddddddd", "ee", "fffffff"]
    texts_y = ["gggg", "h", "iiiiiiiiiiii", "jjj"]

    def metric(self, **kwargs):
        return SOL(n_directions=7, n_projections=6, n_quantiles=4, device="cpu", **kwargs)

    def tokens(self, texts):
        states = list(self.Encoder()(texts))
        return np.concatenate(states), np.r_[0, np.cumsum([len(x) for x in states])]

    def test_streamed_scores_match_token_arrays_for_any_batching(self):
        expected = self.metric(dtype="float64").from_tokens(
            *self.tokens(self.texts_x), *self.tokens(self.texts_y)
        )
        for batch, tokens in ((1, 12), (2, 12), (3, 20), (64, 8192)):
            with self.subTest(batch_size=batch, max_tokens_per_batch=tokens):
                metric = self.metric(dtype="float64", batch_size=batch, max_tokens_per_batch=tokens)
                score = metric.from_texts(self.texts_x, self.texts_y, encoder=self.Encoder())
                self.assertAlmostEqual(score, expected, delta=1e-12 * expected)

    # Temporary directories may be on tmpfs; the RAM-backed warning is tested separately.
    @patch("sol_metric.metric.ram_backed", new=lambda path: False)
    def test_features_stay_in_memory_when_they_fit_and_spill_otherwise(self):
        # 10 documents x 7 directions x 4 bins x 4 bytes is about 1.1e-3 MiB.
        scores = {}
        for limit, storage in ((4096, "memory"), (1e-3, "disk"), (0, "disk")):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(limit=limit):
                metric = self.metric(feature_memory_mb=limit, temp_dir=tmp)
                written = []
                create = Features._create

                def record(shape, dtype, metadata, path, fill):
                    written.append(path is not None and Path(path).parent.parent == Path(tmp))
                    return create(shape, dtype, metadata, path, fill)

                with patch.object(Features, "_create", side_effect=record):
                    scores[limit] = metric.from_texts(
                        self.texts_x, self.texts_y, encoder=self.Encoder()
                    )
                self.assertEqual(metric.last_run["feature_storage"], storage)
                self.assertEqual(written, [storage == "disk"] * 2)
                self.assertEqual(list(Path(tmp).iterdir()), [])
        self.assertEqual(scores[4096], scores[1e-3])
        self.assertEqual(scores[4096], scores[0])

    @patch("sol_metric.metric.ram_backed", new=lambda path: False)
    def test_auto_limit_is_resolved_once_after_the_encoder_starts(self):
        # The 10 documents' features take 1,120 bytes; "auto" allows half the available memory.
        for available, storage in ((2240, "memory"), (2239, "disk")):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(available=available):
                encoder, calls = self.Encoder(), []

                def measure():
                    calls.append(encoder.yielded)
                    return available

                metric = self.metric(temp_dir=tmp)
                with patch("sol_metric.metric.available_memory", side_effect=measure):
                    metric.from_texts(self.texts_x, self.texts_y, encoder=encoder)
                self.assertEqual(len(calls), 1)
                self.assertGreaterEqual(calls[0], 1)
                self.assertEqual(metric.last_run["feature_storage"], storage)
                self.assertEqual(metric.last_run["feature_memory_mb"], available / 2 / 1024**2)

    def test_spilling_to_a_ram_backed_temp_dir_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("sol_metric.metric.ram_backed", return_value=True):
                metric = self.metric(feature_memory_mb=0, temp_dir=tmp)
                with self.assertWarnsRegex(RuntimeWarning, "RAM-backed"):
                    metric.from_texts(self.texts_x, self.texts_y, encoder=self.Encoder())
                metric = self.metric(feature_memory_mb=4096, temp_dir=tmp)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    metric.from_texts(self.texts_x, self.texts_y, encoder=self.Encoder())

    def test_features_are_written_while_encoding(self):
        encoder, metric = self.Encoder(), self.metric(batch_size=2)
        seen = []
        quantiles = metric.backend.quantiles

        def record(*args):
            seen.append(encoder.yielded)
            return quantiles(*args)

        with patch.object(metric.backend, "quantiles", side_effect=record):
            metric.featurize_texts(self.texts_x, encoder=encoder)
        # The first batch of two documents is binned once the third is encoded.
        self.assertEqual(seen[0], 3)
        self.assertEqual(encoder.yielded, len(self.texts_x))

    def test_default_encoder_is_created_once_and_reused(self):
        created = []
        base = self.Encoder

        class Default(base):
            def __init__(self, **options):
                super().__init__()
                created.append(options)

        with patch("sol_metric.text.TextEncoder", Default):
            metric = self.metric()
            first = metric.from_texts(self.texts_x, self.texts_y)
            second = metric.from_texts(self.texts_x, self.texts_y)
            metric.featurize_texts(self.texts_x)
        self.assertEqual(
            created, [{"device": "cpu", "batch_size": 8, "max_tokens_per_batch": 8192}]
        )
        self.assertEqual(first, second)
        self.assertEqual(metric.last_run["encoder"], base.provenance)
        # One batch size sets the default encoder's forward passes and binning.
        with patch("sol_metric.text.TextEncoder", Default):
            self.metric(batch_size=3, max_tokens_per_batch=99).featurize_texts(self.texts_x)
        self.assertEqual(
            created[-1], {"device": "cpu", "batch_size": 3, "max_tokens_per_batch": 99}
        )

    def test_documents_beyond_the_token_budget_are_binned_alone(self):
        # "dddddddddd" (10 tokens) and "iiiiiiiiiiii" (12) exceed the budget of 8.
        expected = self.metric(dtype="float64").from_texts(
            self.texts_x, self.texts_y, encoder=self.Encoder()
        )
        metric = self.metric(dtype="float64", max_tokens_per_batch=8)
        score = metric.from_texts(self.texts_x, self.texts_y, encoder=self.Encoder())
        self.assertAlmostEqual(score, expected, delta=1e-12 * expected)

    def test_rejects_count_mismatch_and_existing_caches(self):
        class Short(self.Encoder):
            def __call__(self, texts):
                yield from list(super().__call__(texts))[:-1]

        with self.assertRaisesRegex(ValueError, "yielded 5 documents for 6 texts"):
            self.metric().featurize_texts(self.texts_x, encoder=Short())
        with tempfile.TemporaryDirectory() as tmp:
            encoder = self.Encoder()
            with self.assertRaises(FileExistsError):
                self.metric().featurize_texts(self.texts_x, encoder=encoder, path=tmp)
            self.assertEqual(encoder.yielded, 0)


try:
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast

    HAS_TEXT = True
except ImportError:
    HAS_TEXT = False


@unittest.skipUnless(HAS_TEXT, "Optional text dependencies unavailable")
class EncoderTests(unittest.TestCase):
    device = "cpu"

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name)
        (cls.path / "vocab.txt").write_text(
            "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\none\ntwo\nthree\nfour\n", encoding="utf-8"
        )
        tok = BertTokenizerFast(vocab_file=str(cls.path / "vocab.txt"))
        tok.save_pretrained(cls.path)
        torch.manual_seed(10)
        model = BertModel(
            BertConfig(
                vocab_size=len(tok),
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=24,
                max_position_embeddings=32,
            )
        )
        model.save_pretrained(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def encoder(self, **kwargs):
        device = kwargs.pop("device", self.device)
        # FP32 keeps CPU and CUDA states comparable; the CUDA default is FP16.
        kwargs.setdefault("dtype", "float32")
        kwargs.setdefault("max_length", 8)
        return TextEncoder(str(self.path), device=device, local_files_only=True, **kwargs)

    def test_untruncated_documents_end_at_their_last_token(self):
        encoder = self.encoder(max_length=None, max_tokens_per_batch=64)
        states = list(encoder(["one " * 20, "two three"]))
        # [CLS] and [SEP] around every word; the tiny model has 32 positions.
        self.assertEqual([len(x) for x in states], [22, 4])
        self.assertIsNone(encoder.provenance["max_length"])
        truncated = list(self.encoder(max_length=22)(["one " * 20]))[0]
        np.testing.assert_allclose(states[0], truncated, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "supports 32 positions"):
            list(encoder(["one " * 40]))

    def test_auto_length_keeps_documents_whole_up_to_the_model_limit(self):
        self.assertEqual(TextEncoder(str(self.path)).max_length, "auto")
        encoder = self.encoder(max_length="auto")
        whole = list(self.encoder(max_length=None, max_tokens_per_batch=64)(["one " * 20]))[0]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            states = list(encoder(["one " * 20, "two three"]))
        self.assertFalse([w for w in caught if "positions" in str(w.message)])
        self.assertEqual([len(x) for x in states], [22, 4])
        np.testing.assert_allclose(states[0], whole, atol=1e-6)
        self.assertEqual(encoder.provenance["max_length"], "auto")
        self.assertEqual(encoder.provenance["position_limit"], 32)

    def test_auto_length_truncates_at_the_model_limit_with_one_warning(self):
        # 40 words plus [CLS] and [SEP] exceed the tiny model's 32 positions.
        encoder = self.encoder(max_length="auto")
        message = r"2 of 3 documents exceed the 32 positions .* \(longest: 52 tokens\)"
        with self.assertWarnsRegex(RuntimeWarning, message) as caught:
            states = list(encoder(["one " * 40, "two", "three " * 50]))
        self.assertEqual(len(caught.warnings), 1)
        self.assertEqual([len(x) for x in states], [32, 3, 32])
        expected = list(self.encoder(max_length=32)(["one " * 40]))[0]
        np.testing.assert_allclose(states[0], expected, atol=1e-6)

    def test_auto_length_encodes_documents_beyond_the_token_budget_alone(self):
        texts = ["one " * 20, "two", "three four"]
        expected = list(self.encoder(max_length=None, max_tokens_per_batch=64)(texts))
        states = list(self.encoder(max_length="auto", max_tokens_per_batch=8)(texts))
        self.assertEqual([len(x) for x in states], [22, 3, 4])
        for x, y in zip(states, expected):
            np.testing.assert_allclose(x, y, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "exceeds max_tokens_per_batch"):
            list(self.encoder(max_length=None, max_tokens_per_batch=8)(texts))

    def test_padding_batch_size_and_token_budget(self):
        texts = ["one", "one two three four", "two three"]
        single = list(self.encoder(batch_size=1)(texts))
        encoder = self.encoder(batch_size=8, max_tokens_per_batch=12)
        batched = list(encoder(texts))
        self.assertEqual([len(x) for x in batched], [3, 6, 4])
        for x, y in zip(single, batched):
            np.testing.assert_allclose(x, y, atol=1e-6)
        self.assertFalse(encoder.model.training)
        self.assertTrue(all(parameter.grad is None for parameter in encoder.model.parameters()))

    def test_text_score_and_validation(self):
        encoder = self.encoder(batch_size=2)
        m = SOL(n_directions=3, n_projections=4, n_quantiles=8, device=self.device)
        x, y = ["one", "two three"], ["three four"]
        score = m.from_texts(x, y, encoder=encoder)
        self.assertGreater(score, 0)
        self.assertEqual(m.last_run["encoder"]["max_length"], 8)
        self.assertEqual(m.from_texts(x, x, encoder=encoder), 0)
        self.assertEqual(len(list(encoder(["one " * 20]))[0]), 8)
        for texts in ([], [""], ["   "], [3]):
            with self.assertRaises(ValueError):
                list(encoder(texts))
        with self.assertRaises(ValueError):
            list(self.encoder(max_tokens_per_batch=2)(["one two"]))
        with self.assertRaises(TypeError):
            list(encoder("one two"))


@unittest.skipUnless(HAS_TEXT and torch.cuda.is_available(), "CUDA text dependencies unavailable")
class CudaEncoderTests(EncoderTests):
    device = "cuda:0"

    def test_cpu_gpu_encoder_agreement(self):
        texts = ["one", "two three four"]
        cpu = list(self.encoder(device="cpu")(texts))
        gpu = list(self.encoder()(texts))
        for a, b in zip(cpu, gpu):
            np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-5)

    def test_states_stay_on_the_metric_gpu(self):
        encoder = self.encoder()
        x, y = ["one", "two three four", "four"], ["three two", "one one"]
        resident = list(encoder._states(x, "cuda:0"))
        self.assertTrue(all(torch.is_tensor(s) and s.device.type == "cuda" for s in resident))
        self.assertTrue(all(isinstance(s, np.ndarray) for s in encoder(x)))

        class HostOnly:
            """Hides the TextEncoder type, forcing the NumPy host round trip."""

            provenance = encoder.provenance

            def __call__(self, texts):
                return encoder(texts)

        metric = SOL(n_directions=5, n_projections=6, n_quantiles=4, device=self.device)
        direct = metric.from_texts(x, y, encoder=encoder)
        self.assertEqual(metric.from_texts(x, y, encoder=HostOnly()), direct)

    def test_default_precision_is_fp16_on_cuda(self):
        texts = ["one", "two three four"]
        reference = list(self.encoder()(texts))
        encoder = self.encoder(dtype="auto")
        self.assertEqual(encoder.provenance["dtype"], "auto")
        states = list(encoder(texts))
        self.assertEqual(encoder.provenance["dtype"], "float16")
        self.assertEqual(next(encoder.model.parameters()).dtype, torch.float16)
        for a, b in zip(reference, states):
            self.assertEqual(b.dtype, np.float32)
            np.testing.assert_allclose(a, b, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
