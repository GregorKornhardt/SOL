"""Offline adapter integration with real, tiny transformer architectures."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sol_metric import SOL, TextEncoder
from sol_metric.text import position_limit

try:
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedTokenizerFast

    HAS_TEXT = True
except ImportError:
    HAS_TEXT = False


class PositionLimitTests(unittest.TestCase):
    def test_position_limit_is_the_smallest_known_limit(self):
        sentinel = int(1e30)  # Tokenizers without a limit report this.
        cases = {
            "GPT-2": ({"max_position_embeddings": 1024}, 1024, 1024),
            "Qwen3, larger tokenizer limit": ({"max_position_embeddings": 32768}, 131072, 32768),
            "position offset": ({"max_position_embeddings": 514}, 512, 512),
            "tokenizer without limit": ({"max_position_embeddings": 32}, sentinel, 32),
            # Dream's config inherits 131,072 positions; its README documents 2048.
            "Dream": ({"max_position_embeddings": 131072, "model_type": "Dream"}, 131072, 2048),
            "unknown": ({}, sentinel, None),
        }
        for name, (config, tokenizer_limit, expected) in cases.items():
            with self.subTest(name):
                tokenizer = SimpleNamespace(model_max_length=tokenizer_limit)
                self.assertEqual(position_limit(SimpleNamespace(**config), tokenizer), expected)


@unittest.skipUnless(HAS_TEXT, "Optional text dependencies unavailable")
class ArchitectureTests(unittest.TestCase):
    def check_architecture(self, model_type):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocabulary = {"[UNK]": 0, "[EOS]": 1, "one": 2, "two": 3, "three": 4, "four": 5}
            tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]"
            )
            tokenizer.save_pretrained(root)
            if model_type == "gpt2":
                options = dict(n_embd=16, n_layer=1, n_head=2, n_positions=32)
            else:
                options = dict(
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    max_position_embeddings=32,
                )
            config = AutoConfig.for_model(
                model_type, vocab_size=len(vocabulary), eos_token_id=1, **options
            )
            torch.manual_seed(21)
            AutoModel.from_config(config).save_pretrained(root, safe_serialization=True)

            def encoder(batch_size):
                return TextEncoder(
                    str(root),
                    device="cpu",
                    max_length=4,
                    batch_size=batch_size,
                    local_files_only=True,
                )

            texts = ["one", "one two three four", "two three"]
            single = list(encoder(1)(texts))
            batched_encoder = encoder(8)
            batched = list(batched_encoder(texts))
            self.assertEqual([value.shape for value in batched], [(1, 16), (4, 16), (2, 16)])
            for left, right in zip(single, batched):
                np.testing.assert_allclose(left, right, atol=2e-6, rtol=2e-5)
            self.assertEqual(
                batched_encoder.tokenizer.pad_token_id, batched_encoder.tokenizer.eos_token_id
            )
            self.assertEqual(batched_encoder.provenance["model_type"], model_type)
            metric = SOL(n_directions=3, n_projections=4, n_quantiles=8, device="cpu")
            score = metric.from_texts(texts, ["four three", "one two"], encoder=batched_encoder)
            self.assertGreater(score, 0)
            left = metric.featurize_texts(texts, encoder=batched_encoder)
            right = metric.featurize_texts(["four three", "one two"], encoder=batched_encoder)
            self.assertAlmostEqual(metric.from_features(left, right), score, places=6)

    def test_gpt2(self):
        self.check_architecture("gpt2")

    def test_olmo2(self):
        self.check_architecture("olmo2")

    def test_qwen3(self):
        self.check_architecture("qwen3")

    def test_incompatible_transformers_fails_before_checkpoint_access(self):
        for version in ("4.48.0", "4.51.0", "5.0.0"):
            with self.subTest(version=version), patch("transformers.__version__", version):
                with patch.object(AutoTokenizer, "from_pretrained") as load:
                    with self.assertRaisesRegex(ImportError, "transformers>=4.51.3,<5"):
                        list(TextEncoder("missing-checkpoint", device="cpu")(["one"]))
                    load.assert_not_called()

    def test_gpu_weights_load_directly_when_accelerate_is_available(self):
        find_spec = importlib.util.find_spec
        for accelerate in (True, False):
            with self.subTest(accelerate=accelerate):
                model = MagicMock(config=SimpleNamespace())
                model.to.return_value = model
                tokenizer = MagicMock(pad_token_id=0, init_kwargs={})

                def spec(name, *args):
                    if name == "accelerate":
                        return object() if accelerate else None
                    return find_spec(name, *args)

                with (
                    patch("sol_metric.text.resolve_device", return_value="cuda:0"),
                    patch("importlib.util.find_spec", side_effect=spec),
                    patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer),
                    patch.object(AutoModel, "from_pretrained", return_value=model) as load,
                ):
                    TextEncoder("checkpoint", device="cuda:0")._load()
                if accelerate:
                    self.assertEqual(load.call_args.kwargs["device_map"], {"": "cuda:0"})
                    model.to.assert_not_called()
                else:
                    self.assertNotIn("device_map", load.call_args.kwargs)
                    model.to.assert_called_once_with("cuda:0")
                model.eval.assert_called_once_with()

    def test_auto_precision_is_fp16_on_cuda_and_fp32_on_cpu(self):
        for device, requested, expected in (
            ("cuda:0", "auto", "float16"),
            ("cuda:0", "float32", "float32"),
            ("cpu", "auto", "float32"),
        ):
            with self.subTest(device=device, requested=requested):
                model = MagicMock(config=SimpleNamespace())
                model.to.return_value = model
                tokenizer = MagicMock(pad_token_id=0, init_kwargs={})
                with (
                    patch("sol_metric.text.resolve_device", return_value=device),
                    patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer),
                    patch.object(AutoModel, "from_pretrained", return_value=model) as load,
                ):
                    encoder = TextEncoder("checkpoint", device=device, dtype=requested)
                    self.assertEqual(encoder.provenance["dtype"], requested)
                    encoder._load()
                options = load.call_args.kwargs
                loaded = options.get("dtype", options.get("torch_dtype"))
                self.assertEqual(loaded, getattr(torch, expected))
                self.assertEqual(encoder.provenance["dtype"], expected)

    def test_dream_wrapper_uses_base_states_and_excludes_padding(self):
        # Exercise the upstream Dream interface without bundling/downloading its
        # custom implementation or 7B weights. Native architectures above use
        # actual AutoModel checkpoints; this is an adapter-contract test.
        encoded = {
            "input_ids": torch.tensor([[2, 3, 1], [4, 1, 1]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 0, 0]]),
        }

        class Batch(dict):
            def to(self, device):
                return Batch({name: value.to(device) for name, value in self.items()})

        class TokenizerStub:
            def pad(self, items, **kwargs):
                return Batch(encoded)

        class DreamBase:
            def __call__(self, input_ids, attention_mask, use_cache):
                self.mask, self.use_cache = attention_mask, use_cache
                states = input_ids.float()[..., None].repeat(1, 1, 4)
                return SimpleNamespace(last_hidden_state=states)

        class DreamWrapper:
            config = SimpleNamespace(model_type="dream")
            model = DreamBase()

            def __call__(self, **kwargs):
                raise AssertionError("The vocabulary-logit wrapper must not run")

        encoder = TextEncoder("local-dream", device="cpu")
        encoder.model, encoder.tokenizer = DreamWrapper(), TokenizerStub()
        values = list(encoder._encode([{}, {}]))
        self.assertEqual([value.shape for value in values], [(3, 4), (1, 4)])
        np.testing.assert_array_equal(values[1], [[4, 4, 4, 4]])
        self.assertEqual(encoder.model.model.mask.shape, (2, 1, 1, 3))
        self.assertEqual(encoder.model.model.mask.dtype, torch.bool)
        self.assertFalse(encoder.model.model.use_cache)


if __name__ == "__main__":
    unittest.main()
