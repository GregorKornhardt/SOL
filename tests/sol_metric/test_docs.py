"""README and guide examples run, with a tiny local model in place of real checkpoints."""

import math
import re
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import sol_metric
import sol_metric.text

ROOT = Path(__file__).resolve().parents[2]
# Examples using these need Torch and Transformers; NumPy-only runs skip them.
TEXT_APIS = ("TextEncoder", "from_texts", "featurize_texts")

try:
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast

    HAS_TEXT = True
except ImportError:
    HAS_TEXT = False


def python_blocks(name):
    """(line, code) for each ```python block of a Markdown file."""
    text = (ROOT / name).read_text(encoding="utf-8")
    return [
        (text[: match.start()].count("\n") + 1, match[1])
        for match in re.finditer(r"^```python\n(.*?)^```", text, re.DOTALL | re.MULTILINE)
    ]


class DocumentationExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model = Path(cls.tmp.name)
        if not HAS_TEXT:
            return
        (cls.model / "vocab.txt").write_text(
            "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nthe\ntrain\nrain\nnight\n", encoding="utf-8"
        )
        tokenizer = BertTokenizerFast(vocab_file=str(cls.model / "vocab.txt"))
        tokenizer.save_pretrained(cls.model)
        torch.manual_seed(0)
        config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=24,
            max_position_embeddings=512,
        )
        BertModel(config).save_pretrained(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def tiny_checkpoints(self):
        """Replace every requested checkpoint, including the Dream default, with the tiny one."""
        model = str(self.model)

        class TinyEncoder(sol_metric.text.TextEncoder):
            def __init__(self, model_name=None, **kwargs):
                for option in ("revision", "local_files_only", "trust_remote_code"):
                    kwargs.pop(option, None)
                super().__init__(model, local_files_only=True, **kwargs)

        stack = ExitStack()
        stack.enter_context(patch.object(sol_metric, "TextEncoder", TinyEncoder))
        stack.enter_context(patch.object(sol_metric.text, "TextEncoder", TinyEncoder))
        return stack

    def run_blocks(self, name, blocks, namespace):
        for line, code in blocks:
            with self.subTest(example=f"{name}:{line}"):
                output = StringIO()
                with redirect_stdout(output):
                    exec(compile(code, f"{name}:{line}", "exec"), namespace)
                for printed in output.getvalue().splitlines():
                    try:
                        score = float(printed)
                    except ValueError:
                        continue
                    self.assertTrue(math.isfinite(score) and score >= 0, printed)

    def runnable(self, blocks):
        if HAS_TEXT:
            return blocks
        return [block for block in blocks if not any(api in block[1] for api in TEXT_APIS)]

    def test_readme_examples_run_in_order(self):
        # Later README examples reuse names from earlier ones, as a reader would.
        blocks = python_blocks("README.md")
        self.assertTrue(blocks)
        with self.tiny_checkpoints():
            self.run_blocks("README.md", self.runnable(blocks), {"__name__": "__readme__"})

    def test_complete_guide_examples_run(self):
        # Guide blocks that start with an import are complete examples; the
        # others sketch call signatures with placeholder names.
        for name in ("docs/API.md", "docs/MODELS.md"):
            blocks = [b for b in python_blocks(name) if b[1].startswith(("import ", "from "))]
            with self.tiny_checkpoints():
                for block in self.runnable(blocks):
                    self.run_blocks(name, [block], {"__name__": "__guide__"})


if __name__ == "__main__":
    unittest.main()
