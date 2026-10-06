# Frozen 5k validation pairs

Each dataset contains one reference corpus and one generated corpus, with
**5,000 nonempty texts in each arm**:

| File | Contents |
| --- | --- |
| `lm1b_reference.jsonl.gz` | Held-out LM1B sentences, packed in the generator's training format |
| `lm1b_generated.jsonl.gz` | Samples from the LM1B spherical (vMF) flow model: 64 predictor steps, one corrector step each |
| `owt_reference.jsonl.gz` | OpenWebText reference G, the paper's OWT reference |
| `owt_generated.jsonl.gz` | Samples from the released MDLM OpenWebText checkpoint, exact sampling (DDPM-cache, T=1000) |

The LM1B files keep the first 5,000 rows of their frozen source shards. The
OpenWebText files are the paper's GP-seed-0 subsets:
`numpy.random.default_rng(100).choice(10000, 5000, replace=False)` of reference G
and `default_rng(200).choice(5024, 5000, replace=False)` of the MDLM pool,
stored in ascending row order. The text is unchanged; token IDs and unused
fields are omitted. `corpora.json` records bundled file hashes, source files and
hashes, frozen paper shard names and the selected rows.

The four `.jsonl.gz` archives (~20 MB) are kept in the
[Git repository](https://github.com/GregorKornhardt/SOL) and its release tags.
The PyPI source distribution includes this README, `corpora.json` and
`validation.json`, but not the archives.

## Expected scores

`validation.json` fixes one protocol: pinned GPT-2 Large on retokenized text, a
1,024-token horizon, encoder batches of 8 documents (8,192 tokens), 1,024 token
directions and 1,024 GP draws, 64 bins, length scale 0.1, both seeds zero,
dimension scaling (1,280), FP32 metric arithmetic and disabled TF32.

| Dataset | FP16 encoder | FP32 encoder |
| --- | ---: | ---: |
| LM1B, vMF flow samples | 0.9618720932960572 | 0.961950296904681 |
| OpenWebText, MDLM exact sampling | 2.1674155071924597 | 2.1673446416141333 |

The FP16 values were computed by the paper's OpenWebText-table worker on these
exact texts. Its hidden-state extraction, bin averages, GP sampling and CUDA
transport kernel are independent of this package, whose own FP16 runs agree to
within 1e-10 relative. That worker encodes only in FP16, so the FP32 values are
this package's measurements and serve as a regression check.

For comparison, the paper reports 2.167356301995784 for this OpenWebText cell
(GPT-2 Large, direction seed 0, GP seed 0; 2.185 ± 0.028 over nine seed pairs).
That run encoded reference G from native GPT-2 token IDs rather than the decoded
text bundled here, a difference of 2.7e-5 relative.

The score tolerance is relative `1e-4` or absolute `2e-6`. FP16 states depend
slightly on batch composition, so the encoder batch settings are part of the
protocol. The reference environment is recorded in the manifest.

From the repository root:

```bash
python examples/compare_corpora.py lm1b --validate --device cuda:0
python examples/compare_corpora.py owt --validate --device cuda:0 --dtype float32
```

`--dtype auto`, the default, selects FP16 on CUDA and FP32 on CPU. Add
`--dry-run` to verify inputs without loading the encoder. A completed
validation prints `validation.passed: true`; mismatched inputs, protocol or
scores exit with an error. Use `--output PATH` to save the complete result and
`--check-result PATH` to check it later without re-encoding.

These fixtures validate the standalone package under one fixed protocol.
No generation code, generation checkpoint or experiment runner is included.
The source-corpus terms are separate from the Apache-2.0 software license.
