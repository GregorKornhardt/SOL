# Example texts

These four compressed JSONL files supply one LM1B comparison and one
OpenWebText comparison. Each contains 32 documents with a single `text` field:
the first 32 rows of the corresponding frozen 5k corpora in
[`../validation_data`](../validation_data/README.md).

| File | Contents |
| --- | --- |
| `lm1b_reference.jsonl.gz` | Held-out LM1B sentences, packed in the generator's training format |
| `lm1b_generated.jsonl.gz` | Samples from the LM1B spherical (vMF) flow model |
| `owt_reference.jsonl.gz` | OpenWebText reference G documents, the paper's OWT reference |
| `owt_generated.jsonl.gz` | Samples from the released MDLM OpenWebText checkpoint, exact sampling |

The text is copied unchanged; token IDs and other unused fields are omitted.
Source files, hashes, row selections and bundled file hashes are recorded in
`corpora.json`. Runtime code reads only the bundled files.

The examples retokenize these strings with pinned GPT-2 Large and truncate at
1,024 tokens by default. This small selection demonstrates the API and does
not establish model rankings or dataset-level quality. Generated files are
existing samples; no generation code or model weights are included.

The Apache-2.0 software license applies to code, not to third-party source text.
The LM1B and OpenWebText inputs retain their source-corpus terms. Preserve this
provenance when reusing or redistributing the example inputs.
