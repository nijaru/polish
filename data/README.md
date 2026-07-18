# Data policy

Datasets are not committed by default. Keep local copies under `data/raw/` or
`data/processed/`; those paths are ignored by git. The tracked dataset manifest
contract is defined in
[`dataset-manifest.schema.json`](dataset-manifest.schema.json) and validated
with:

```bash
polish validate-data --manifest data/manifests/authored-seed-v0.json
```

Every dataset used in an experiment needs a tracked manifest containing:

- source URL or archive identifier;
- acquisition date;
- license and permitted use;
- preprocessing steps;
- deduplication and filtering decisions;
- train, validation, and holdout split identifiers.

Training examples should be stored as JSONL outside Git and follow
[`training-record.schema.json`](training-record.schema.json). Each record keeps
the brief or draft separate from the target passage, so we can evaluate
generation, revision, and instruction adherence independently. Provenance is
per record rather than only per file because a corpus may combine sources with
different licenses.

The manifest is the data gate. It records the source and rights policy,
preprocessing and deduplication decisions, a deliberate train/validation/holdout
policy, and a SHA-256 reference for every local record. The validator reads the
ignored JSONL, checks each record against the tracked provenance, rejects
duplicate content across splits, and requires holdout to be sealed and excluded
from training and model selection.

The minimum useful training record is:

```json
{
  "record_id": "source-a-0001",
  "split": "train",
  "source": {
    "id": "source-a",
    "kind": "public-domain",
    "origin": "<source URL or archive identifier>",
    "license": "<license>",
    "rights_note": "<why this use is permitted>",
    "acquired_at": "2026-07-16"
  },
  "task": "revision",
  "input": {"brief": "<brief>", "draft": "<rough draft>"},
  "target": "<complete target passage>",
  "metadata": {"genre": "<genre>"}
}
```

Do not add private writing, credentials, scraped material, or copyrighted text
without a documented right to use it.

`authored-seed-v0.json` is a metadata-only tracked manifest for the first small
project-authored corpus slice: 18 records across generation and revision
tasks. Its raw examples live under the ignored `data/raw/authored-seed-v0/`
path, are licensed in the manifest for local research/SFT use, and are
intentionally not committed.

`writing-seed-v1.json` combines that fixture with 18 paragraph-level excerpts
from three pinned Standard Ebooks repositories. The source records identify
the title, repository commit, XHTML path, paragraph number, and local
transformation. Standard Ebooks describes its editions as based on works
believed to be in the U.S. public domain and dedicates its own edition work to
CC0; see its [public-domain policy](https://standardebooks.org/about/standard-ebooks-and-the-public-domain).
The combined local slice has 15 train, 10 validation, and 11 holdout records.
The raw JSONL remains ignored and is rebuilt with
`scripts/prepare_public_domain_seed.py` from pinned source checkouts.
