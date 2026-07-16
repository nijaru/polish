# Data policy

Datasets are not committed by default. Keep local copies under `data/raw/` or
`data/processed/`; those paths are ignored by git.

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
