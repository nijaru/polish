# Evaluation contract

The evaluation harness is a first-class part of this project. Training changes
are not useful unless they improve writing quality on prompts and documents
that were held out from tuning.

The initial harness should keep these sets separate:

- **Development:** prompts used for iteration and debugging.
- **Validation:** prompts used for model and hyperparameter selection.
- **Holdout:** sealed prompts used only for final comparisons.

Each evaluation run should record:

- model revision and quantization;
- prompt-set revision;
- generation settings and random seeds;
- scorer versions;
- raw outputs and aggregate metrics;
- human judgments when available.

The project will measure more than phrase frequency. Candidate dimensions
include specificity, meaningful detail, coherence, factuality, repetition,
sentence and paragraph variety, instruction adherence, and reader preference.

Detector scores are explicitly out of scope as an optimization target.

## Versioned manifest

Manifests are JSON files with a stable revision and an explicit split on every
case. The prompt text is intentionally part of the manifest so a run can be
audited against the exact input set.

```json
{
  "schema_version": "0.1",
  "revision": "holdout-2026-07-15",
  "cases": [
    {
      "id": "case-001",
      "split": "holdout",
      "prompt": "Write a 500-word profile of a neighborhood repair shop.",
      "metadata": {"genre": "profile"}
    }
  ]
}
```

Do not commit private holdout prompts or raw datasets. Keep private manifests
outside the repository and record only their revision in reports.

## Recorded-output reports

The first runner consumes outputs captured by any local model runtime. This
keeps the evaluation contract independent of a particular inference library.
Each recorded-output file contains model identity and one output per case:

```json
{
  "schema_version": "0.1",
  "manifest_revision": "holdout-2026-07-15",
  "split": "holdout",
  "settings": {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": 0
  },
  "model": {
    "id": "google/gemma-4-E4B-it",
    "revision": "<commit-or-tag>",
    "quantization": "<serving-quantization>"
  },
  "outputs": [{"case_id": "case-001", "text": "..."}]
}
```

The runtime-specific capture script is intentionally outside this package.
Whatever local runtime is used must write this contract with the exact model
revision and quantization. The evaluator rejects recorded files whose manifest
revision, split, or generation settings do not match the requested run.

Once two model-output files have been captured for the same manifest, create a
stable JSON report with:

```bash
uv --config-file /dev/null run polish evaluate \
  --manifest /path/to/holdout.json \
  --recorded-output /path/to/gemma-e4b.json \
  --recorded-output /path/to/gemma-12b.json \
  --output runs/baseline/report.json \
  --seed 0
```

The report records model revisions, quantization, generation settings, raw
outputs, and transparent surface measurements. These measurements are
descriptive baselines, not a quality score or a substitute for human review.
