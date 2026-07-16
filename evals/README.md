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

The contract is deliberately runtime-agnostic. A Gemma, Mellum, or Qwen run
uses the same prompt manifest and report builder; only the capture adapter and
model metadata change. This lets us compare model size, quantization, and
fine-tuning without changing the experiment itself.

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
Each recorded-output file contains model identity and one output per case. The
`text` field is the user-facing response used for metrics; runtimes that emit
internal reasoning may also retain their exact output in `raw_text`:

```json
{
  "schema_version": "0.1",
  "manifest_revision": "holdout-2026-07-15",
  "split": "holdout",
  "settings": {
    "max_new_tokens": 768,
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": 0
  },
  "model": {
    "id": "google/gemma-4-E4B-it",
    "revision": "<commit-or-tag>",
    "quantization": "<serving-quantization>"
  },
  "outputs": [{"case_id": "case-001", "text": "...", "raw_text": "...", "finish_reason": "stop"}]
}
```

The runtime-specific capture script is intentionally outside this package.
Whatever local runtime is used must write this contract with the exact model
revision and quantization. The evaluator rejects recorded files whose manifest
revision, split, generation settings, or case set do not match the requested
run.

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
The default capture explicitly disables the Gemma chat template's separate
thinking channel so the token budget measures the requested writing; use
`--enable-thinking` only for a deliberate reasoning-channel experiment.

## Apple Silicon capture

The first local runtime is MLX-LM. It uses Hugging Face Hub model IDs and
downloads weights into the normal Hugging Face cache; model weights never belong
in this repository. Create the regular-CPython MLX environment once, then run
the capture script from the repository root:

```bash
uv --config-file /dev/null venv --python /opt/homebrew/bin/python3.14 .venv-mlx
uv --config-file /dev/null pip install --python .venv-mlx/bin/python 'mlx-lm>=0.31.3'
HF_HUB_DISABLE_XET=1 PYTHONPATH=src .venv-mlx/bin/python scripts/capture_mlx.py \
  --model mlx-community/unsloth-gemma-4-E4B-it-qat-oQ4 \
  --model-id mlx-community/unsloth-gemma-4-E4B-it-qat-oQ4 \
  --quantization mlx-qat-oQ4 \
  --manifest evals/manifests/public-v1.json \
  --split validation \
  --output runs/gemma-e4b/public-validation.json
```

The Gemma 4 12B OptiQ artifact in the model catalog uses the unified Gemma 4
architecture. It needs the optional `mlx-optiq` registrar and MLX-LM from the
source tree; the PyPI `mlx-lm==0.31.3` release alone does not expose that model
type. The runtime used for the 12B comparison was installed with:

```bash
uv --config-file /dev/null pip install --python .venv-mlx/bin/python \
  'mlx-optiq>=0.3.5' \
  'mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git@15b522f593b7ca5fbc0cac6f7572d40859d2d8fe'
```

`capture_mlx.py` imports `optiq` when it is available, so the same script works
for the text-only E4B artifact and the unified 12B artifact. The exact runtime
and model revision remain part of each recorded output.

The script resolves and records the Hub commit, exact generation settings,
runtime details, and one output for every case in the selected split. The
public manifest is a bring-up set, not the final sealed holdout; final model
selection should use a private holdout manifest kept outside Git.

`HF_HUB_DISABLE_XET=1` is included because the first E4B download stalled on
this Mac while using the Xet transfer path; the normal HTTP Hub transfer
resumed from the already cached shard and completed. Remove it if the local
Hub transfer is healthy.
