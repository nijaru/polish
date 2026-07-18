# Experiments

Store small, reviewable experiment manifests here. A manifest should identify
the data, model, training method, evaluation revision, and result location; it
should not contain credentials or large generated artifacts.

Prefer one hypothesis per experiment. Keep failed runs documented when they
change the next decision, especially regressions in factuality or general
instruction following.

The first adapter experiment is described by
[`gemma4b-lora-v0.json`](gemma4b-lora-v0.json) and the MLX-LM options are in
[`gemma4b-lora-v0.yaml`](gemma4b-lora-v0.yaml). Prepare its ignored training
directory with:

```bash
.venv/bin/python scripts/prepare_mlx_lora_data.py \
  --manifest data/manifests/authored-seed-v0.json \
  --output data/processed/gemma4b-lora-v0
```

The exporter writes only `train.jsonl` and `valid.jsonl`; it never copies the
holdout split. MLX-LM's `mask_prompt` option trains against target prose while
keeping the brief and any draft in the context.

The first expanded-corpus run is described by
[`gemma4b-lora-v1.json`](gemma4b-lora-v1.json). Rebuild its local records and
target-free evaluation manifest with:

```bash
.venv/bin/python scripts/prepare_public_domain_seed.py \
  --authored-records data/raw/authored-seed-v0/records.jsonl \
  --wind-root /path/to/kenneth-grahame_the-wind-in-the-willows \
  --pride-root /path/to/jane-austen_pride-and-prejudice \
  --alice-root /path/to/lewis-carroll_alices-adventures-in-wonderland \
  --output data/raw/writing-seed-v1/records.jsonl

PYTHONPATH=src .venv/bin/python scripts/build_data_manifest.py \
  --records data/raw/writing-seed-v1/records.jsonl \
  --data-root data/raw/writing-seed-v1 \
  --output data/manifests/writing-seed-v1.json \
  --dataset-id writing-seed --revision writing-seed-v1 \
  --record-schema data/training-record.schema.json

PYTHONPATH=src .venv/bin/python scripts/build_eval_manifest.py \
  --data-manifest data/manifests/writing-seed-v1.json \
  --split validation --split holdout \
  --output evals/manifests/writing-seed-v1.json
```

The v1 training config is intentionally a small, fixed LoRA surface. The
evaluation manifest contains prompts only; it never copies target passages.
Use `evals/HUMAN_REVIEW.md` to create the final blind holdout packet after
capturing both base and adapter outputs.
