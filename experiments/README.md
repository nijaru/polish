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
