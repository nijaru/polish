# Blind human review

Automatic surface metrics describe outputs but do not decide whether one
passage is better. Polish therefore uses a deterministic A/B packet for the
final comparison.

Build a packet from two reports for the same manifest and keep the key out of
the reviewer's working view:

```bash
.venv/bin/python scripts/build_blind_review.py \
  --manifest evals/manifests/writing-seed-v1.json \
  --report runs/gemma4b-lora-v1/holdout-report.json \
  --output runs/gemma4b-lora-v1/blind-review.json \
  --key-output /tmp/polish-gemma4b-lora-v1-review-key.json \
  --seed 0
```

Render the JSON as a readable local Markdown document for review:

```bash
.venv/bin/python scripts/render_blind_review.py \
  --input runs/gemma4b-lora-v1/blind-review.json \
  --output runs/gemma4b-lora-v1/blind-review.md
```

The packet contains the prompt, two responses labeled only `A` and `B`, and
blank fields for pairwise preference, seven short rubric dimensions, and
notes. It contains no model IDs or reasoning traces. The key maps labels back
to model IDs only after the scores are recorded.

Use a 1-5 score for specificity, coherence, meaningful detail, repetition,
rhythm, instruction adherence, and overall quality. Choose `tie` or `neither`
when a forced winner would be misleading. Review holdout cases only after the
experiment and model-selection decisions are frozen.

For the 11-case `writing-seed-v1` holdout, the adapter advances only if it is
preferred on at least 7 cases, has no clear instruction-adherence or factuality
failure that the base avoids, and does not show a broad regression in the
dimension scores. This is a decision rule for this small gate, not a claim of
statistical proof.

The review packet is ignored under `runs/`; the rubric and packet builder are
tracked, while the raw outputs, filled scores, and unblinding key stay local.
