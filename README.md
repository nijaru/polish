# Polish

Research toolkit for improving local language-model writing.

Polish explores whether a model can produce stronger, less repetitive prose by
learning the distribution of high-quality writing, rather than relying only on
prompt instructions or phrase-level cleanup.

This repository is an early, public research project. It does not yet contain
a trained model or claim to reproduce any proprietary system. The first goal is
to make the experiments reproducible: fixed evaluation prompts, explicit data
provenance, and recorded training runs.

## Research direction

The working hypothesis is that writing quality is partly a distributional
property. A useful local system may therefore combine:

1. A carefully curated writing corpus with clear provenance.
2. Briefs, outlines, constraints, and complete target documents.
3. Supervised fine-tuning or LoRA for the basic writing behavior.
4. Grouped generation and distribution-aware scoring for refinement.
5. Human and model-based evaluation of specificity, coherence, detail,
   repetition, and factuality.

The exact training objective is an open research question here. We will label
our methods as independent implementations and report what actually works.

## What this is not

- An attempt to evade AI detectors.
- A copy of Deft or Rosmine's private implementation.
- A place to commit model checkpoints, private data, credentials, or scraped
  material without a documented license.

## Quick start

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

## Repository layout

```text
src/polish/       Python package and future CLI
tests/            Fast unit and contract tests
evals/            Fixed evaluation harness specification
experiments/      Run notes and experiment manifests
data/             Provenance docs; raw data remains local
docs/             Research notes and design decisions
```

## Inspiration and independence

The project is inspired by public discussion of distribution-aware fine-tuning
for writing, including Rosmine's public work on Distribution Fine-Tuning and
the Deft writing product. It is independently implemented, unaffiliated, and
does not use private beta information or proprietary code. See
[`SOURCES.md`](SOURCES.md) for provenance and attribution.

## License

The original code in this repository is released under the MIT License. Model
weights, datasets, and third-party assets remain subject to their own licenses.
