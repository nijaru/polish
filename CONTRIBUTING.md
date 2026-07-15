# Contributing

Polish is an open research project. Contributions should make the result more
reproducible, more useful for real writing, or easier to evaluate.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

Use `uv add` for dependencies. Do not commit model checkpoints, raw datasets,
credentials, or generated artifacts. Include provenance for data and cite
public research that materially informs a method.

If a local `uv` policy sets `no-build = true`, the editable project cannot be
installed until that policy is explicitly bypassed. For this repository, use
`uv --config-file /dev/null` for the commands above after reviewing the
project's dependencies.

Please report regressions in factuality, specificity, coherence, or general
instruction following even when a change improves a narrower metric.
