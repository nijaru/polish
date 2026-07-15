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

Do not add private writing, credentials, scraped material, or copyrighted text
without a documented right to use it.
