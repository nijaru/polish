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
