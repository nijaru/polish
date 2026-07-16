# Elocutio experiment plan

Elocutio is the provisional name for a future writing-focused model or
adapter. Polish remains the repository and research-tool name.

## First milestone

Build a reproducible local evaluation loop on Apple Silicon with Gemma 4 E4B:

1. Load a pinned Hugging Face artifact through MLX-LM.
2. Apply the model's chat template to a fixed prompt manifest.
3. Record raw outputs, model commit, quantization, runtime, and generation
   settings.
4. Produce a descriptive report with transparent surface measurements.
5. Compare later models through the same manifest and report contract.

The public manifest in `evals/manifests/public-v1.json` is only a bring-up
fixture. A final comparison needs a private holdout manifest that is not shown
to the models during tuning or selection.

## Current local comparison

On 2026-07-16, Gemma 4 E4B and Gemma 4 12B were captured on the same public
validation and holdout cases with `max_new_tokens=768`, `temperature=0.7`,
`top_p=0.95`, `seed=0`, and the chat template's separate thinking channel
disabled. Both models stopped normally on all four cases.

The 12B outputs were 296 and 151 words on validation, and 378 and 130 words
on holdout. E4B produced 252 and 134 words on validation, and 363 and 100
words on holdout. The holdout revision asks for 110-160 words, so E4B missed
that range while 12B landed inside it. These are useful instruction-adherence
signals, not a quality verdict; the reports still need human judgments.

The 12B OptiQ artifact requires the optional `mlx-optiq` registrar and MLX-LM
from the source tree. Its pinned Hub revision and runtime setup are recorded in
the local capture output and the model catalog; Qwen3.6 remains deferred.

## Model order and size judgment

The workflow is model-agnostic. We are not choosing a base model by intuition
and then treating its output as proof. We will run the same cases and record
whether each model improves the actual writing criteria.

- Gemma 4 E4B is the first student because it is small enough to iterate on this
  Mac and has a usable MLX artifact.
- Gemma 4 12B is the next comparison because it is still practical locally and
  tests whether the extra capacity materially improves prose.
- Mellum2 12B is a specialist challenger, not an automatic winner. Its code
  and technical training may help structure and precision, but that must be
  tested on writing and revision cases.
- Qwen3.6 27B is a strong general-purpose reference or later teacher. It is
  not too large for the task conceptually, but its roughly 20 GiB MLX artifact
  makes it expensive for rapid Mac iteration. Fedora can make this comparison
  more convenient later.

The current artifact choices and approximate download sizes are recorded in
`experiments/model-catalog.json`. They are snapshots, not permanent claims
about the Hub.

## Training workflow

The first loop is inference-only. That gives us a baseline and tells us what
the task actually needs before training changes obscure the diagnosis.

The likely sequence after the baseline is:

1. **Corpus and provenance.** Assemble a small, licensed corpus of strong
   writing, plus briefs, constraints, and complete target documents. Keep raw
   data local and record source, license, transformation, and split metadata.
2. **Ordinary SFT/LoRA.** Teach the base model to map a brief or rough draft to
   a useful finished passage. Start with a parameter-efficient adapter rather
   than a full checkpoint rewrite.
3. **Held-out evaluation.** Compare base versus adapter on the same prompts,
   with factuality and instruction adherence checked alongside style.
4. **Preference data.** Collect human pairwise choices and short rubric scores
   for specificity, meaningful detail, coherence, repetition, rhythm,
   factuality, and adherence. Human preference is the main quality signal.
5. **Preference optimization or grouped refinement.** Only if SFT leaves a
   measurable gap, test DPO/KTO or grouped generation with a documented
   scorer. These are experiments, not assumptions about the final method.

This means the resulting system may be a LoRA adapter, a fully fine-tuned
model, an inference-time editor, or a combination. We will name the artifact
according to what we actually train, for example
`Elocutio-Gemma4-12B-SFT`, rather than implying that a method worked before it
has been measured.

## Data sources and safeguards

The preferred data sources are:

- public-domain or clearly permissively licensed prose, verified at the source;
- user-provided writing for which we have permission to process it;
- authored briefs, outlines, revision prompts, and target passages;
- human-created revision pairs and preference judgments.

We should not use scraped text with unclear rights, private beta material, or
model-generated prose as unquestioned ground truth. Synthetic examples may be
useful as labeled augmentation, but they stay identifiable and must not replace
human quality checks.

## Decision gates

Each stage earns the next one:

- **Runtime gate:** a model can be loaded from a pinned Hub revision and its
  outputs can be reproduced from a manifest.
- **Baseline gate:** we have recorded outputs and a report before training.
- **Data gate:** every training example has provenance and a deliberate split.
- **SFT gate:** the adapter beats the base on held-out human preference without
  a regression in factuality or instruction adherence.
- **Preference gate:** a more complex objective earns its complexity through a
  consistent held-out improvement.

No detector score is an optimization target. The goal is better writing and
better revision for the intended reader.
