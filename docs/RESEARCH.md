# Research framing

## Public starting point

Public discussion of Distribution Fine-Tuning describes a high-level goal:
move model outputs toward the distribution of better writing. The exact Deft
training algorithm is not public, so this project does not assume that a
particular loss, metric, or optimization loop is the original method.

## Questions to test

1. Does corpus quality matter more than model size for this task?
2. Which distributional signals predict human preference after controlling for
   factuality and instruction following?
3. Does grouped sampling provide a useful training signal beyond ordinary SFT?
4. Which parts belong in model training versus an inference-time editing loop?
5. How much quality survives quantization and local serving constraints?

## Decision rule

Prefer methods that improve held-out human judgments and concrete writing
criteria. Do not select a method because it merely reduces a detector score or
removes a hand-written list of allegedly suspicious phrases.
