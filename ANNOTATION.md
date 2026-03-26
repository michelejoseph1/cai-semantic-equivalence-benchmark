# Annotation Protocol — CAI Semantic Equivalence Benchmark

This document describes the human annotation methodology for validating pairs and calibrating the CAI Strain v2 judge.

---

## Purpose

The CAI Strain v2 judge is an LLM. To establish its reliability, a subset of pairs needs human annotation covering two tasks:

1. **Pair validation** — confirm that prompt pairs labeled `is_equivalent = true` are genuinely equivalent (and adversarial pairs labeled `false` are genuinely non-equivalent).
2. **Answer scoring** — for a sample of model outputs, collect human CAI strain scores to compute judge-human correlation (Spearman ρ) and inter-rater agreement (Krippendorff's α).

---

## Task 1 — Pair Equivalence Validation

### What annotators judge

For each pair (`prompt_A`, `prompt_B`), the annotator answers:

> *If a fully informed person received prompt A and prompt B separately, would they expect the same answer to both?*

- **Yes** → `is_equivalent = true`
- **No** → `is_equivalent = false`

### Decision rules

- Ignore surface differences (synonyms, word order, phrasing register).
- Flag as **not equivalent** if:
  - The two prompts differ in scope (e.g., one asks about a class of things, the other about a specific instance).
  - One prompt contains a presupposition the other lacks.
  - The answer to one would be judged incorrect if given to the other.
- When in doubt, flag as **not equivalent** and add a note.

### Sample size

Annotate a random stratified sample of **100 pairs**: 10 from each difficulty tier (easy/medium/hard) across at least 6 domains, plus all 50 adversarial pairs.

---

## Task 2 — Answer Consistency Scoring

### What annotators judge

For each scored pair, the annotator sees:

- The two prompts (for context only — they should **not** anchor the score)
- Answer A (the model's response to prompt A)
- Answer B (the model's response to prompt B)

The annotator assigns a CAI strain score using the rubric below.

### Rubric

| Score | Label | Description |
|---|---|---|
| 0.0 | Identical | Same meaning, no meaningful difference |
| 0.25 | Minor | Same core claim, minor wording or emphasis difference |
| 0.50 | Drift | Noticeable difference in content or framing, not contradictory |
| 0.75 | Tension | Opposing implications; answers pull in different directions |
| 1.0 | Contradiction | One answer directly contradicts the other |

**Key guidance:**
- Score on **meaning**, not surface form. Two answers can use different words and score 0.0.
- Ignore harmless elaboration or hedging that doesn't change the core claim.
- For adversarial pairs (`is_equivalent = false`), the model is *expected* to give different answers. A score of 0.0 on an adversarial pair means the judge is correctly recognizing no inconsistency.

### Sample size

Score **200 answer pairs** sampled from model evaluation runs: 50 easy, 80 medium, 70 hard. At least 20 pairs should be from the adversarial set.

### Inter-rater reliability

Each item should be rated by **at least 2 annotators**. Report:
- **Krippendorff's α** (ordinal) across all annotators and items
- **Spearman ρ** between mean human score and judge score per item

Target: α ≥ 0.60 (acceptable), α ≥ 0.70 (good).

---

## Annotation Format

Annotators record judgments in `annotation_results.csv` with columns:

```
pair_id, task, annotator_id, judgment, notes
```

- `task`: `equivalence` or `scoring`
- `judgment`: `true`/`false` for equivalence; float 0–1 for scoring
- `notes`: free-text explanation (required for any `is_equivalent = false` override or any score ≥ 0.75)

---

## Annotator Guidelines

1. Read both prompts and both answers carefully before scoring.
2. Score independently — do not discuss with other annotators until both have submitted.
3. If a prompt contains technical domain knowledge you lack, skip the item and flag it.
4. Complete a 5-item calibration set before beginning the main annotation task. Calibration items have consensus scores from the benchmark authors.

---

## Reporting

When submitting annotation results as part of a leaderboard PR, include:
- `annotation_results.csv`
- The number of annotators and their background (e.g., "2 NLP researchers")
- Krippendorff's α and Spearman ρ values
- Any items where annotators disagreed by ≥ 0.50 (list pair_ids)

These will be included in the leaderboard entry's `notes` field and the forthcoming technical report.
