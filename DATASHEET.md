# Dataset Datasheet — CAI Semantic Equivalence Benchmark

*Following the Datasheets for Datasets framework (Gebru et al., 2021).*

---

## Motivation

**For what purpose was the dataset created?**
To measure a specific failure mode in language models: CAI failure — giving contradictory answers to two prompts that mean the same thing. Most evaluation benchmarks test factual correctness or instruction-following; none systematically measure consistency under semantic-preserving paraphrase. This benchmark fills that gap.

**Who created the dataset?**
Michele Joseph, independently.

**Who funded the dataset creation?**
Self-funded.

---

## Composition

**What do the instances represent?**
Each instance is a prompt pair: two natural-language prompts with the same intended meaning but different surface form, tagged with a domain and difficulty tier. A subset of instances are adversarial pairs (prompts that look similar but are not semantically equivalent) used to calibrate the scoring judge.

**How many instances are there?**
560 total: 510 equivalent pairs (is_equivalent = true) and 50 adversarial calibration pairs (is_equivalent = false).

**What data does each instance consist of?**

| Field | Type | Description |
|---|---|---|
| pair_id | integer | Unique identifier |
| domain | string | Thematic category (19 values) |
| prompt_A | string | First phrasing |
| prompt_B | string | Second phrasing |
| difficulty | easy/medium/hard | Paraphrase complexity tier |
| is_equivalent | true/false | false = adversarial calibration pair |

No model outputs are included in the dataset. Model outputs are generated at evaluation time.

**Is there a label or target?**
For equivalent pairs, the implicit ground truth is that a consistent model should produce the same answer to both prompts. For adversarial pairs, a correct model should produce different answers. The is_equivalent field encodes this.

**Are there recommended data splits?**
No train/validation/test split. This is an evaluation-only dataset. Do not fine-tune models on these prompts — doing so contaminates the benchmark.

**Does the dataset contain offensive content?**
The ethics domain contains moral dilemma prompts (trolley problem variants, not graphic). The ai_safety domain contains prompts about AI refusal behavior. No hate speech or explicit content.

---

## Collection Process

**How was the data collected?**
Prompts were written by the dataset author. Pair construction methodology by difficulty tier:

- **Easy**: synonym substitution, word-order change, question type shift (how/what/which)
- **Medium**: structural paraphrase, passive/active flip, negation equivalent
- **Hard**: speech act variation (interrogative vs. conditional), presupposition shift, implied vs. explicit framing
- **Policy domains**: additionally vary formality register and emotional register to reflect real-world user rephrase patterns observed in deployed chatbots
- **Adversarial**: surface-similar but semantically divergent, differing in scope, referent, or pragmatic presupposition

**Who collected the data?**
Michele Joseph, 2025-2026.

---

## Uses

**Intended uses:**
- Evaluating representational stability of LLMs under paraphrase
- Calibrating LLM-as-judge consistency scoring systems
- Research on prompt sensitivity and surface form brittleness
- Analyzing domain-specific failure modes in deployed LLM applications

**Limitations:**
- The dataset reflects one author's judgment of semantic equivalence. Edge cases may not generalize across cultures or specialized domains.
- Policy pairs use simplified phrasing. Real deployed systems face considerably more varied rephrasings.
- Multi-turn context is not included; consistency across conversation turns is a separate dimension.
- Difficulty labels for v0.1-v0.3 pairs are assigned retroactively as easy pending audit.

**Do not use for:**
- Training language models (contaminates the benchmark)
- Adversarial attacks (adversarial pairs are calibration tools, not attack templates)

---

## Distribution

- GitHub: https://github.com/michelejoseph1/cai-semantic-equivalence-benchmark
- Hugging Face: https://huggingface.co/datasets/compressionawareintelligence/cai-semantic-equivalence-benchmark
- License: MIT
- No third-party IP restrictions. All prompts are original.

---

## Maintenance

**Maintainer:** Michele Joseph (michelejoseph1 on GitHub)

**How to contribute:** Pull request to the GitHub repository. See CONTRIBUTING.md.

**Version history:** v0.1 (300 pairs), v0.2 (380 pairs), v0.3 (420 pairs), v0.4 (560 pairs). All versions preserved via git tags.

**Citation:**

```bibtex
@misc{joseph2025cai,
  title  = {CAI Semantic Equivalence Benchmark},
  author = {Joseph, Michele},
  year   = {2025},
  url    = {https://github.com/michelejoseph1/cai-semantic-equivalence-benchmark},
  note   = {v0.4}
}
```
