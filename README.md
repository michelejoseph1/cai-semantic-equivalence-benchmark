# CAI Semantic Equivalence Benchmark

A benchmark for measuring consistency under meaning-preserving paraphrase.

Most eval benchmarks ask: is this response correct? This one asks: does the model give the same answer when the question is rephrased? That gap — between surface accuracy and representational stability — is what CAI measures.

A **CAI failure** is when a model gives contradictory answers to two prompts that mean the same thing. The benchmark finds them and counts them.

---

## Dataset

**380 semantically equivalent prompt pairs** across 14 domains.

| Domain | Pairs | Notes |
|---|---|---|
| `factual` | 20 | General knowledge |
| `everyday_reasoning` | 20 | Common sense |
| `math_logic` | 20 | Arithmetic, algebra, probability |
| `counterfactual` | 20 | Hypothetical scenarios |
| `ethics` | 30 | Moral dilemmas |
| `ai_safety` | 20 | AI behavior and refusal |
| `practical_planning` | 20 | Scheduling, budgeting |
| `causal_reasoning` | 20 | Cause and effect |
| `philosophy` | 20 | Identity, consciousness, free will |
| `social_emotional` | 20 | Interpersonal scenarios |
| `creative_writing` | 20 | Imaginative prompts |
| `summarization` | 40 | Paraphrase and summarization |
| `cai_meta` | 30 | CAI theory and methodology |
| `policy` | 80 | **Ecommerce, HR, healthcare, legal** |

The `policy` domain is new in v0.2. These are the pairs where CAI failures matter most — policy-bound LLM systems that contradict themselves when a question is rephrased. Real support bots, HR assistants, healthcare chatbots, legal tools.

Each row: `pair_id, domain, prompt_A, prompt_B`

Formats: `dataset.csv`, `dataset.json`

---

## Results

| Model | Pairs | Avg CAI Strain | Date |
|---|---|---|---|
| gpt-4o | 300 | 0.3642 | 2025 |

Surface mismatch (string-level) for gpt-4o: 0.99. Meaningful semantic drift: 0.36. That gap is the point. A model can look inconsistent at the word level and still be coherent. And it can look fluent while holding internal contradictions.

More model results coming. See [Contributing](#contributing) to add yours.

---

## Scoring

CAI Strain v2 is a model-based judge that scores inconsistency between two answers on a 0–1 scale:

- `0.0` — same meaning, no drift
- `0.25` — minor wording differences, same claim
- `0.50` — noticeable drift, not contradictory
- `0.75` — tension, opposing implications
- `1.0` — direct contradiction

The judge focuses on meaning, not surface form.

**Run on OpenAI models:**

```bash
git clone https://github.com/michelejoseph1/cai-semantic-equivalence-benchmark.git
cd cai-semantic-equivalence-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
python evaluate_openai.py --model gpt-4o --max_pairs 380
```

**Run on Anthropic models:**

```bash
export ANTHROPIC_API_KEY="your-key"
python evaluate_anthropic.py --model claude-opus-4-6 --max_pairs 380
```

Results write to `results_{model}.csv` and append to `scores.csv`.

---

## Methodology

**Pair construction.** Each pair consists of two prompts with the same intended meaning, different surface form. Pairs were constructed to preserve factual content, logical structure, and intent while varying phrasing, syntactic order, vocabulary, and framing. Policy pairs additionally vary formality, presupposition, and emotional register — the variation patterns that most often trigger inconsistent responses in deployed systems.

**Semantic equivalence.** Two prompts are equivalent if a fully informed human would expect the same answer to both. This excludes cases where phrasing legitimately changes the scope or implication of the question.

**Scoring.** CAI Strain v2 uses a separate judge model to assess semantic consistency between two responses. The judge is instructed to focus on meaning, not wording. Scores are averaged across all pairs to produce a model-level CAI score.

**Limitations.** The judge is an LLM. LLM judges can disagree with humans on edge cases, particularly in the 0.25–0.75 range. We recommend treating aggregate scores as directional rather than precise. Per-domain scores are more informative than the overall average.

---

## Contributing

**Add model results.** Run the scorer, then open a PR adding your result to `leaderboard.json`:

```json
{
  "model": "your-model-name",
  "provider": "provider-name",
  "pairs": 380,
  "avg_cai_strain": 0.28,
  "date": "2026-03-25",
  "scorer": "evaluate_openai.py or evaluate_anthropic.py",
  "notes": "optional"
}
```

**Add pairs.** New pairs should follow the format in `dataset.csv`. Include a `domain` tag. Open a PR with a brief description of what the new pairs test. Policy pairs (ecommerce, HR, healthcare, legal, finance, insurance) are especially welcome.

---

## Citation

```bibtex
@misc{joseph2025cai,
  title   = {CAI Semantic Equivalence Benchmark},
  author  = {Joseph, Michele},
  year    = {2025},
  url     = {https://github.com/michelejoseph1/cai-semantic-equivalence-benchmark},
  note    = {v0.2}
}
```

Or see `CITATION.cff`.

---

## Related

- [contradish](https://github.com/michelejoseph/contradish) — CAI testing tool for LLM apps
- [contradish.com](https://contradish.com) — leaderboard and live benchmark results
- [CAI leaderboard](https://contradish.com/leaderboard.html) — scores for real deployed products

---

MIT License
