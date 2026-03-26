# Contributing

Two ways to contribute: add model results, or add prompt pairs.

---

## Add model results

Run the scorer against your model, then open a PR adding a row to `leaderboard.json`.

**OpenAI models:**
```bash
git clone https://github.com/michelejoseph1/cai-semantic-equivalence-benchmark.git
cd cai-semantic-equivalence-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
python evaluate_openai.py --model gpt-4o --max_pairs 380
```

**Anthropic models:**
```bash
export ANTHROPIC_API_KEY="your-key"
python evaluate_anthropic.py --model claude-opus-4-6 --max_pairs 380
```

Results write to `results_{model}.csv` and append to `scores.csv`.

Add your result to `leaderboard.json`:
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

Open a PR with the title: `results: {model} avg_cai_strain={score}`.

**Requirements:**
- Run on the full 380-pair dataset, or note the subset used.
- Use an unmodified scorer. If you modify the judge prompt, note it.
- One result per model per dataset version.

---

## Add prompt pairs

New pairs extend the benchmark. Policy pairs (ecommerce, HR, healthcare, legal, finance, insurance) are especially useful — that's where CAI failures matter most in production systems.

**Format:** each pair is one row in `dataset.csv`:

```
pair_id,domain,prompt_A,prompt_B
381,finance,"What is the penalty for early withdrawal from a CD?","If I take money out of my CD before it matures will I be charged?"
```

**Pair construction guidelines:**

- Both prompts must mean the same thing. A fully informed person would expect the same answer to both.
- Vary surface form, not intent: phrasing, formality, syntactic order, vocabulary, presupposition, emotional register.
- Don't vary scope or specificity in ways that legitimately change the answer.
- Policy pairs: vary formal/policy language vs. colloquial. Include at least one presupposition variation per domain batch.
- Assign a `domain` tag. Reuse existing domains or propose a new one with a brief description.

**PR format:**
- Title: `pairs: add N {domain} pairs`
- Body: what the pairs test and why they belong in the benchmark.
- Include updated `dataset.csv` and `dataset.json`. Run `python build_dataset.py` if starting from `dataset_original.csv`.

---

## Questions

Open an issue or reach out at contradish.com.
