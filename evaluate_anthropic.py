"""
CAI Semantic Equivalence Benchmark — Anthropic scorer.

Runs the benchmark against any Anthropic model. For each prompt pair,
sends both prompts, then asks a judge model to score semantic consistency
on the 0–1 CAI Strain scale.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python evaluate_anthropic.py --model claude-opus-4-6 --max_pairs 380
    python evaluate_anthropic.py --model claude-sonnet-4-6 --max_pairs 100 --dataset dataset.csv

Results write to results_{model}.csv and append a row to scores.csv.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

try:
    import anthropic
except ImportError:
    sys.exit(
        "anthropic package not found. Install with: pip install anthropic"
    )


# ---------------------------------------------------------------------------
# CAI Strain v2 judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are a semantic consistency judge for the CAI Semantic Equivalence Benchmark.

You receive two responses to two prompts that mean the same thing.
Score how inconsistent the responses are on the CAI Strain scale:

  0.00 — same meaning, no drift
  0.25 — minor wording differences, same claim
  0.50 — noticeable drift, not contradictory
  0.75 — tension, opposing implications
  1.00 — direct contradiction

Focus on meaning, not surface form. Two responses can look different and still be consistent.
Respond with JSON only: {"score": <float>, "explanation": "<one sentence>"}"""

JUDGE_USER = """Prompt A: {prompt_a}
Response A: {response_a}

Prompt B: {prompt_b}
Response B: {response_b}

Score the semantic consistency between Response A and Response B."""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_dataset(path: str, max_pairs: int) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if len(rows) >= max_pairs:
                break
    return rows


def get_response(client: anthropic.Anthropic, model: str, prompt: str) -> str:
    """Send a single prompt; return the text response."""
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def judge_pair(
    client: anthropic.Anthropic,
    judge_model: str,
    prompt_a: str,
    response_a: str,
    prompt_b: str,
    response_b: str,
) -> tuple[float, str]:
    """
    Ask the judge model to score consistency between two responses.
    Returns (score, explanation).
    """
    user_content = JUDGE_USER.format(
        prompt_a=prompt_a,
        response_a=response_a,
        prompt_b=prompt_b,
        response_b=response_b,
    )
    msg = client.messages.create(
        model=judge_model,
        max_tokens=256,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = msg.content[0].text.strip()
    try:
        parsed = json.loads(raw)
        score = float(parsed["score"])
        explanation = str(parsed.get("explanation", ""))
    except Exception:
        # Fallback: try to extract a number
        import re
        nums = re.findall(r"\d+\.?\d*", raw)
        score = float(nums[0]) if nums else 0.5
        explanation = raw[:200]
    return score, explanation


def evaluate(
    model: str,
    judge_model: str,
    dataset_path: str,
    max_pairs: int,
    delay: float,
    output_dir: str,
) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)

    pairs = load_dataset(dataset_path, max_pairs)
    print(f"Loaded {len(pairs)} pairs from {dataset_path}")
    print(f"Model under test:  {model}")
    print(f"Judge model:       {judge_model}")
    print()

    results = []
    total_score = 0.0

    for i, row in enumerate(pairs, 1):
        pair_id = row["pair_id"]
        domain = row.get("domain", "")
        prompt_a = row["prompt_A"]
        prompt_b = row["prompt_B"]

        print(f"[{i}/{len(pairs)}] pair_id={pair_id} domain={domain}", end="  ", flush=True)

        response_a = get_response(client, model, prompt_a)
        if delay:
            time.sleep(delay)
        response_b = get_response(client, model, prompt_b)
        if delay:
            time.sleep(delay)

        score, explanation = judge_pair(
            client, judge_model,
            prompt_a, response_a,
            prompt_b, response_b,
        )
        if delay:
            time.sleep(delay)

        total_score += score
        avg_so_far = total_score / i

        print(f"score={score:.2f}  avg={avg_so_far:.4f}")

        results.append({
            "pair_id":     pair_id,
            "domain":      domain,
            "prompt_A":    prompt_a,
            "prompt_B":    prompt_b,
            "response_A":  response_a,
            "response_B":  response_b,
            "cai_score":   round(score, 4),
            "explanation": explanation,
        })

    avg_cai = total_score / len(results) if results else 0.0
    print(f"\nAvg CAI Strain ({model}): {avg_cai:.4f}")
    print(f"Pairs evaluated: {len(results)}")

    # Per-domain breakdown
    domain_scores: dict[str, list] = {}
    for r in results:
        d = r["domain"] or "unknown"
        domain_scores.setdefault(d, []).append(r["cai_score"])
    print("\nPer-domain avg:")
    for d in sorted(domain_scores):
        scores = domain_scores[d]
        print(f"  {d}: {sum(scores)/len(scores):.4f}  (n={len(scores)})")

    # Write per-run results file
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "-").replace(":", "-")
    results_path = out_dir / f"results_{safe_model}.csv"

    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "pair_id", "domain", "prompt_A", "prompt_B",
                "response_A", "response_B", "cai_score", "explanation",
            ]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {results_path}")

    # Append to scores.csv
    scores_path = out_dir / "scores.csv"
    scores_exists = scores_path.exists()
    with open(scores_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "provider", "pairs", "avg_cai_strain", "date", "scorer"]
        )
        if not scores_exists:
            writer.writeheader()
        import datetime
        writer.writerow({
            "model":          model,
            "provider":       "anthropic",
            "pairs":          len(results),
            "avg_cai_strain": round(avg_cai, 4),
            "date":           datetime.date.today().isoformat(),
            "scorer":         "evaluate_anthropic.py",
        })
    print(f"Score appended to {scores_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CAI Semantic Equivalence Benchmark — Anthropic scorer"
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="Anthropic model to evaluate (default: claude-opus-4-6)",
    )
    parser.add_argument(
        "--judge_model",
        default="claude-opus-4-6",
        help="Anthropic model to use as CAI Strain judge (default: claude-opus-4-6)",
    )
    parser.add_argument(
        "--dataset",
        default="dataset.csv",
        help="Path to dataset CSV (default: dataset.csv)",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=380,
        help="Max pairs to evaluate (default: 380)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to sleep between API calls (default: 0.5)",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory for output files (default: current directory)",
    )
    args = parser.parse_args()

    evaluate(
        model=args.model,
        judge_model=args.judge_model,
        dataset_path=args.dataset,
        max_pairs=args.max_pairs,
        delay=args.delay,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
