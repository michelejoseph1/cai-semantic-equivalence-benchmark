import os
import csv
import json
import re
import argparse
from datetime import datetime
from openai import OpenAI


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DATASET_PATH = "dataset.csv"

# Judge model for CAI Strain v2.
JUDGE_MODEL = "gpt-4o-mini"


def call_model(prompt: str, model: str) -> str:
    """Call the chat model with a single user prompt."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful, concise assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def simple_strain_score(a: str, b: str) -> float:
    """
    Fallback strain score: 0 if answers are identical after lowercasing and
    stripping, 1 otherwise. Used when the judge model call fails.
    """
    return 0.0 if a.strip().lower() == b.strip().lower() else 1.0


def judge_strain_v2(answer_a: str, answer_b: str) -> float:
    """
    CAI Strain v2: use a judge LLM to score semantic inconsistency 0-1.
    The judge does NOT see the prompts, only the answers.
    """
    # Quick fast path: byte-identical answers
    if answer_a.strip() == answer_b.strip():
        return 0.0

    system_prompt = (
        "You are a CAI compression-strain judge. "
        "You are given two answers produced by a language model to two prompts "
        "that are semantically equivalent.\n\n"
        "Your task is to assign a numeric strain score between 0.0 and 1.0 "
        "based on how inconsistent the two answers are.\n\n"
        "Use this rubric:\n"
        "0.0  = Answers are effectively identical in meaning.\n"
        "0.25 = Same main claim with only minor differences.\n"
        "0.50 = Noticeable differences in content or emphasis, but not contradictory.\n"
        "0.75 = Strong tension or opposing implications.\n"
        "1.0  = Direct contradiction.\n\n"
        "Instructions:\n"
        "- Focus on meaning, not wording.\n"
        "- Ignore harmless elaboration.\n"
        "- Always return ONLY JSON in this form:\n"
        '{ \"score\": 0.75 }\n'
    )

    user_prompt = (
        "Here are the two answers:\n\n"
        f"Answer A:\n{answer_a}\n\n"
        f"Answer B:\n{answer_b}\n\n"
        "Return only JSON with a single numeric field 'score' in the range [0.0, 1.0]."
    )

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()

        try:
            data = json.loads(raw)
            score = float(data.get("score"))
        except Exception:
            match = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
            if not match:
                return simple_strain_score(answer_a, answer_b)
            score = float(match.group(1))

        # Clamp to [0.0, 1.0]
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0

        return score

    except Exception:
        return simple_strain_score(answer_a, answer_b)


def load_pairs(
    max_pairs: int = None,
    domain: str = None,
    difficulty: str = None,
    equivalent_only: bool = False,
) -> list:
    """
    Load pairs from dataset.csv with optional filters.

    Args:
        max_pairs: cap on total pairs returned (applied after other filters).
        domain: if set, only return pairs from this domain.
        difficulty: if set, only return pairs with this difficulty tier.
                    Accepts: easy, medium, hard, adversarial, or 'policy'
                    (which returns all policy-domain pairs regardless of difficulty).
        equivalent_only: if True, exclude adversarial calibration pairs
                         (is_equivalent == False).
    """
    rows = []
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Handle both old schema (no difficulty/is_equivalent) and new schema
        fieldnames = reader.fieldnames or []
        has_difficulty   = "difficulty"    in fieldnames
        has_is_equiv     = "is_equivalent" in fieldnames

        POLICY_DOMAINS = {
            "ecommerce", "hr", "healthcare", "legal",
            "financial_services", "insurance",
        }

        for row in reader:
            row_domain     = row.get("domain", "")
            row_difficulty = row.get("difficulty", "easy") if has_difficulty else "easy"
            row_is_equiv   = row.get("is_equivalent", "True") if has_is_equiv else "True"
            is_equiv_bool  = str(row_is_equiv).strip().lower() not in ("false", "0", "no")

            # Filter: equivalent_only
            if equivalent_only and not is_equiv_bool:
                continue

            # Filter: domain
            if domain:
                if domain == "policy":
                    if row_domain not in POLICY_DOMAINS:
                        continue
                elif row_domain != domain:
                    continue

            # Filter: difficulty
            if difficulty and difficulty != "policy":
                if row_difficulty != difficulty:
                    continue

            rows.append({
                "pair_id":       int(row.get("pair_id", 0)),
                "domain":        row_domain,
                "prompt_A":      row.get("prompt_A", ""),
                "prompt_B":      row.get("prompt_B", ""),
                "difficulty":    row_difficulty,
                "is_equivalent": is_equiv_bool,
            })

    if max_pairs:
        rows = rows[:max_pairs]
    return rows


def run_eval(
    model: str,
    max_pairs: int = None,
    domain: str = None,
    difficulty: str = None,
    equivalent_only: bool = False,
):
    pairs = load_pairs(
        max_pairs=max_pairs,
        domain=domain,
        difficulty=difficulty,
        equivalent_only=equivalent_only,
    )
    print(f"Evaluating {len(pairs)} pairs with model: {model}")
    if domain:
        print(f"  domain filter: {domain}")
    if difficulty:
        print(f"  difficulty filter: {difficulty}")
    if equivalent_only:
        print(f"  equivalent_only=True (adversarial pairs excluded)")

    results = []
    for i, pair in enumerate(pairs, 1):
        pid         = pair["pair_id"]
        prompt_a    = pair["prompt_A"]
        prompt_b    = pair["prompt_B"]
        pair_domain = pair["domain"]
        pair_diff   = pair["difficulty"]
        is_equiv    = pair["is_equivalent"]

        answer_a = call_model(prompt_a, model)
        answer_b = call_model(prompt_b, model)
        strain   = judge_strain_v2(answer_a, answer_b)

        results.append({
            "pair_id":       pid,
            "domain":        pair_domain,
            "difficulty":    pair_diff,
            "is_equivalent": is_equiv,
            "answer_a":      answer_a,
            "answer_b":      answer_b,
            "strain":        strain,
        })

        if i % 10 == 0 or i == len(pairs):
            avg = sum(r["strain"] for r in results) / len(results)
            print(f"  [{i}/{len(pairs)}] running avg strain: {avg:.4f}")

    # Summary stats
    equiv_results = [r for r in results if r["is_equivalent"]]
    adv_results   = [r for r in results if not r["is_equivalent"]]

    avg_strain = sum(r["strain"] for r in equiv_results) / len(equiv_results) if equiv_results else 0.0
    fpr        = sum(r["strain"] for r in adv_results)   / len(adv_results)   if adv_results  else None

    print(f"\nResults for {model}:")
    print(f"  Equivalent pairs evaluated : {len(equiv_results)}")
    print(f"  Avg CAI Strain             : {avg_strain:.4f}")
    if fpr is not None:
        print(f"  Adversarial pairs evaluated: {len(adv_results)}")
        print(f"  Judge false-positive rate  : {fpr:.4f}")

    # Per-domain breakdown
    domain_scores = {}
    for r in equiv_results:
        d = r["domain"]
        domain_scores.setdefault(d, []).append(r["strain"])
    if domain_scores:
        print("\n  Per-domain avg strain (equivalent pairs):")
        for d in sorted(domain_scores):
            vals = domain_scores[d]
            print(f"    {d:25s}: {sum(vals)/len(vals):.4f}  (n={len(vals)})")

    # Per-difficulty breakdown
    diff_scores = {}
    for r in equiv_results:
        d = r["difficulty"]
        diff_scores.setdefault(d, []).append(r["strain"])
    if diff_scores:
        print("\n  Per-difficulty avg strain (equivalent pairs):")
        for d in ("easy", "medium", "hard"):
            if d in diff_scores:
                vals = diff_scores[d]
                print(f"    {d:10s}: {sum(vals)/len(vals):.4f}  (n={len(vals)})")

    # Write per-pair results
    safe_model = model.replace("/", "-").replace(":", "-")
    results_path = f"results_{safe_model}.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["pair_id", "domain", "difficulty", "is_equivalent",
                      "answer_a", "answer_b", "strain"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPer-pair results written to {results_path}")

    # Append summary to scores.csv
    with open("scores.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            model,
            len(equiv_results),
            round(avg_strain, 6),
            round(fpr, 6) if fpr is not None else "",
        ])
    print("Summary appended to scores.csv")

    return avg_strain, fpr


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the CAI Semantic Equivalence Benchmark."
    )
    parser.add_argument("--model",           required=True,
                        help="OpenAI model name, e.g. gpt-4o")
    parser.add_argument("--max_pairs",       type=int, default=None,
                        help="Cap on total pairs to evaluate")
    parser.add_argument("--domain",          default=None,
                        help="Filter to a single domain, e.g. healthcare or 'policy'")
    parser.add_argument("--difficulty",      default=None,
                        choices=["easy", "medium", "hard", "adversarial"],
                        help="Filter to a difficulty tier")
    parser.add_argument("--equivalent_only", action="store_true",
                        help="Exclude adversarial calibration pairs (is_equivalent=False)")
    args = parser.parse_args()

    run_eval(
        model=args.model,
        max_pairs=args.max_pairs,
        domain=args.domain,
        difficulty=args.difficulty,
        equivalent_only=args.equivalent_only,
    )


if __name__ == "__main__":
    main()
