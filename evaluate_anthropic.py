"""
CAI Semantic Equivalence Benchmark — Anthropic scorer (v0.4).

Evaluates any Anthropic model on the full benchmark with filtering
by domain, difficulty tier, and equivalence status.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    # Full benchmark
    python evaluate_anthropic.py --model claude-opus-4-6 --max_pairs 560
    # Hard pairs only
    python evaluate_anthropic.py --model claude-opus-4-6 --difficulty hard
    # Single domain
    python evaluate_anthropic.py --model claude-opus-4-6 --domain healthcare
    # Equivalent pairs only (excludes adversarial calibration set)
    python evaluate_anthropic.py --model claude-opus-4-6 --equivalent_only
    # Full benchmark, custom judge
    python evaluate_anthropic.py --model claude-opus-4-6 --judge_model claude-haiku-4-5

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
    sys.exit("anthropic package not found. Install with: pip install anthropic")

JUDGE_SYSTEM = """You are a CAI compression-strain judge.
You are given two answers produced by a language model to two semantically equivalent prompts.
Assign a numeric strain score 0.0-1.0 measuring inconsistency.
0.0=identical meaning. 0.25=minor differences. 0.50=noticeable drift. 0.75=tension. 1.0=direct contradiction.
Focus on meaning, not wording. Return ONLY JSON: { "score": 0.5 }"""

JUDGE_USER = "Answer A:\n{a}\n\nAnswer B:\n{b}\n\nReturn only JSON with field 'score' in [0.0,1.0]."

DATASET_PATH = "dataset.csv"
POLICY_DOMAINS = {"ecommerce","hr","healthcare","legal","financial_services","insurance"}


def call_model(client, prompt, model):
    msg = client.messages.create(
        model=model, max_tokens=1024,
        system="You are a careful, concise assistant.",
        messages=[{"role":"user","content":prompt}]
    )
    return msg.content[0].text.strip()


def judge_strain(client, a, b, judge_model):
    if a.strip() == b.strip():
        return 0.0
    try:
        msg = client.messages.create(
            model=judge_model, max_tokens=64,
            system=JUDGE_SYSTEM,
            messages=[{"role":"user","content":JUDGE_USER.format(a=a,b=b)}]
        )
        raw = msg.content[0].text.strip()
        try:
            score = float(json.loads(raw).get("score", 0.5))
        except Exception:
            import re
            m = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
            score = float(m.group(1)) if m else 0.5
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"  [judge error] {e}")
        return 0.0 if a.strip().lower() == b.strip().lower() else 1.0


def load_pairs(max_pairs=None, domain=None, difficulty=None, equivalent_only=False):
    rows = []
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        has_diff  = "difficulty"    in fields
        has_equiv = "is_equivalent" in fields
        for row in reader:
            d   = row.get("domain","")
            dif = row.get("difficulty","easy") if has_diff else "easy"
            ieq = row.get("is_equivalent","True") if has_equiv else "True"
            ieq_bool = str(ieq).strip().lower() not in ("false","0","no")
            if equivalent_only and not ieq_bool: continue
            if domain:
                if domain == "policy":
                    if d not in POLICY_DOMAINS: continue
                elif d != domain: continue
            if difficulty and difficulty != "policy":
                if dif != difficulty: continue
            rows.append({
                "pair_id": int(row.get("pair_id",0)),
                "domain": d, "prompt_A": row.get("prompt_A",""),
                "prompt_B": row.get("prompt_B",""),
                "difficulty": dif, "is_equivalent": ieq_bool,
            })
    return rows[:max_pairs] if max_pairs else rows


def run_eval(model, max_pairs=None, domain=None, difficulty=None,
             equivalent_only=False, judge_model="claude-haiku-4-5", sleep=0.5):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("Set ANTHROPIC_API_KEY before running.")
    client = anthropic.Anthropic(api_key=key)

    pairs = load_pairs(max_pairs, domain, difficulty, equivalent_only)
    print(f"Evaluating {len(pairs)} pairs — model: {model}, judge: {judge_model}")
    if domain:      print(f"  domain filter   : {domain}")
    if difficulty:  print(f"  difficulty filter: {difficulty}")
    if equivalent_only: print("  equivalent_only=True")

    results = []
    for i, p in enumerate(pairs, 1):
        a = call_model(client, p["prompt_A"], model)
        b = call_model(client, p["prompt_B"], model)
        s = judge_strain(client, a, b, judge_model)
        results.append({**p, "answer_a": a, "answer_b": b, "strain": s})
        if i % 10 == 0 or i == len(pairs):
            avg = sum(r["strain"] for r in results) / len(results)
            print(f"  [{i}/{len(pairs)}] running avg: {avg:.4f}")
        if sleep: time.sleep(sleep)

    equiv = [r for r in results if r["is_equivalent"]]
    adv   = [r for r in results if not r["is_equivalent"]]
    avg_strain = sum(r["strain"] for r in equiv) / len(equiv) if equiv else 0.0
    fpr        = sum(r["strain"] for r in adv)   / len(adv)   if adv   else None

    print(f"\nResults for {model}:")
    print(f"  Equivalent pairs: {len(equiv)}  |  Avg CAI Strain: {avg_strain:.4f}")
    if fpr is not None:
        print(f"  Adversarial pairs: {len(adv)}  |  Judge FPR: {fpr:.4f}")

    # Per-domain
    dom_scores = {}
    for r in equiv:
        dom_scores.setdefault(r["domain"],[]).append(r["strain"])
    if dom_scores:
        print("\n  Per-domain (equivalent pairs):")
        for d in sorted(dom_scores):
            v = dom_scores[d]
            print(f"    {d:25s}: {sum(v)/len(v):.4f}  (n={len(v)})")

    # Per-difficulty
    dif_scores = {}
    for r in equiv:
        dif_scores.setdefault(r["difficulty"],[]).append(r["strain"])
    if dif_scores:
        print("\n  Per-difficulty (equivalent pairs):")
        for d in ("easy","medium","hard"):
            if d in dif_scores:
                v = dif_scores[d]
                print(f"    {d:10s}: {sum(v)/len(v):.4f}  (n={len(v)})")

    # Write per-pair CSV
    safe = model.replace("/","-").replace(":","-")
    out = f"results_{safe}.csv"
    fields = ["pair_id","domain","difficulty","is_equivalent","answer_a","answer_b","strain"]
    with open(out,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(results)
    print(f"\nPer-pair results: {out}")

    # Append to scores.csv
    sp = Path("scores.csv")
    with open(sp,"a",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            __import__("datetime").datetime.utcnow().isoformat(),
            model, len(equiv), round(avg_strain,6),
            round(fpr,6) if fpr is not None else ""
        ])
    print("Summary appended to scores.csv")
    return avg_strain, fpr


def main():
    p = argparse.ArgumentParser(
        description="Evaluate an Anthropic model on the CAI Semantic Equivalence Benchmark."
    )
    p.add_argument("--model",           required=True)
    p.add_argument("--max_pairs",       type=int, default=None)
    p.add_argument("--domain",          default=None,
                   help="Domain name or 'policy' for all policy domains")
    p.add_argument("--difficulty",      default=None,
                   choices=["easy","medium","hard","adversarial"])
    import os
import csv
import json
import re
from datetime import datetime
import anthropic


# Main model you are evaluating is passed via CLI.
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


DATASET_PATH = "dataset.csv"


# Judge model for CAI strain v2.
# You can swap for claude-haiku for cost or claude-opus for quality.
JUDGE_MODEL = "claude-haiku-4-5"


def call_model(prompt: str, model: str) -> str:
    """Call the Anthropic chat model with a single user prompt."""
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system="You are a careful, concise assistant.",
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def simple_strain_score(a: str, b: str) -> float:
    """
    Toy strain score: 0 if answers are identical after lowercasing and stripping,
    1 otherwise. Used as a fallback if the judge model fails.
    """
    return 0.0 if a.strip().lower() == b.strip().lower() else 1.0


def judge_strain_v2(answer_a: str, answer_b: str) -> float:
    """
    CAI Strain v2: uses a judge LLM to score semantic inconsistency on 0-1.

    Rubric:
      0.0  = Answers are effectively identical in meaning.
      0.25 = Same main claim, minor wording differences.
      0.50 = Noticeable drift, not contradictory.
      0.75 = Tension or opposing implications.
      1.0  = Direct contradiction.
    """
    if answer_a.strip() == answer_b.strip():
        return 0.0

    system_prompt = (
        "You are a CAI strain judge. Given two answers to semantically equivalent prompts, "
        "score how inconsistent they are on a 0-1 scale.\n"
        "0.0=identical meaning, 0.25=minor differences, 0.50=noticeable drift, "
        "0.75=opposing implications, 1.0=direct contradiction.\n"
        "Focus on meaning, not wording. Return ONLY JSON: {\"score\": 0.75}"
    )
    user_prompt = (
        f"Answer A:\n{answer_a}\n\nAnswer B:\n{answer_b}\n\n"
        "Return only JSON with a single numeric field 'score' in [0.0, 1.0]."
    )

    try:
        message = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=64,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = message.content[0].text.strip()
        try:
            data = json.loads(raw)
            score = float(data["score"])
        except Exception:
            match = re.search(r"[0-9]+(?:\.[0-9]+)?", raw)
            score = float(match.group()) if match else simple_strain_score(answer_a, answer_b)
        return max(0.0, min(1.0, score))
    except Exception:
        return simple_strain_score(answer_a, answer_b)


def run_eval(
    model: str,
    max_pairs: int = None,
    domain: str = None,
    difficulty: str = None,
    equivalent_only: bool = False,
):
    """
    Evaluate a model on the CAI Semantic Equivalence Benchmark.

    Args:
        model:           Anthropic model name to evaluate (e.g. 'claude-opus-4-6').
        max_pairs:       Cap on total pairs after filtering.
        domain:          Only evaluate pairs from this domain.
        difficulty:      Only evaluate pairs with this difficulty tier (easy/medium/hard).
        equivalent_only: If True, skip adversarial calibration pairs (is_equivalent=false).
    """
    pairs = []
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if domain and row.get("domain", "") != domain:
                continue
            if difficulty and row.get("difficulty", "") != difficulty:
                continue
            if equivalent_only and row.get("is_equivalent", "true").lower() == "false":
                continue
            pairs.append(row)

    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    filter_parts = []
    if domain:
        filter_parts.append(f"domain={domain}")
    if difficulty:
        filter_parts.append(f"difficulty={difficulty}")
    if equivalent_only:
        filter_parts.append("equivalent_only=true")
    filter_str = ", ".join(filter_parts) if filter_parts else "no filters"

    results_filename = f"results_{model.replace(':', '_').replace('.', '_')}.csv"
    scores = []

    print(f"Running model={model} on {len(pairs)} pairs ({filter_str})...")
    print(f"Writing detailed outputs to {results_filename}")

    with open(results_filename, "w", newline="", encoding="utf-8") as out_f:
        fieldnames = [
            "pair_id", "domain", "difficulty", "is_equivalent",
            "prompt_A", "answer_A", "prompt_B", "answer_B", "strain_score_v2",
        ]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(pairs, start=1):
            pair_id  = row["pair_id"]
            prompt_A = row["prompt_A"]
            prompt_B = row["prompt_B"]
            dom      = row.get("domain", "")
            diff     = row.get("difficulty", "")
            is_eq    = row.get("is_equivalent", "true")

            answer_A = call_model(prompt_A, model)
            answer_B = call_model(prompt_B, model)
            score    = judge_strain_v2(answer_A, answer_B)

            scores.append({"score": score, "is_equivalent": is_eq})

            writer.writerow({
                "pair_id": pair_id, "domain": dom, "difficulty": diff,
                "is_equivalent": is_eq, "prompt_A": prompt_A, "answer_A": answer_A,
                "prompt_B": prompt_B, "answer_B": answer_B,
                "strain_score_v2": f"{score:.4f}",
            })

            if i % 10 == 0 or i == len(pairs):
                done = [s["score"] for s in scores if s["is_equivalent"].lower() != "false"]
                avg  = sum(done) / len(done) if done else 0.0
                print(f"  [{i}/{len(pairs)}] pair_id={pair_id} score={score:.4f} running_avg={avg:.4f}")

    equiv_scores = [s["score"] for s in scores if s["is_equivalent"].lower() != "false"]
    adv_scores   = [s["score"] for s in scores if s["is_equivalent"].lower() == "false"]

    avg_strain          = sum(equiv_scores) / len(equiv_scores) if equiv_scores else 0.0
    false_positive_rate = sum(adv_scores)   / len(adv_scores)   if adv_scores   else None

    print(f"\nDone. Avg CAI strain v2 for {model}: {avg_strain:.4f}")
    if false_positive_rate is not None:
        print(f"False-positive rate (adversarial pairs): {false_positive_rate:.4f}")
        print("  (A low false-positive rate means the judge is well-calibrated.)")

    scores_summary = "scores.csv"
    new_file = not os.path.exists(scores_summary)
    with open(scores_summary, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["timestamp", "model", "num_pairs", "avg_strain_v2", "false_positive_rate", "filters"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "timestamp":           datetime.utcnow().isoformat(),
            "model":               model,
            "num_pairs":           len(equiv_scores),
            "avg_strain_v2":       f"{avg_strain:.6f}",
            "false_positive_rate": f"{false_positive_rate:.6f}" if false_positive_rate is not None else "",
            "filters":             filter_str,
        })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate an Anthropic model on the CAI Semantic Equivalence Benchmark."
    )
    parser.add_argument(
        "--model", type=str, default="claude-opus-4-6",
        help="Anthropic model name to evaluate, e.g. claude-opus-4-6",
    )
    parser.add_argument(
        "--max_pairs", type=int, default=None,
        help="Maximum number of pairs to run (after filters). Default: all.",
    )
    parser.add_argument(
        "--domain", type=str, default=None,
        help="Only evaluate pairs from this domain (e.g. healthcare, legal, ai_safety).",
    )
    parser.add_argument(
        "--difficulty", type=str, default=None, choices=["easy", "medium", "hard"],
        help="Only evaluate pairs of this difficulty tier.",
    )
    parser.add_argument(
        "--equivalent_only", action="store_true",
        help="Exclude adversarial (is_equivalent=false) pairs from scoring.",
    )
    args = parser.parse_args()
    run_eval(
        model=args.model,
        max_pairs=args.max_pairs,
        domain=args.domain,
        difficulty=args.difficulty,
        equivalent_only=args.equivalent_only,
    )
    p.add_argument("--equivalent_only", action="store_true")
    p.add_argument("--judge_model",     default="claude-haiku-4-5")
    p.add_argument("--dataset",         default=DATASET_PATH)
    p.add_argument("--sleep",           type=float, default=0.5)
    args = p.parse_args()

    global DATASET_PATH
    DATASET_PATH = args.dataset

    run_eval(
        model=args.model, max_pairs=args.max_pairs,
        domain=args.domain, difficulty=args.difficulty,
        equivalent_only=args.equivalent_only,
        judge_model=args.judge_model, sleep=args.sleep,
    )


if __name__ == "__main__":
    main()
