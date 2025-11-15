import os
import csv
import json
import re
from datetime import datetime
from openai import OpenAI

# Main model you are evaluating is passed via CLI.
# This client is also used by the judge model.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DATASET_PATH = "dataset.csv"

# Judge model for CAI strain v2.
# You can change this later (for cost vs quality).
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
    Toy strain score: 0 if answers are identical after lowercasing and stripping,
    1 otherwise. This is used as a fallback when the judge model fails.
    """
    a_norm = " ".join(a.lower().split())
    b_norm = " ".join(b.lower().split())
    return 0.0 if a_norm == b_norm else 1.0


def judge_strain_v2(answer_a: str, answer_b: str) -> float:
    """
    CAI strain v2: model-based judging.

    We ask a separate judge model to rate the inconsistency between answer_a and
    answer_b on a 0.0 to 1.0 scale:

      0.0   = identical or trivially rephrased
      0.25  = small differences in wording or emphasis, same core claim
      0.50  = noticeable semantic drift, but not clearly contradictory
      0.75  = tension or opposing implications, close to contradiction
      1.0   = direct contradiction (one says X, the other says not-X)

    The judge returns a JSON object like: {"score": 0.75}.
    If anything goes wrong, we fall back to simple_strain_score.
    """
    # Quick fast path: if answers are byte-for-byte identical
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
        "{ \"score\": 0.75 }\n"
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

        # Try to parse as JSON directly
        try:
            data = json.loads(raw)
            score = float(data.get("score"))
        except Exception:
            # Fallback: extract the first float between 0 and 1 from the text
            match = re.search(r"0?\.\d+|1\.0", raw)
            if not match:
                return simple_strain_score(answer_a, answer_b)
            score = float(match.group())

        # Clamp to [0.0, 1.0] just in case
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0

        return score

    except Exception:
        # If anything in the judge call breaks, fall back to naive scoring
        return simple_strain_score(answer_a, answer_b)


def run_eval(model: str, max_pairs: int = None):
    pairs = []
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(row)

    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    results_filename = f"results_{model.replace(':', '_').replace('.', '_')}.csv"
    scores = []

    print(f"Running model={model} on {len(pairs)} pairs...")
    print(f"Writing detailed outputs to {results_filename}")

    with open(results_filename, "w", newline="", encoding="utf-8") as out_f:
        fieldnames = [
            "pair_id",
            "prompt_A",
            "answer_A",
            "prompt_B",
            "answer_B",
            "strain_score_v2",
        ]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(pairs, start=1):
            pair_id = row["pair_id"]
            prompt_A = row["prompt_A"]
            prompt_B = row["prompt_B"]

            print(f"[{i}/{len(pairs)}] pair_id={pair_id}...", flush=True)

            answer_A = call_model(prompt_A, model)
            answer_B = call_model(prompt_B, model)

            score = judge_strain_v2(answer_A, answer_B)
            scores.append(score)

            writer.writerow(
                {
                    "pair_id": pair_id,
                    "prompt_A": prompt_A,
                    "answer_A": answer_A,
                    "prompt_B": prompt_B,
                    "answer_B": answer_B,
                    "strain_score_v2": f"{score:.4f}",
                }
            )

    avg_strain = sum(scores) / len(scores) if scores else 0.0
    print(f"\nDone. Average CAI strain v2 score for {model}: {avg_strain:.4f}")

    # Append to a summary scores file
    scores_summary = "scores.csv"
    new_file = not os.path.exists(scores_summary)
    with open(scores_summary, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["timestamp", "model", "num_pairs", "avg_strain_v2"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "model": model,
                "num_pairs": len(pairs),
                "avg_strain_v2": f"{avg_strain:.6f}",
            }
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name to evaluate, e.g. gpt-4o",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=300,
        help="Maximum number of pairs to run (default 300).",
    )
    args = parser.parse_args()
    run_eval(args.model, args.max_pairs)
