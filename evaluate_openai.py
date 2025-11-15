import os
import csv
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DATASET_PATH = "dataset.csv"


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
    1 otherwise. This is a baseline, not a full CAI judge.
    """
    a_norm = " ".join(a.lower().split())
    b_norm = " ".join(b.lower().split())
    return 0.0 if a_norm == b_norm else 1.0


def run_eval(model: str, max_pairs: int | None = None):
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
            "strain_score",
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

            score = simple_strain_score(answer_A, answer_B)
            scores.append(score)

            writer.writerow(
                {
                    "pair_id": pair_id,
                    "prompt_A": prompt_A,
                    "answer_A": answer_A,
                    "prompt_B": prompt_B,
                    "answer_B": answer_B,
                    "strain_score": score,
                }
            )

    avg_strain = sum(scores) / len(scores) if scores else 0.0
    print(f"\nDone. Average strain score for {model}: {avg_strain:.4f}")

    # Append to a summary scores file
    scores_summary = "scores.csv"
    new_file = not os.path.exists(scores_summary)
    with open(scores_summary, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["timestamp", "model", "num_pairs", "avg_strain"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "model": model,
                "num_pairs": len(pairs),
                "avg_strain": f"{avg_strain:.6f}",
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
