# CAI Semantic Equivalence Benchmark v0.1  
*A focused probe for consistency under meaning-preserving paraphrase*

This benchmark measures how reliably a model preserves its internal beliefs when two prompts ask **the same question in different words**.  
It is built as the first operational slice of **Compression-Aware Intelligence (CAI)**: the view that reliability failures emerge when a model’s compressed representation cannot satisfy all constraints in a semantic equivalence class.

A coherent model should answer both prompts the same way.  
When it does not, CAI interprets the disagreement as **compression strain**.

---

## 📁 Dataset

The dataset consists of **300 semantically equivalent prompt pairs** across:

- factual queries  
- everyday reasoning  
- math and logic  
- counterfactuals  
- ethics / social norms  
- creative writing  
- summarization / paraphrasing  
- meta prompts about models  
- CAI-specific stressors (ambiguity, abstraction shifts, underspecification)

Each row contains:
pair_id, prompt_A, prompt_B

**Example:**
1,"Who wrote Pride and Prejudice?","Which author is responsible for the novel Pride and Prejudice?"

Formats:

- `dataset.csv` — canonical version  
- `dataset.json` — JSON list of all pairs  

---

## 📊 CAI Strain Results (gpt-4o)

We evaluate models using a simple semantic-judge scoring function (“**CAI-strain-v2**”) that compares:

- factual agreement  
- reasoning consistency  
- directional claims  
- tone/intent shifts  
- contradictory or mutually exclusive statements  

Scored 0 (stable) to 1 (contradictory).

For **gpt-4o** on all **300** pairs:

**→ Average CAI Strain v2: 0.3642**  
**→ String mismatch baseline: 0.9900**

Surface mismatch is almost universal (0.99), but meaningful drift appears in ~36% of semantically equivalent pairs.  
This gap is the core motivation for the CAI view: **reliability is about representation-level coherence, not text-level similarity**.

All outputs are available in:

- `results_gpt-4o.csv`  
- `scores.csv`  

---

## ▶️ Reproducing the Results

```bash
git clone https://github.com/michelejoseph1/cai-semantic-equivalence-benchmark.git
cd cai-semantic-equivalence-benchmark

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

export OPENAI_API_KEY="YOUR_KEY_HERE"

python evaluate_openai.py --model gpt-4o --max_pairs 300

