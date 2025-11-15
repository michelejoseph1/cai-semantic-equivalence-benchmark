# CAI Semantic Equivalence Benchmark v0.1

This benchmark measures model inconsistency under semantically equivalent perturbations. Each pair contains two prompts that have the same meaning but different surface form. A model that is internally coherent should produce stable outputs across each pair. This dataset is part of Compression Aware Intelligence (CAI), which defines compression strain as the measurable signal of internal contradiction under semantic equivalence. High strain indicates instability in the model's internal representation.

The dataset is provided in two formats:`dataset.csv` & `dataset.json`

Each entry has `pair_id`, `prompt_A`, `prompt_B`

Example:
pair_id,prompt_A,prompt_B
1,"Who wrote Pride and Prejudice?","Which author is responsible for the novel Pride and Prejudice?"


## CAI Strain Results

We evaluate the benchmark using a semantic judge model (“CAI-strain-v2”).  
For gpt-4o on all 300 pairs:

**Average CAI Strain Score:** 0.05  

Lower is better (0 = perfect semantic consistency, 1 = contradiction).


## Recommended Evaluation Procedure

For each pair:

1. Query the model with prompt A.
2. Query the model with prompt B.
3. Measure:
   - output difference (edit distance or embedding distance)
   - contradiction or factual disagreement
   - refusal inconsistency
   - reasoning drift

4. Average these measures to compute a compression strain score.

Lower scores indicate higher internal coherence.

## Intended Use

This benchmark helps evaluate:

- truthfulness under paraphrase
- robustness to meaning preserving perturbations
- hidden internal contradictions
- stability across rephrasings
- hallucination susceptibility under compression stress

It is useful for:
- reliability research
- safety evaluations
- interpretability studies
- model comparison

## License

This dataset is free to use with attribution: Michele Joseph, Compression Aware Intelligence (2025).

## Contact

For collaboration or questions:
- michele.a.joseph@gmail.com
