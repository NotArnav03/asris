# Bakshi — Presentation Guide

---

## Opening (if you introduce the project)

> "FAIMR is a full-stack AI system for resume screening — not just a model, but something you can actually run and interact with. You upload resumes, paste a job description, and get a ranked shortlist with scores, fairness guarantees, and per-candidate explanations in under two seconds.
>
> On the research side, we validated the system rigorously: 26,000 resume-JD pairs, multiple baselines, ablation studies, and a fairness stress test. AUC 0.719 for our model versus 0.604 for BM25 and 0.656 for SBERT. Every number in the paper comes from the evaluation suite we built."

---

## Your Deep Area — Evaluation, Experiments & API/Frontend

When asked about results, the demo, or the system architecture:

> "The evaluation framework computes five metrics: Precision@K, NDCG, MRR, MAP, and ROC-AUC. We report at K=5 and K=10 because those are the practically relevant cutoffs — a recruiter looking at a top-5 shortlist cares about precision, but ranking quality across the full list matters too, which is what NDCG captures.
>
> We ran three experiments beyond the main result: an ablation study removing each signal one at a time to quantify contribution, a label quality validation to check that our programmatically generated pairs are consistent, and a fairness stress test that artificially worsens the bias level and measures how well the re-ranker recovers.
>
> The API is FastAPI with eight endpoints — rank, explain, upload PDFs, stats, and health. The frontend is a single-page app: a ranking tab where you paste a JD and upload resumes, an explanation tab where you click any candidate and see their skill-gap breakdown, and a dashboard with dataset statistics. The Top K slider controls how many candidates are returned — default 10, max 20."

---

## Speaking About the Full Pipeline (sound natural)

- "The metrics run over the same labeled pairs Ameya's pipeline produces, so evaluation is end-to-end — no separate held-out dataset that could introduce distribution shift."
- "The API rank endpoint calls Arnav's LTR model, then Aditya's re-ranker, and returns the final list with scores. It's a clean sequential call — each component is independently testable."
- "The explain endpoint calls Aditya's counterfactual module, which uses Ayush's skill embeddings under the hood. From the API's perspective it's just one call."
- "MLflow tracks every experiment run — hyperparameters, metrics, artifacts. That's how we generated the results tables in the paper without losing track of which config produced which number."

---

## Likely Invigilator Questions

**Q: Why these five metrics and not just accuracy?**
> "Accuracy is meaningless for ranking — there's no binary correct answer for 'which resume is best'. NDCG accounts for the position of relevant items in the ranked list, penalising you more for putting a strong candidate at rank 10 than rank 2. MRR tells you where the first good match appears. MAP averages precision across all recall levels. Together they give a complete picture of ranking quality at different points in the list."

**Q: How did you avoid overfitting in the experiments?**
> "We use a train/validation/test split — the LTR model never sees the test pairs during training. The ablation and stress tests run on the same held-out test set. We also ran cross-domain evaluation, training on some domains and testing on others, to check that the model generalises."

**Q: Can you demo the system right now?**
> "Yes — the server runs locally on port 8000. You hit the frontend at localhost:8000/frontend, paste a JD, upload some resumes, set Top K, and rank. The first request takes a few seconds because SBERT loads lazily, but after that it's fast. The explain tab shows per-candidate skill gaps."

**Q: What does the dashboard show?**
> "Dataset statistics — resume count by domain, JD distribution, average skill overlap, and the fairness metrics for the current ranked result. It gives a recruiter a quick sanity check before acting on the shortlist."

**Q: How does MLflow help here?**
> "Every time we ran an experiment — different hyperparameters, different feature subsets, different pair generation strategies — MLflow logged the config and the resulting metrics automatically. That meant we could reproduce any result in the paper by loading the corresponding run, and we could compare 50 runs on a single chart to pick the best configuration."

**Q: Why FastAPI over Flask or Django?**
> "FastAPI gives us automatic request validation via Pydantic, async support out of the box, and auto-generated Swagger docs at /docs with no extra work. For a system where we're demoing to people, having a live interactive API docs page is genuinely useful. Flask would have needed a lot more boilerplate for the same result."

---

## Tone Tips

- You've run the system end to end more than anyone. If there's a live demo, you should drive it.
- When citing metrics, be confident: "our NDCG@10 is X" — you generated those numbers.
- If asked about model internals, connect back: "the evaluation doesn't care how the model works internally — it just sees the ranked list, which is what matters for comparing fairly against baselines."
- Numbers to remember: AUC 0.719, NDCG@10, P@5, P@10, 8 API endpoints, port 8000.
