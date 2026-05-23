# Arnav — Presentation Guide

---

## Opening (if you go first or introduce the project)

> "FAIMR stands for Fairness-Aware Interpretable Multi-Signal Ranking. The core problem we wanted to solve is that traditional keyword-based screening misses strong candidates and carries hidden bias. So we built a full pipeline — from raw PDFs all the way to a ranked shortlist with explanations — that is fast, fair, and interpretable.
>
> The system has three novel contributions: a multi-signal Learning-to-Rank model that fuses semantic, lexical, and skill-graph features; a Fairness-Constrained Re-ranking layer that enforces the four-fifths rule; and counterfactual skill-gap explanations so candidates know exactly what to improve.
>
> We validated it on over 26,000 resume-JD pairs. Our LTR model hits AUC 0.719 versus 0.604 for BM25 and 0.656 for plain SBERT, so the gains are real and consistent."

---

## Your Deep Area — LTR Core & Architecture

When asked about the model or how everything fits together:

> "The ranking model is a LambdaMART-style Learning-to-Rank trained on pairwise comparisons. For each resume-JD pair we build a feature vector — SBERT cosine similarity, TF-IDF cosine, skill overlap ratio, section-weighted scores — and feed that into XGBoost. The key insight was that no single signal dominates across domains: SBERT is better for semantic roles, TF-IDF catches exact keyword matches, and skill overlap handles technical JDs. The ensemble learns those weights from data rather than us hard-coding them.
>
> We experimented with LightGBM too — slightly faster training, comparable AUC — and kept both for the ablation study."

---

## Speaking About the Full Pipeline (sound natural)

- "We built the data layer first — Ameya drove a lot of that — parsing PDFs with pdfplumber, cleaning and section-splitting each resume so we had structured fields going in."
- "Ayush set up the embedding cache so SBERT didn't re-encode the same resume twice. That made experimentation much faster for everyone."
- "Aditya handled the fairness and explainability side, which honestly was the part that made the paper interesting — the counterfactual explanations are something none of the baselines have."
- "Bakshi wired everything into the FastAPI server and the frontend so we could actually demo it live. The evaluation suite he built is also how we generated all our result tables."

---

## Likely Invigilator Questions

**Q: Why LTR over a simple cosine similarity threshold?**
> "Cosine similarity with a single embedding gives you one signal. The problem is it fails on keyword-heavy JDs and doesn't account for skill matches at all. LTR lets us combine multiple signals and learn their relative importance from labeled data. Our ablation confirms each signal contributes — removing any one drops AUC measurably."

**Q: How did you label your training data?**
> "We used a combination of domain-matching heuristics, semantic similarity thresholds, and skill-overlap scores to generate soft labels for pairwise comparisons. We also ran a label quality validation experiment to check consistency. It's not human-annotated ground truth, but the downstream AUC numbers suggest the labels are informative."

**Q: What's the latency of the full pipeline?**
> "Ranking 100 resumes against one JD takes under two seconds after the initial embedding load. The counterfactual explanation for one candidate averages 0.62 milliseconds, which makes it practical to explain every result, not just the top one."

**Q: What would you improve with more time?**
> "Two things: first, true human annotation on a subset to give us a harder eval benchmark. Second, fine-tuning the SBERT model on domain-specific resume-JD pairs rather than using the off-the-shelf all-MiniLM checkpoint."

---

## Tone Tips

- Speak slowly when citing numbers — invigilators write them down.
- When referencing teammates, say "we found" not "they did" — you all built this together.
- If you don't know an exact detail, say "the specifics are in the paper, but the key takeaway is…" and redirect to the insight.
