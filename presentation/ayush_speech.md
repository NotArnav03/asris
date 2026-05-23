# Ayush — Presentation Guide

---

## Opening (if you introduce the project)

> "FAIMR tackles a problem that every large recruiter faces — how do you screen hundreds of resumes quickly without missing strong candidates or disadvantaging certain groups? We built a system that combines three types of signals: semantic meaning from sentence embeddings, keyword overlap from TF-IDF, and structured skill matching from a skill graph. No single signal is enough on its own — the power comes from learning how to weight them together.
>
> We ran this on 26,000 resume-JD pairs and the multi-signal model beats plain BERT, BM25, and TF-IDF baselines by a significant margin."

---

## Your Deep Area — Embeddings & Baseline Models

When asked about the models or representations:

> "The embedding backbone is Sentence-BERT — specifically all-MiniLM-L6-v2, which is a strong general-purpose model that runs fast enough for production use. We encode every resume and JD once and cache the vectors to disk, so repeated experiments don't re-encode anything. That caching layer was important for iteration speed — we ran a lot of ablations.
>
> We implemented four baselines: TF-IDF cosine similarity, SBERT cosine similarity, a skill-graph overlap ranker, and a semantic evaluation ranker that scores against section-level embeddings. Each captures something different. TF-IDF is strong when JDs use exact terminology. SBERT handles paraphrasing and synonyms. Skill-graph matching catches structured technical requirements that embedding similarity can miss — things like 'PyTorch' matching 'deep learning frameworks'.
>
> The LTR model Arnav built sits on top of features derived from all four of these. The ablation shows that removing any one signal drops AUC — they're genuinely complementary."

---

## Speaking About the Full Pipeline (sound natural)

- "Ameya's preprocessing gave us clean, section-labelled text. That mattered because we embed the skills section separately from the full resume — section-level embeddings carry more signal than whole-document embeddings for structured JDs."
- "The cached embeddings feed into Arnav's feature engineering directly. We agreed on a vector format early so integration was clean."
- "Aditya's counterfactual explainer also uses the skill embeddings — it finds the smallest skill additions that would flip a candidate's rank."
- "Bakshi runs the full evaluation over our baseline outputs, so you can see in the results table exactly where each baseline fails."

---

## Likely Invigilator Questions

**Q: Why not use a larger BERT model like RoBERTa or BERT-large?**
> "We evaluated on speed-accuracy tradeoff. all-MiniLM-L6-v2 is 22MB, encodes a resume in under 10ms, and achieves competitive semantic similarity scores. BERT-large is 10x the size and 5x slower with marginal gain on this task. For a system that needs to rank hundreds of candidates, latency matters."

**Q: What is a skill graph?**
> "We build a graph where nodes are skills extracted from resumes and JDs, and edges connect related skills — for example, 'TensorFlow' and 'Keras' are linked because they appear together frequently. The skill-based ranker computes overlap between the candidate's skill node set and the JD's required skill set, with graph-walk expansion to catch near-matches."

**Q: Isn't cosine similarity on SBERT embeddings already strong enough? Why add more signals?**
> "It's strong in aggregate but fragile in specific cases. Keyword-heavy JDs — like a job posting that lists 20 specific tools — get underscored by SBERT because dense embeddings smooth over rare tokens. And SBERT has no explicit notion of skill structure. Our ablation confirms: TF-IDF adds 2.1 points AUC over SBERT alone, and skill features add another 1.8 points."

**Q: How does the embedding cache work?**
> "We hash each text input, store the resulting vector as a numpy array keyed by that hash, and load from disk on subsequent calls. If the cache misses, we encode and store. It's a simple read-through cache but it cut our experiment cycle time from hours to minutes."

**Q: Did you try fine-tuning SBERT on your dataset?**
> "We didn't fine-tune in this work — it's listed as a future direction. Fine-tuning would require human-annotated pairs, which we don't have at scale. The off-the-shelf model already gives strong results, and our LTR layer compensates for its weaknesses."

---

## Tone Tips

- The baselines aren't "simple" — they're strong, deliberate reference points. Frame them that way.
- You understand the whole model; Arnav built the LTR on top of features you designed together.
- Numbers to remember: AUC 0.656 (SBERT baseline), 0.604 (BM25), 0.719 (full model).
