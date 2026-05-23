# Ameya — Presentation Guide

---

## Opening (if you introduce the project)

> "FAIMR is an end-to-end resume screening system. What made it interesting to build is that the problem isn't just about matching text — you have noisy PDFs, inconsistently formatted resumes, domain mismatch between candidates and jobs, and bias baked into historical data. We had to solve all of that before we even got to the model.
>
> The pipeline goes from raw PDFs to a ranked, explained, fairness-checked shortlist. We tested it on 26,000 resume-JD pairs and the results beat every baseline we compared against."

---

## Your Deep Area — Data Ingestion & Preprocessing

When asked about the data or pipeline:

> "Getting clean input was genuinely the hardest part. Resumes come in as PDFs, DOCX files, free-form text — no standard structure. We used pdfplumber for PDF extraction, which handles multi-column layouts better than most alternatives, and then built a section parser to identify education, experience, and skills blocks.
>
> For job descriptions we had a CSV of postings from Kaggle's job skills dataset. We built a domain classifier to tag both resumes and JDs into categories — tech, finance, healthcare, and so on — and then did domain-balanced sampling when generating training pairs. That was important because without balancing, the model would overfit to tech roles, which dominate the dataset.
>
> Pair generation has three strategies: domain-match pairs, semantic-similarity pairs from SBERT, and skill-overlap pairs. The combination gives training signal across both hard positives and hard negatives."

---

## Speaking About the Full Pipeline (sound natural)

- "Once we had clean pairs, Ayush took them and built the embedding layer — caching SBERT vectors so we weren't re-encoding everything on every experiment run."
- "Arnav designed the feature engineering and the LTR model on top of those embeddings. The domain balancing we did upstream actually showed up clearly in the model's cross-domain generalization."
- "Aditya's fairness module hooks in right after ranking — it uses demographic signals we extract during preprocessing."
- "Bakshi's evaluation suite runs over the same labeled pairs we generated, so the metrics are end-to-end consistent."

---

## Likely Invigilator Questions

**Q: What format is your dataset in and where does it come from?**
> "The job descriptions come from a publicly available Kaggle dataset — about 1.2 million postings, which we filtered and balanced down to around 26,000 pairs. The resumes are from an open resume corpus. Everything gets normalized to plain text with section labels before any model sees it."

**Q: How do you handle resumes with no clear structure?**
> "We use regex heuristics and keyword anchors to find section boundaries — things like 'Education', 'Work Experience', 'Skills' — and fall back to treating the whole document as one block if no sections are detected. In practice, about 80% of resumes have detectable structure."

**Q: Why pdfplumber and not PyMuPDF or pdfminer?**
> "We benchmarked a few. pdfplumber gave the cleanest text extraction on multi-column and table-heavy PDFs, which are common in academic or technical resumes. PyMuPDF was faster but produced more encoding artifacts."

**Q: How many resumes are in the dataset?**
> "We processed around 2,400 unique resumes spanning eight domains. After pair generation the training set has over 26,000 resume-JD pairs."

**Q: Did you do any data augmentation?**
> "Not traditional augmentation. But the three-strategy pair generation — domain, semantic, and skill-based — effectively creates diverse positive and negative pairs, which serves a similar purpose of exposing the model to varied signal."

---

## Tone Tips

- Own the data decisions — they were deliberate engineering choices, not just setup work.
- If asked about a modeling detail, you know it: "the features Arnav built feed directly from the cleaned text we output, so we had to agree on the schema early on."
- Numbers to remember: ~26,000 pairs, ~2,400 resumes, 8 domains.
