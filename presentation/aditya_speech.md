# Aditya — Presentation Guide

---

## Opening (if you introduce the project)

> "Most resume screening tools optimize for one thing — similarity to the job description. We went further and asked: is the ranking fair? And can the candidate understand why they were ranked where they were?
>
> FAIMR has two components that no baseline we compared against has: a Fairness-Constrained Re-ranking layer that actively corrects for demographic bias after scoring, and a counterfactual explanation engine that tells each candidate the exact skills they'd need to move up in the ranking. These aren't add-ons — they're integral to what makes this system usable in a real hiring context."

---

## Your Deep Area — Fairness & Explainability

When asked about fairness or explanations:

> "The fairness module works in two stages. First, a bias detector computes the Adverse Impact Ratio — the ratio of selection rates between demographic groups. EEOC guidelines require this to be at least 0.8, the four-fifths rule. If the ranked list violates this, the Fairness-Constrained Re-ranker kicks in.
>
> The re-ranker doesn't just shuffle randomly. It finds the minimum-cost swap sequence — promoting the highest-scoring under-represented candidates while demoting the lowest-marginal-value over-represented ones — until the AIR constraint is satisfied. We tested this under stress: even at severe initial bias levels the re-ranker restores fairness while losing less than 3% NDCG on average.
>
> The explainability side uses greedy submodular optimization. For any candidate ranked below where they'd want to be, we find the smallest set of skills they could add to their resume to cross the score threshold. Average explanation size is 1.1 skills at 0.62 milliseconds — so it's fast enough to run for every candidate, not just top ones."

---

## Speaking About the Full Pipeline (sound natural)

- "The re-ranker sits downstream of Arnav's LTR model — it takes the scored list and applies the fairness constraint as a post-processing step. That separation was intentional: you can swap out the scorer without touching fairness logic."
- "The demographic signals the bias detector uses come from metadata we extract during preprocessing — Ameya's pipeline tags resumes with inferred attributes."
- "The counterfactual explainer uses Ayush's skill embeddings to measure which skill additions would move a candidate's feature vector enough to cross the ranking threshold."
- "Bakshi's frontend has a dedicated explanation tab — you can click any candidate and see their skill-gap breakdown visually."

---

## Likely Invigilator Questions

**Q: Isn't post-processing fairness just hiding the bias rather than fixing it?**
> "That's a fair challenge. Post-processing is pragmatic — you can't always retrain the scorer, and in deployment you need a guarantee you can enforce at inference time. The alternative, in-processing fairness constraints, requires modifying the loss function and retraining, which is expensive and doesn't generalize across score functions. We document the tradeoff in the paper and argue that for hiring systems where legal compliance is the goal, post-processing with a known constraint is actually preferable because it's auditable."

**Q: What is the four-fifths rule?**
> "It's a guideline from the EEOC — the US Equal Employment Opportunity Commission. It says that the selection rate for any demographic group should be at least 80% of the rate for the highest-selected group. If you're selecting 50% of male applicants, you should be selecting at least 40% of female applicants. It's a practical, legally-grounded threshold, which is why we chose it over more theoretical fairness metrics."

**Q: How do you know which demographic group a candidate belongs to?**
> "We infer it from name and language cues in the resume text — it's a proxy, not ground truth. In a real deployment you'd use self-reported data. We're transparent about this limitation in the paper. The module is designed so you can plug in any group labels."

**Q: What is submodular optimization and why use it for explanations?**
> "Submodular functions have diminishing returns — adding the fifth skill gives less marginal gain than adding the first. That property means a greedy algorithm finds a provably near-optimal explanation in linear time. We want the smallest skill set that pushes the candidate over the threshold, and greedy submodular gives us that efficiently without exhaustive search."

**Q: Could the explanations be gamed — candidates just adding buzzwords?**
> "Potentially, yes — and we discuss this. But that's true of any transparent system. The counterargument is that a candidate who adds the flagged skill is now more qualified for the role, which is the intended outcome. The system isn't trying to hide its criteria; it's trying to help candidates meet them."

---

## Tone Tips

- The fairness and explainability work is what makes this a research contribution, not just an engineering project. Speak to it with confidence.
- If asked about something in the LTR model internals, you know them: "the re-ranker operates on the scored output, so we had to understand the score distribution well to set the constraint threshold."
- Key numbers: AIR ≥ 0.8 (four-fifths rule), <3% NDCG loss under re-ranking, 1.1 skills average explanation size, 0.62ms per explanation.
