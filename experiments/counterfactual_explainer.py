"""
FAIMR — Counterfactual Explainer with Metrics
Greedy submodular search that finds the minimal skill subset
to flip a rejected candidate past the threshold.
Includes brute-force verification for small deficiency sets.
Reports: mean |E*|, median, max, latency, % greedy==optimal.
"""

import time
import numpy as np
from itertools import combinations


def greedy_counterfactual(model, base_features, feature_names, jd_skills, resume_skills, threshold=0.5):
    missing = list(jd_skills - resume_skills)
    if not missing:
        return [], 0.0

    start = time.perf_counter()

    scr_idx = feature_names.index("skill_coverage") if "skill_coverage" in feature_names else None
    nms_idx = feature_names.index("num_matched_skills") if "num_matched_skills" in feature_names else None
    kw_idx = feature_names.index("keyword_overlap_ratio") if "keyword_overlap_ratio" in feature_names else None

    n_jd = len(jd_skills)
    n_matched = len(jd_skills & resume_skills)
    selected = []
    remaining = list(missing)

    for _ in range(len(missing)):
        best_skill = None
        best_score = -np.inf

        for skill in remaining:
            trial_matched = n_matched + len(selected) + 1
            trial_features = base_features.copy()

            if scr_idx is not None and n_jd > 0:
                trial_features[scr_idx] = trial_matched / n_jd
            if nms_idx is not None:
                trial_features[nms_idx] = trial_matched

            score = model.predict_proba(trial_features.reshape(1, -1))[0, 1]
            if score > best_score:
                best_score = score
                best_skill = skill

        if best_skill is None:
            break

        selected.append(best_skill)
        remaining.remove(best_skill)

        final_matched = n_matched + len(selected)
        check_features = base_features.copy()
        if scr_idx is not None and n_jd > 0:
            check_features[scr_idx] = final_matched / n_jd
        if nms_idx is not None:
            check_features[nms_idx] = final_matched

        prob = model.predict_proba(check_features.reshape(1, -1))[0, 1]
        if prob >= threshold:
            break

    elapsed_ms = (time.perf_counter() - start) * 1000
    return selected, elapsed_ms


def brute_force_counterfactual(model, base_features, feature_names, jd_skills, resume_skills, threshold=0.5):
    missing = list(jd_skills - resume_skills)
    if not missing:
        return []

    scr_idx = feature_names.index("skill_coverage") if "skill_coverage" in feature_names else None
    nms_idx = feature_names.index("num_matched_skills") if "num_matched_skills" in feature_names else None

    n_jd = len(jd_skills)
    n_matched = len(jd_skills & resume_skills)

    for size in range(1, len(missing) + 1):
        for subset in combinations(missing, size):
            trial_matched = n_matched + len(subset)
            trial_features = base_features.copy()

            if scr_idx is not None and n_jd > 0:
                trial_features[scr_idx] = trial_matched / n_jd
            if nms_idx is not None:
                trial_features[nms_idx] = trial_matched

            prob = model.predict_proba(trial_features.reshape(1, -1))[0, 1]
            if prob >= threshold:
                return list(subset)

    return missing


def evaluate_counterfactuals(model, X, y, y_pred, job_ids, feature_names,
                             pipeline, threshold=0.5, max_samples=200):
    rejected_mask = (y_pred < threshold) & (y == 0)
    rejected_indices = np.where(rejected_mask)[0]

    if len(rejected_indices) == 0:
        return {"n_evaluated": 0}

    if len(rejected_indices) > max_samples:
        np.random.seed(42)
        rejected_indices = np.random.choice(rejected_indices, size=max_samples, replace=False)

    greedy_sizes = []
    latencies = []
    optimal_matches = 0
    total_verified = 0

    pairs = pipeline.pairs

    for idx in rejected_indices:
        row = pairs.iloc[idx]
        job_id = row["job_id"]
        resume_file = row["resume_filename"]

        jd_skills = pipeline.get_jd_skills(job_id)
        resume_skills = pipeline.get_resume_skills(resume_file)
        missing = jd_skills - resume_skills

        if len(missing) == 0:
            continue

        base_features = X[idx].copy()
        greedy_result, latency_ms = greedy_counterfactual(
            model, base_features, feature_names, jd_skills, resume_skills, threshold
        )

        greedy_sizes.append(len(greedy_result))
        latencies.append(latency_ms)

        if len(missing) <= 8:
            brute_result = brute_force_counterfactual(
                model, base_features, feature_names, jd_skills, resume_skills, threshold
            )
            if len(greedy_result) == len(brute_result):
                optimal_matches += 1
            total_verified += 1

    if not greedy_sizes:
        return {"n_evaluated": 0}

    return {
        "n_evaluated": len(greedy_sizes),
        "mean_e_star": round(np.mean(greedy_sizes), 2),
        "median_e_star": round(np.median(greedy_sizes), 2),
        "max_e_star": int(np.max(greedy_sizes)),
        "mean_latency_ms": round(np.mean(latencies), 2),
        "pct_greedy_optimal": round(optimal_matches / total_verified * 100, 1) if total_verified > 0 else None,
        "n_verified_brute": total_verified,
    }
