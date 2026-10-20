from __future__ import annotations

import numpy as np


def aggregate(per_face: list[dict], threshold: float = 0.5):
    if not per_face:
        return 0.5, 0.0, {}
    probs = [r["fake_prob"] for r in per_face]
    mean_p = float(np.mean(probs))
    std_p = float(np.std(probs)) if len(probs) > 1 else 0.0
    confidence = 1.0 - min(std_p * 2.0, 0.9)
    stats = {
        "mean": round(mean_p, 4),
        "min": round(float(min(probs)), 4),
        "max": round(float(max(probs)), 4),
        "std": round(std_p, 4),
        "median": round(float(np.median(probs)), 4),
        "count": len(probs),
        "above_threshold": sum(1 for p in probs if p > threshold),
        "below_threshold": sum(1 for p in probs if p < 1 - threshold),
    }
    return mean_p, confidence, stats


def aggregate_regions(per_face: list[dict]) -> dict | None:
    if not per_face:
        return None
    region_keys = set()
    for r in per_face:
        if r.get("region_energy"):
            region_keys.update(r["region_energy"].keys())
    if not region_keys:
        return None
    agg = {k: {"attention_pct": [], "face_count": 0} for k in region_keys}
    for r in per_face:
        re = r.get("region_energy")
        if not re:
            continue
        for k in region_keys:
            if k in re:
                agg[k]["attention_pct"].append(re[k])
                agg[k]["face_count"] += 1
    result = {}
    for k, v in agg.items():
        if not v["attention_pct"]:
            continue
        result[k] = {
            "attention_pct": round(float(np.mean(v["attention_pct"])), 1),
            "face_count": v["face_count"],
        }
    return result


def build_reasoning(detector_name, mean_p, stats, timing, device_name,
                    per_face=None, weight_status="official", cam_available=True,
                    threshold: float = 0.5, special_signals: dict | None = None,
                    unit: str = "face"):
    """Human-readable reasoning text. ``special_signals`` may carry detector-
    specific claims (e.g. RECCE reconstruction). ``unit`` names what was
    scored: face, frame, clip or window (audio)."""
    special_signals = special_signals or {}
    units = unit + "s"
    region_agg = aggregate_regions(per_face) if per_face else None
    top_region = None
    if region_agg:
        top_region = max(region_agg.items(), key=lambda kv: kv[1]["attention_pct"])

    if not stats:
        if unit == "face":
            summary = (f"{detector_name} UNCERTAIN: No faces were detected in the media, "
                       f"so no per-face classification could be performed.")
            factors = ["No faces detected — nothing to classify", "No input media features analyzed"]
        else:
            summary = (f"{detector_name} UNCERTAIN: No {units} could be extracted from the media, "
                       f"so nothing was classified.")
            factors = [f"No {units} available — nothing to classify"]
        if weight_status == "missing":
            factors.insert(0, "Weight file not present — this run used random initialization.")
        return summary, factors

    factors = []
    if weight_status == "missing":
        factors.append(
            f"Weight file not present — this run used random initialization "
            f"and the scores are NOT meaningful. Place the checkpoint (see "
            f"weights/README.md) and re-run."
        )
    elif weight_status == "backbone":
        factors.append(
            "Only a pretrained backbone is available (no detector-trained "
            "checkpoint is released); the classification head is untrained, so "
            "scores should be treated as unreliable."
        )
    elif weight_status == "random":
        factors.append(
            "No pretrained weights exist for this detector — model was randomly "
            "initialized and scores are not meaningful."
        )

    if cam_available and unit != "window":
        factors.append(
            "Gradient-weighted activation maps localize the regions the model "
            "relies on for the verdict."
        )
    elif unit != "window":
        factors.append("No activation map could be produced for this model.")

    if mean_p > threshold:
        summary = (
            f"{detector_name} FAKE: Media detected as manipulated. "
            f"Average fake probability {mean_p:.4f} across {stats['count']} {units} "
            f"(confidence: {1.0 - min(stats['std']*2, 0.9):.4f}). "
            f"{stats['above_threshold']}/{stats['count']} {units} exceed the FAKE "
            f"threshold ({threshold})."
        )
        if top_region:
            summary += (
                f" Most-implicated region: {top_region[0]} "
                f"({top_region[1]['attention_pct']}% of attention energy)."
            )
    elif mean_p < 1 - threshold:
        summary = (
            f"{detector_name} REAL: Media classified as authentic. "
            f"Average fake probability {mean_p:.4f} across {stats['count']} {units}. "
            f"No significant anomalies detected."
        )
        if top_region:
            summary += (
                f" Attention peaks in {top_region[0]} "
                f"({top_region[1]['attention_pct']}%), consistent with natural structure."
            )
    else:
        summary = (
            f"{detector_name} UNCERTAIN: Unable to reach a conclusive classification. "
            f"Average fake probability {mean_p:.4f} across {stats['count']} {units}."
        )

    for k, v in special_signals.items():
        factors.append(f"{k}: {v}")

    factors.append({"face": f"Analyzed {stats['count']} face(s) across sampled frames",
                    "frame": f"Analyzed {stats['count']} whole frame(s)",
                    "clip": f"Analyzed {stats['count']} clip(s) of consecutive frames",
                    "window": f"Analyzed {stats['count']} audio window(s)"}.get(unit, f"Analyzed {stats['count']} {units}"))
    factors.append(f"Inference performed on {device_name}")
    return summary, factors
