"""
STAGE 3: EVALUATION

Measures the model on the held-out test split, overall AND for each group of
the protected attribute (region: metro vs rural).

Why per group? A model can look excellent on average and be poor for the
smaller group. Averages hide exactly the people governance exists to protect.

Metrics produced
  auc            : ranking quality (0.5 = coin flip, 1.0 = perfect)
  accuracy       : share of correct approve/decline calls at the threshold
  approval_rate  : share of applicants the model would approve
  tpr            : true positive rate = share of applicants who DID repay
                   that the model would approve (good borrowers not turned away)
  approval_disparity_ratio : lowest group approval rate / highest group approval rate
                   (the "four-fifths" screening heuristic: below 0.80 is a red flag)
  tpr_ratio      : lowest group TPR / highest group TPR ("equal opportunity")
"""
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def group_metrics(y_true, scores, approved):
    positives = y_true == 1
    return {
        "n": int(len(y_true)),
        "auc": round(float(roc_auc_score(y_true, scores)), 4),
        "approval_rate": round(float(approved.mean()), 4),
        "tpr": round(float(approved[positives].mean()), 4),
    }


def evaluate(model, config, test_df, group_col="region", label="repaid"):
    scores = model.predict_proba(test_df[config["features"]])[:, 1]
    approved = scores >= config["decision_threshold"]
    y = test_df[label].values

    groups = {}
    for g in sorted(test_df[group_col].unique()):
        mask = (test_df[group_col] == g).values
        groups[g] = group_metrics(y[mask], scores[mask], approved[mask])

    rates = [m["approval_rate"] for m in groups.values()]
    tprs = [m["tpr"] for m in groups.values()]

    return {
        "model_version": config["model_version"],
        "author": config.get("author", ""),
        "model_type": config["model_type"],
        "features": config["features"],
        "decision_threshold": config["decision_threshold"],
        "n_test": int(len(y)),
        "overall_auc": round(float(roc_auc_score(y, scores)), 4),
        "overall_accuracy": round(float(accuracy_score(y, approved)), 4),
        "groups": groups,
        "approval_disparity_ratio": round(min(rates) / max(rates), 4) if max(rates) > 0 else 0.0,
        "tpr_ratio": round(min(tprs) / max(tprs), 4) if max(tprs) > 0 else 0.0,
    }


def save_metrics(metrics, path=None):
    ARTIFACTS.mkdir(exist_ok=True)
    path = path or ARTIFACTS / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path
