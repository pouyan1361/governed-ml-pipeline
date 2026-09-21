"""
STAGE 5: THE MODEL CARD

Every run, pass or blocked, produces a model card: a one-page, human-readable
record of what the model is, how it performed, and what the gate decided.

In CI this file is uploaded as a build artifact. That makes it evidence: an
auditor (Week 4) can download the card for any past run and see exactly what
was known when the deployment decision was made.
"""
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def build_card(config, metrics, report, policy, run_info):
    lines = []
    lines.append(f"# Model Card: Credit Decision Model {config['model_version']}")
    lines.append("")
    lines.append(f"**Gate decision: {report['decision']}**  (policy v{policy['policy_version']}, owner: {policy['owner']})")
    lines.append("")
    lines.append(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  |  commit `{run_info.get('commit', 'local')}`  |  data sha256 `{run_info.get('data_hash', '')[:12]}`")
    lines.append("")
    lines.append("## Model details")
    lines.append(f"- Author: {config.get('author', '')}")
    lines.append(f"- Description: {config.get('description', '')}")
    lines.append(f"- Algorithm: {config['model_type']}  |  decision threshold: {config['decision_threshold']}")
    lines.append(f"- Features ({len(config['features'])}): {', '.join(config['features'])}")
    lines.append("")
    lines.append("## Intended use")
    lines.append("- Drafts approve/decline recommendations for consumer loans, reviewed by a human loan officer.")
    lines.append("- Not for use as the sole basis of a credit decision. Not validated for business lending.")
    lines.append("")
    lines.append("## Performance (held-out test set)")
    lines.append(f"- Overall AUC {metrics['overall_auc']:.4f}  |  accuracy {metrics['overall_accuracy']:.4f}  |  n = {metrics['n_test']}")
    lines.append("")
    lines.append("| Group | n | AUC | Approval rate | TPR (good borrowers approved) |")
    lines.append("|---|---|---|---|---|")
    for g, m in metrics["groups"].items():
        lines.append(f"| {g} | {m['n']} | {m['auc']:.4f} | {m['approval_rate']:.4f} | {m['tpr']:.4f} |")
    lines.append("")
    lines.append(f"- Approval disparity ratio: {metrics['approval_disparity_ratio']:.4f} (policy minimum {policy['min_approval_disparity_ratio']})")
    lines.append(f"- TPR ratio: {metrics['tpr_ratio']:.4f} (policy minimum {policy['min_tpr_ratio']})")
    lines.append("")
    lines.append("## Gate checks")
    lines.append("| Check | Value | Threshold | Result |")
    lines.append("|---|---|---|---|")
    for r in report["checks"]:
        lines.append(f"| {r['check']} | {r['value']} | {r['threshold']} | {'pass' if r['passed'] else 'FAIL'} |")
    lines.append("")
    lines.append("## Known limitations")
    lines.append("- Trained on historical repayment outcomes, which may encode past lending practices.")
    lines.append("- Fairness is evaluated for region only. Other groups have not been tested.")
    lines.append("- Synthetic data: for teaching only.")
    lines.append("")
    return "\n".join(lines)


def save_card(text, path=None):
    ARTIFACTS.mkdir(exist_ok=True)
    path = path or ARTIFACTS / "model_card.md"
    with open(path, "w") as f:
        f.write(text)
    return path
