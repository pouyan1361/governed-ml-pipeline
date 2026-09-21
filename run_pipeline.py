"""
THE PIPELINE: run every stage, in order, exactly as CI does.

    python run_pipeline.py                        # uses model_config.json (your model)
    python run_pipeline.py --config configs/v1.json
    python run_pipeline.py --config configs/v2.json

Stages
  0  Data tests      pytest tests/        (stop if the data breaks policy)
  1  Policy pre-check                     (stop if the config uses prohibited features)
  2  Prepare         src/prepare.py
  3  Train           src/train.py
  4  Evaluate        src/evaluate.py
  5  Gate            src/gate.py          (PASS or BLOCKED)
  6  Model card      src/model_card.py    (written either way: it is evidence)
  7  Audit log       artifacts/audit_log.jsonl (one line appended per run)

Exit code 0 means PASS. Exit code 1 means a test failed or the gate BLOCKED.
CI reads the exit code: 1 turns the build red.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import evaluate, gate, model_card, prepare, train  # noqa: E402

ARTIFACTS = ROOT / "artifacts"


def banner(n, text):
    print(f"\n[stage {n}] {text}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def git_commit():
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"][:7]
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "local"
    except Exception:
        return "local"


def append_audit(entry):
    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / "audit_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Governed ML pipeline")
    ap.add_argument("--config", default="model_config.json", help="model config file")
    ap.add_argument("--skip-tests", action="store_true", help="skip stage 0 (CI runs tests as its own step)")
    args = ap.parse_args()

    config_path = ROOT / args.config
    config = train.load_config(config_path)
    policy = prepare.load_policy()
    run_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "config_file": args.config,
        "model_version": config["model_version"],
        "author": config.get("author", ""),
        "data_hash": sha256(prepare.RAW_PATH),
        "policy_hash": sha256(prepare.POLICY_PATH)[:12],
        "policy_version": policy["policy_version"],
    }
    print(f"Governed ML pipeline  |  config {args.config}  |  model {config['model_version']}  |  commit {run_info['commit']}")

    if not args.skip_tests:
        banner(0, "Data tests (pytest)")
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/"], cwd=ROOT)
        if result.returncode != 0:
            print("\nSTOPPED: data tests failed. The pipeline will not train on data that breaks policy.")
            append_audit({**run_info, "stage_reached": "data_tests", "decision": "STOPPED"})
            sys.exit(1)

    banner(1, "Policy pre-check on the model config (preventive)")
    blocked = sorted(set(config["features"]) & set(policy["prohibited_features"]))
    if blocked:
        print(f"  BLOCKED before training: config uses prohibited features {blocked}.")
        print("  Preventive controls run BEFORE the risky step, not after it.")
        append_audit({**run_info, "stage_reached": "policy_precheck", "decision": "BLOCKED",
                      "failed_checks": ["no_prohibited_features"]})
        sys.exit(1)
    print("  no prohibited features in the config")

    banner(2, "Prepare data")
    df = prepare.prepare_data()
    train_df, test_df = prepare.split(df)
    print(f"  train={len(train_df)}  test={len(test_df)}  columns={len(df.columns)}")

    banner(3, f"Train {config['model_type']} on {len(config['features'])} features")
    model = train.train(config, train_df)
    train.save_model(model)

    banner(4, "Evaluate overall and by region")
    metrics = evaluate.evaluate(model, config, test_df, group_col=policy["protected_attribute"])
    evaluate.save_metrics(metrics)
    print(f"  overall AUC {metrics['overall_auc']:.4f}   accuracy {metrics['overall_accuracy']:.4f}")
    for g, m in metrics["groups"].items():
        print(f"  {g:<6} n={m['n']:<5} AUC {m['auc']:.4f}  approval {m['approval_rate']:.4f}  TPR {m['tpr']:.4f}")

    banner(5, "Deployment gate")
    report = gate.run_gate(metrics, policy)
    gate.save_report(report)
    gate.print_report(report, metrics)

    banner(6, "Model card")
    card_path = model_card.save_card(model_card.build_card(config, metrics, report, policy, run_info))
    print(f"  written to {card_path.relative_to(ROOT)}")

    banner(7, "Audit log")
    append_audit({**run_info, "stage_reached": "gate", "decision": report["decision"],
                  "overall_auc": metrics["overall_auc"],
                  "approval_disparity_ratio": metrics["approval_disparity_ratio"],
                  "failed_checks": [r["check"] for r in report["checks"] if not r["passed"]]})
    print("  appended to artifacts/audit_log.jsonl")

    sys.exit(0 if report["decision"] == "PASS" else 1)


if __name__ == "__main__":
    main()
