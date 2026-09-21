"""
GATE TESTS: testing the control itself.

If the gate has a bug, every model it approves is suspect. So the gate
gets its own tests, using made-up metrics where we already know the
right answer. These should pass from the start. Do not change them.
"""
from src.gate import run_gate
from src.prepare import load_policy

POLICY = load_policy()


def fake_metrics(**overrides):
    m = {
        "model_version": "test",
        "features": ["annual_income", "debt_to_income"],
        "overall_auc": 0.85,
        "groups": {
            "metro": {"n": 100, "auc": 0.85, "approval_rate": 0.50, "tpr": 0.80},
            "rural": {"n": 50, "auc": 0.84, "approval_rate": 0.45, "tpr": 0.78},
        },
        "approval_disparity_ratio": 0.90,
        "tpr_ratio": 0.975,
    }
    m.update(overrides)
    return m


def test_gate_passes_a_good_model():
    assert run_gate(fake_metrics(), POLICY)["decision"] == "PASS"


def test_gate_blocks_low_disparity_ratio():
    assert run_gate(fake_metrics(approval_disparity_ratio=0.60), POLICY)["decision"] == "BLOCKED"


def test_gate_blocks_prohibited_feature():
    m = fake_metrics(features=["annual_income", "region"])
    assert run_gate(m, POLICY)["decision"] == "BLOCKED"


def test_gate_blocks_weak_subgroup_even_if_average_is_fine():
    groups = {
        "metro": {"n": 100, "auc": 0.90, "approval_rate": 0.50, "tpr": 0.85},
        "rural": {"n": 50, "auc": 0.70, "approval_rate": 0.45, "tpr": 0.78},
    }
    assert run_gate(fake_metrics(groups=groups), POLICY)["decision"] == "BLOCKED"
