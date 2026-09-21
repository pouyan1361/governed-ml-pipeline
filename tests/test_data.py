"""
DATA TESTS: governance rules written as code.

Each test is a control. It has an owner (the policy owner), a trigger
(every pipeline run, every pull request), and evidence (the pytest output
in the CI log). When a test fails, the pipeline stops before training.

Run them with:   pytest -q tests/
"""
import pandas as pd
import pytest

from src.prepare import LABEL, load_policy, prepare_data

POLICY = load_policy()


@pytest.fixture(scope="module")
def processed():
    return prepare_data()


def test_required_columns_present(processed):
    """Schema check: the columns the models depend on must exist."""
    required = ["annual_income", "debt_to_income", "credit_history_years",
                "delinquencies_2y", "employment_years", "loan_amount",
                POLICY["protected_attribute"], LABEL]
    missing = [c for c in required if c not in processed.columns]
    assert not missing, f"Missing required columns: {missing}"


def test_no_missing_values_in_numeric_features(processed):
    """Completeness check: models cannot train on blanks."""
    numeric = processed.select_dtypes("number")
    bad = numeric.columns[numeric.isna().any()].tolist()
    assert not bad, f"Missing values remain in: {bad}"


def test_value_ranges_are_plausible(processed):
    """Validity check: catch unit errors and corrupt records early."""
    assert (processed["annual_income"] > 0).all(), "Non-positive income found"
    assert processed["debt_to_income"].between(0, 2).all(), "Debt-to-income outside 0 to 2"
    assert (processed["delinquencies_2y"] >= 0).all(), "Negative delinquency count"


def test_label_is_binary(processed):
    assert set(processed[LABEL].unique()) <= {0, 1}, "Label must be 0 or 1"


def test_no_pii_columns(processed):
    """PRIVACY CONTROL: no direct identifier or quasi-identifier may reach training.
    The list comes from governance/policy.json, not from this file."""
    leaked = [c for c in POLICY["pii_columns"] if c in processed.columns]
    assert not leaked, f"PII or quasi-identifiers reached the training data: {leaked}"


def test_no_known_leakage_columns(processed):
    """LEAKAGE CONTROL (known list): no column recorded after the decision."""
    leaked = [c for c in POLICY["leakage_columns"] if c in processed.columns]
    assert not leaked, f"Post-decision (leakage) columns present: {leaked}"


def test_no_suspiciously_predictive_column(processed):
    """LEAKAGE CONTROL (unknown unknowns): a column that predicts the label
    almost perfectly is usually leakage, even if nobody put it on a list.
    This is a DETECTIVE check: it catches problems the list did not anticipate."""
    limit = POLICY["max_feature_label_correlation"]
    numeric = processed.select_dtypes("number").drop(columns=[LABEL])
    corr = numeric.corrwith(processed[LABEL]).abs()
    suspicious = corr[corr > limit].round(3).to_dict()
    assert not suspicious, f"Columns correlated with the label above {limit}: {suspicious}"
