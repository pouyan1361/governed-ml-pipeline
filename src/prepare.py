"""
STAGE 1: DATA PREPARATION

Turns the raw application file into a training-ready table.

You inherited this file from a previous analyst. The data tests in
tests/test_data.py say it has problems. Lab 1 is fixing them.

Remember from BANA 622: small functions that each do one job are easy to test.
That is exactly why this file is split into functions. Each one is a place
where a governance rule can be checked.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "applicants_raw.csv"
POLICY_PATH = ROOT / "governance" / "policy.json"

LABEL = "repaid"
SPLIT_SEED = 622


def load_policy(path=POLICY_PATH):
    """Read the governance policy. The policy is data, not code: it is the
    committee's decision, versioned in the repo like everything else."""
    with open(path) as f:
        return json.load(f)


def load_raw(path=RAW_PATH):
    return pd.read_csv(path)


def drop_identifiers(df):
    """Remove columns that identify a person.

    Previous analyst's note: "Dropped name, SSN, email, and street address.
    Kept zip and birth_date because they are useful for the model."

    LAB 1, TASK A: the PII test fails. Read the policy's pii_columns list and
    think about quasi-identifiers: zip code plus birth date is enough to
    re-identify most people, even without a name.
    """
    direct_identifiers = ["applicant_name", "ssn", "email", "street_address"]
    return df.drop(columns=direct_identifiers)


def remove_leakage(df):
    """Remove columns that would not exist at the moment of the lending decision.

    LAB 1, TASK B: the leakage test fails. A column recorded AFTER the decision
    (did the loan go to collections?) lets the model 'see the future'. It makes
    test scores look brilliant and the deployed model useless.
    The policy lists known leakage columns under leakage_columns.
    """
    # TODO: this function currently does nothing.
    return df


def impute_missing(df):
    """Fill missing numeric values with the column median.
    A simple, documented rule beats an undocumented clever one."""
    out = df.copy()
    for col in ["annual_income", "employment_years"]:
        out[col] = out[col].fillna(out[col].median())
    return out


def add_features(df):
    """Optional engineered features. You may use these in your v3 model."""
    out = df.copy()
    out["log_income"] = np.log(out["annual_income"])
    out["loan_to_income"] = out["loan_amount"] / out["annual_income"]
    return out


def prepare_data(raw_path=RAW_PATH):
    """The full preparation stage, in order. Returns the processed table.

    Note: 'region' stays in the processed table. It is NOT allowed as a model
    feature (see prohibited_features in the policy), but the evaluation stage
    needs it to check whether the model treats metro and rural applicants
    fairly. You keep the sensitive attribute to TEST fairness, never to DECIDE.
    """
    df = load_raw(raw_path)
    df = drop_identifiers(df)
    df = remove_leakage(df)
    df = impute_missing(df)
    df = add_features(df)
    return df


def split(df):
    """Fixed-seed stratified split so every run, and every student, is comparable."""
    return train_test_split(df, test_size=0.3, random_state=SPLIT_SEED, stratify=df[LABEL])


if __name__ == "__main__":
    data = prepare_data()
    print(f"processed rows={len(data)} columns={len(data.columns)}")
    print(list(data.columns))
