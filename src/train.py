"""
STAGE 2: TRAINING

Builds the model described in a config file (configs/v1.json, configs/v2.json,
or your own model_config.json) and trains it on the training split.

The config is the whole point: the model is DEFINED by a small, reviewable
file. A reviewer reading a pull request can see exactly which features and
which algorithm changed, without reading any Python.
"""
import json
import pickle
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

ALLOWED_MODEL_TYPES = ["logistic_regression", "random_forest", "gradient_boosting"]


def load_config(path):
    with open(path) as f:
        return json.load(f)


def build_model(model_type, params=None):
    """Map a model_type string from the config to a scikit-learn estimator."""
    params = params or {}
    if model_type == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **params))
    if model_type == "random_forest":
        params.setdefault("random_state", 622)
        return RandomForestClassifier(**params)
    if model_type == "gradient_boosting":
        params.setdefault("random_state", 622)
        return GradientBoostingClassifier(**params)
    raise ValueError(f"model_type must be one of {ALLOWED_MODEL_TYPES}, got '{model_type}'")


def train(config, train_df, label="repaid"):
    missing = [f for f in config["features"] if f not in train_df.columns]
    if missing:
        raise ValueError(f"Config asks for features that do not exist in the data: {missing}")
    model = build_model(config["model_type"], dict(config.get("params", {})))
    model.fit(train_df[config["features"]], train_df[label])
    return model


def save_model(model, path=None):
    ARTIFACTS.mkdir(exist_ok=True)
    path = path or ARTIFACTS / "model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path
