# Governed ML Pipeline

**BANA 690A AI Governance, Week 7 lab: CI/CD as a governance control**

In BANA 622 you learned CI/CD as a way to ship code safely. Tonight the same
pipeline ships a *decision*: a credit model that recommends approving or
declining loan applications. The difference is that a model can pass every
unit test and still be unfair, leak personal data, or learn from the future.
This repo turns those governance rules into code that runs automatically.

The data is synthetic. No real person appears in it.

---

## What is in this repo

```
governance/policy.json     The rules. Thresholds set by the AI Risk Committee in advance.
data/applicants_raw.csv    5,000 synthetic loan applications (with problems, on purpose)
src/prepare.py             Stage: clean the data           <- Lab 1 fixes this file
src/train.py               Stage: train a model from a config
src/evaluate.py            Stage: measure overall AND by region
src/gate.py                Stage: PASS or BLOCKED against the policy
src/model_card.py          Stage: write the evidence
tests/test_data.py         Data rules as tests (3 fail at the start: that is Lab 1)
tests/test_gate.py         Tests of the gate itself
configs/v1.json            The baseline model
configs/v2.json            A "better" model. Run it and see.
model_config.json          YOUR model (Lab 2 challenge)
run_pipeline.py            Runs every stage in order, exactly like CI
ci/check_policy_unchanged.py   Blocks pull requests that edit the policy
.github/workflows/         The GitHub Actions pipeline
```

---

## Setup (once, in VS Code)

1. Fork this repo on GitHub (button at top right), then in VS Code:
   **View > Command Palette > Git: Clone**, paste the URL of **your fork**.
   *No git? Use the green Code button > Download ZIP, unzip, and open the folder in VS Code.
   You can do Labs 1 and 2 this way; Lab 3 needs git.*
2. Open a terminal in VS Code (**Terminal > New Terminal**) and run:

   **Windows**
   ```
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```
   **Mac**
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```
   *If Windows refuses to activate the environment, skip the first two lines
   and just run the pip install. It will still work.*
3. Check it: `python -m pytest -q tests/` should show **3 failed, 8 passed**.
   Those three failures are Lab 1.

---

## The labs

**Lab 1: Tests as controls.** Run the tests, read the three failures, and fix
`drop_identifiers()` and `remove_leakage()` in `src/prepare.py`. Rule: read the
lists from the policy, do not hard-code them. Done when all 11 tests pass.

**Lab 2: The deployment gate.**
```
python run_pipeline.py --config configs/v1.json
python run_pipeline.py --config configs/v2.json
```
One passes, one is blocked. v2 has the higher AUC. Work out why it is blocked.
Then the challenge: edit `model_config.json` to build the model with the
**highest overall AUC that still passes the gate**, and run
`python run_pipeline.py`. Allowed model types: `logistic_regression`,
`random_forest`, `gradient_boosting`. Available features include
`log_income` and `loan_to_income`.

**Lab 3: Ship it through CI.** Commit your fixed `src/prepare.py` and your
`model_config.json` to a branch, push to your fork, and open a pull request to
the instructor's repo. Fill in the template. GitHub Actions will run the same
pipeline once the instructor approves the run. Your pull request is your
submission: it will not be merged.

---

## Rules

* Do not edit anything in `governance/`, `tests/`, `src/gate.py`, `ci/`, or
  `.github/`. The pipeline will block your pull request if you do. That is the point.
* Every run writes evidence to `artifacts/`: `metrics.json`, `gate_report.json`,
  `model_card.md`, and a line in `audit_log.jsonl`. Read them.
