# Governed ML Pipeline

**BANA 690A AI Governance: CI/CD as a governance control**

In BANA 622 you learned CI/CD as a way to ship code safely. Tonight the same
pipeline ships a *decision*: a credit model that recommends approving or
declining loan applications. A model can pass every unit test and still be
unfair, leak personal data, or learn from the future. This repo turns those
governance rules into code that runs automatically.

The data is synthetic. No real person appears in it.

---

## Start here: open your own Codespace (nothing to install)

1. **Fork** this repo: click **Fork** at the top right, then **Create fork**.
2. On **your fork** (your username at the top left), click the green
   **Code** button, choose the **Codespaces** tab, and click
   **Create codespace on main**.
3. Wait one to three minutes. VS Code opens in your browser, and the
   terminal at the bottom installs the libraries automatically.
   When it prints `Setup complete`, you are ready.
4. In the terminal, run:
   ```
   python -m pytest -q tests/
   ```
   You should see **3 failed, 8 passed**. Those three failures are Lab 1.

Your Codespace is a full computer in the cloud, set up by the file
`.devcontainer/devcontainer.json` in this repo. No virtual environment is
needed: the Codespace itself is the isolated environment.

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
ci/check_policy_unchanged.py   Blocks pull requests that edit protected files
.github/workflows/         The GitHub Actions pipeline
.devcontainer/             The Codespace setup (environment as code)
```

---

## The labs

**Lab 1: Tests as controls.** Fix `drop_identifiers()` and `remove_leakage()`
in `src/prepare.py`. Read the lists from the policy; do not hard-code them.
Progress: 3 failed, then 2 failed, then 11 passed.

**Lab 2: The deployment gate.**
```
python run_pipeline.py --config configs/v1.json
python run_pipeline.py --config configs/v2.json
```
One passes, one is blocked, and v2 has the higher AUC. Work out why. Then edit
`model_config.json` to build the model with the **highest overall AUC that
still passes the gate**, and run `python run_pipeline.py`.

**Lab 3: Ship it through CI.** In your Codespace: create a branch named
`student/yourname`, commit only `src/prepare.py` and `model_config.json`,
publish the branch, and open a pull request to **pouyan1361/governed-ml-pipeline**.
The checks run after the instructor approves. Your pull request is your
submission; it will not be merged.

---

## Rules

* Do not edit anything in `governance/`, `tests/`, `src/gate.py`, `ci/`,
  `.github/`, or `.devcontainer/`. The pipeline blocks pull requests that do.
* Every run writes evidence to `artifacts/`: `metrics.json`,
  `gate_report.json`, `model_card.md`, and a line in `audit_log.jsonl`.
* When you finish, stop your Codespace to save your free hours:
  github.com/codespaces, click the three dots next to it, **Stop codespace**.

---

## Backup: run it on your own laptop instead

If Codespaces is unavailable, clone your fork (or download the ZIP), open the
folder in VS Code, and in its terminal run:

```
# Windows                                   # Mac
python -m venv .venv                        python3 -m venv .venv
.venv\Scripts\activate                      source .venv/bin/activate
python -m pip install -r requirements.txt   python -m pip install -r requirements.txt
python -m pytest -q tests/                  python -m pytest -q tests/
```
