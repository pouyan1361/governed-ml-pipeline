"""
Generates the synthetic loan application dataset used in the lab.

Everything here is fictional. Names, SSNs, emails, and addresses are fabricated
(SSNs use the 900 range, which is never issued). You do not need to run this
file: data/applicants_raw.csv is already in the repo. It is included so you can
see exactly how the data, and its built-in problems, were created.

Built-in problems (on purpose, for the lab):
  1. Direct identifiers (name, SSN, email, street address) are in the raw file.
  2. Quasi-identifiers (zip + birth date) are in the raw file.
  3. A post-decision column (collections_flag_post_decision) leaks the label.
  4. The historical label carries a penalty against rural applicants that is
     NOT explained by their finances (think: past lending practices).
  5. A vendor-supplied "area_risk_score" is strongly correlated with region,
     so a model can learn the rural penalty through it without ever seeing
     the region column. That is a proxy.
"""
import numpy as np
import pandas as pd

SEED = 622
N = 5000

FIRST = ["Alex", "Maria", "James", "Linh", "Omar", "Sofia", "Daniel", "Aisha", "Kevin", "Priya",
         "Carlos", "Emma", "Hiro", "Fatima", "Noah", "Grace", "Mateo", "Olivia", "Ravi", "Chloe"]
LAST = ["Garcia", "Nguyen", "Smith", "Patel", "Kim", "Lopez", "Johnson", "Chen", "Martinez", "Brown",
        "Hernandez", "Davis", "Singh", "Wilson", "Ramirez", "Taylor", "Ali", "Moore", "Lee", "Clark"]
STREETS = ["Oak St", "Maple Ave", "Pine Rd", "Cedar Ln", "Elm Dr", "Lakeview Blvd", "Hillcrest Way"]


def generate(n=N, seed=SEED):
    rng = np.random.default_rng(seed)

    region = np.where(rng.random(n) < 0.72, "metro", "rural")
    rural = (region == "rural").astype(float)

    # Finances: rural applicants earn somewhat less on average
    income = np.exp(rng.normal(11.05 - 0.07 * rural, 0.38, n)).round(-2)
    dti = np.clip(rng.normal(0.34, 0.12, n), 0.02, 1.2).round(3)
    history = np.clip(rng.gamma(3.0, 3.2, n), 0, 40).round(1)
    delinq = rng.poisson(0.45, n)
    employment = np.clip(rng.gamma(2.2, 3.0, n), 0, 40).round(1)
    loan = np.clip(rng.normal(21400, 8200, n), 2000, 60000).round(-2)

    # True creditworthiness from finances only
    z = (1.05 * (np.log(income) - 11.0) / 0.38
         - 1.25 * (dti - 0.34) / 0.12
         + 0.45 * (history - 9.6) / 5.5
         - 0.75 * delinq
         + 0.25 * (employment - 6.6) / 4.4
         - 0.35 * (loan / income - 0.35) / 0.2)

    # Historical label: repaid, with a penalty on rural applicants that their
    # finances do not justify (the bias the model can inherit)
    logit = 0.75 + z - 1.3 * rural + rng.normal(0, 1.7, n)
    repaid = (logit > 0).astype(int)

    # Vendor "area risk score": mostly a region signal, plus noise
    area_risk = np.clip(35 + 38 * rural + rng.normal(0, 9, n), 0, 100).round(1)

    # Leakage: collections flag is set AFTER the decision, almost only for non-repayers
    collections = np.where(repaid == 0, rng.random(n) < 0.83, rng.random(n) < 0.03).astype(int)

    # Identifiers and quasi-identifiers
    names = [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)]
    ssn = [f"9{rng.integers(0, 100):02d}-{rng.integers(10, 100)}-{rng.integers(1000, 10000)}" for _ in range(n)]
    email = [f"{nm.lower().replace(' ', '.')}{rng.integers(1, 999)}@example.com" for nm in names]
    street = [f"{rng.integers(10, 9999)} {rng.choice(STREETS)}" for _ in range(n)]
    zips = np.where(rural == 1, rng.integers(95900, 96199, n), rng.integers(90001, 91999, n)).astype(str)
    birth = pd.to_datetime("1950-01-01") + pd.to_timedelta(rng.integers(0, 365 * 52, n), unit="D")

    df = pd.DataFrame({
        "applicant_id": [f"APP{i:05d}" for i in range(1, n + 1)],
        "applicant_name": names,
        "ssn": ssn,
        "email": email,
        "street_address": street,
        "zip": zips,
        "birth_date": birth.strftime("%Y-%m-%d"),
        "region": region,
        "annual_income": income,
        "debt_to_income": dti,
        "credit_history_years": history,
        "delinquencies_2y": delinq,
        "employment_years": employment,
        "loan_amount": loan,
        "area_risk_score": area_risk,
        "collections_flag_post_decision": collections,
        "repaid": repaid,
    })

    # A little real-world mess: some missing incomes and employment values
    miss_inc = rng.random(n) < 0.03
    miss_emp = rng.random(n) < 0.04
    df.loc[miss_inc, "annual_income"] = np.nan
    df.loc[miss_emp, "employment_years"] = np.nan
    return df


if __name__ == "__main__":
    out = generate()
    out.to_csv("data/applicants_raw.csv", index=False)
    print(f"wrote data/applicants_raw.csv  rows={len(out)}  repaid rate={out.repaid.mean():.3f}")
    print(out.groupby("region")["repaid"].agg(["count", "mean"]).round(3))
