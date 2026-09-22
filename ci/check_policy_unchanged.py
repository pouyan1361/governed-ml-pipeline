"""
CI STAGE: POLICY INTEGRITY CHECK (separation of duties)

A model author must not be able to pass the gate by lowering the gate.
This check fails any pull request that edits the policy, the tests, the
gate, or the CI machinery itself.

Policy changes are still possible, but not through a model pull request:
they need their own pull request, reviewed and approved by the policy owner
(enforced by .github/CODEOWNERS and branch protection on main).

Usage in CI:   python ci/check_policy_unchanged.py <base_commit_sha>
"""
import subprocess
import sys

PROTECTED = ["governance/", "tests/", "src/gate.py", "ci/", ".github/", ".devcontainer/"]


def changed_files(base):
    out = subprocess.run(["git", "diff", "--name-only", base, "HEAD"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print("Could not compare against the base commit:")
        print(out.stderr.strip())
        print("Failing closed: a control that cannot run must not report success.")
        sys.exit(1)
    return [f for f in out.stdout.splitlines() if f.strip()]


def main():
    if len(sys.argv) < 2 or not sys.argv[1]:
        print("No base commit supplied (not a pull request). Policy integrity check skipped.")
        return
    files = changed_files(sys.argv[1])
    policy_changes = [f for f in files if any(f.startswith(p) for p in PROTECTED)]
    print(f"Files changed in this pull request: {len(files)}")
    for f in files:
        print(f"  {f}")
    if policy_changes:
        print("\nPROTECTED FILE CHANGE DETECTED:")
        for f in policy_changes:
            print(f"  {f}")
        print("\nA model pull request may not modify the policy, tests, or gate that judge it.")
        print("Submit control changes separately for AI Risk Committee approval.")
        sys.exit(1)
    print("\nPolicy unchanged. Integrity check passed.")


if __name__ == "__main__":
    main()
