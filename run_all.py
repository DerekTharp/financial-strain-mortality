"""
Runs the analysis scripts in order.

    python run_all.py

The two Stata scripts run in batch mode. The Stata executable is taken from the STATA_BIN
environment variable, then from stata-mp on the PATH, then from the default macOS location
of StataNow.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

PIPELINE = [
    "01_primary_analysis.py",
    "01b_export_survey_data.py",
    "03_secondary_analysis.py",
    "04_flow_diagram.py",
    "05_sensitivity_diagnostics.py",
    "06_sensitivity_clustering.py",
    "07_competing_risks_heart_disease.do",
    "10_biomarker_analysis.py",
    "11_survey_prep.py",
    "12_survey_models.do",
    "02_figures.py",
]

STATA_BIN = (os.environ.get("STATA_BIN") or shutil.which("stata-mp")
             or "/Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp")


def run_python(script):
    return subprocess.run([sys.executable, script], cwd=PROJECT_DIR).returncode == 0


def run_stata(script):
    """Run a do-file in batch mode. Stata can exit with status 0 after an error in the
    do-file, so the log is checked for a return code as well as the exit status."""
    if not Path(STATA_BIN).exists():
        print(f"Stata not found at {STATA_BIN}. Set STATA_BIN to the stata-mp executable.")
        return False
    log = PROJECT_DIR / (Path(script).stem + ".log")
    log.unlink(missing_ok=True)
    status = subprocess.run([STATA_BIN, "-b", "do", script], cwd=PROJECT_DIR).returncode
    if status != 0:
        print(f"Stata exited with status {status} running {script}.")
        return False
    if not log.exists():
        print(f"Stata wrote no log for {script}.")
        return False
    errors = re.findall(r"^r\(\d+\);", log.read_text(errors="replace"), re.M)
    if errors:
        print(f"Stata error {errors[0]} in {log.name}.")
        return False
    return True


def main():
    for script in PIPELINE:
        print(f"\n--- {script} ---", flush=True)
        ok = run_stata(script) if script.endswith(".do") else run_python(script)
        if not ok:
            sys.exit(f"{script} failed.")
    print("\nAll scripts completed.")


if __name__ == "__main__":
    main()
