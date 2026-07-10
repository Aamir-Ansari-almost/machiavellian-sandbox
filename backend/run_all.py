"""
run_all.py — run every core experiment batch back-to-back, in one go.

Fires run_experiment.py once per scenario as a subprocess and WAITS for each to
finish before starting the next (no parallelism — the local llama.cpp server
serves one run at a time anyway). The communication flag for each cell is baked
into its scenario YAML; the comment beside each entry is just a reminder, there is
nothing to toggle here.

Prereq: the llama.cpp server must be running on :8080.

    python run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RUNNER = HERE / "run_experiment.py"
RUNS_DIR = HERE / "runs"
PY = sys.executable  # use the same (venv) python that's running this script

RUNS_PER_BATCH = 20  # repetitions per scenario — raise for tighter confidence intervals

# (scenario yaml name, run_id tag, comm state — the last field is a reminder only)
BATCHES = [
    ("axelrod_known",       "rr_known",         "comm OFF | known horizon (rational baseline)"),
    ("axelrod_hidden",      "rr_hidden_nocomm", "comm OFF | hidden horizon (silent)"),
    ("axelrod_hidden_comm", "rr_hidden_comm",   "comm ON  | hidden horizon (talking)"),
    ("axelrod_forgiving",   "rr_forgiving",     "comm ON  | forgiving persona"),
    ("punic",               "rr_punic",         "comm ON  | persona vs payoff (war of attrition)"),
]


def main() -> None:
    t0 = time.time()
    total = len(BATCHES)
    print(f"Running {total} batches x {RUNS_PER_BATCH} runs each, one after another.\n")

    for i, (scenario, tag, note) in enumerate(BATCHES, 1):
        print("\n" + "=" * 72)
        print(f"  BATCH {i}/{total}  |  {scenario}  |  {note}")
        print("=" * 72, flush=True)
        cmd = [PY, str(RUNNER),
               "--scenario", scenario,
               "--runs", str(RUNS_PER_BATCH),
               "--tag", tag]
        # ticks are NOT overridden here — each scenario keeps the horizon set in its YAML.
        result = subprocess.run(cmd, cwd=str(HERE))
        if result.returncode != 0:
            print(f"\n!! BATCH {i} ({scenario}) FAILED (exit {result.returncode}). Stopping.")
            sys.exit(result.returncode)

    mins = (time.time() - t0) / 60
    print("\n" + "=" * 72)
    print(f"  ALL {total} BATCHES DONE in {mins:.1f} min. Analyse each with:")
    print("=" * 72)
    for scenario, tag, _ in BATCHES:
        manifests = sorted(RUNS_DIR.glob(f"*{tag}*_manifest.json"))
        if manifests:
            batch = manifests[-1].name.replace("_manifest.json", "")
            print(f"    python analyse.py --batch {batch}")


if __name__ == "__main__":
    main()
