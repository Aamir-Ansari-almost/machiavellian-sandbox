"""
Experiment runner: fire many simulation runs across a parameter sweep, each into
its own run_id, so analysis can aggregate over repetitions and conditions.

Examples
--------
# Axelrod strategy-consistency: same config, 5 repetitions
python run_experiment.py --scenario axelrod --ticks 20 --runs 5 --tag moral

# Senate H1 scarcity sweep: 4 scarcity levels x 3 repetitions = 12 runs
python run_experiment.py --scenario senate --scarcity 0.5 1.0 1.5 2.0 --runs 3 --tag h1
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from core.scenario_loader import load_scenario
from core.tick_engine import TickEngine

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
RUNS_DIR = Path(__file__).parent / "runs"


async def run_one(scenario_name: str, scarcity: float | None, ticks: int | None, run_id: str) -> dict:
    scenario = load_scenario(SCENARIOS_DIR / f"{scenario_name}.yaml")
    if scarcity is not None:
        scenario.scarcity = scarcity
    if ticks is not None:
        scenario.ticks = ticks

    engine = TickEngine(scenario, run_id=run_id, verbose=False)
    started = time.time()
    try:
        await engine.run()
    finally:
        engine.close()
    return {
        "run_id": run_id,
        "scenario": scenario_name,
        "scarcity": scenario.scarcity,
        "ticks": scenario.ticks,
        "seconds": round(time.time() - started, 1),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of simulations across a parameter sweep.")
    parser.add_argument("--scenario", default="senate", help="scenario name (no .yaml)")
    parser.add_argument("--scarcity", type=float, nargs="*", default=[None],
                        help="one or more scarcity values to sweep; omit to use the scenario default")
    parser.add_argument("--ticks", type=int, default=None, help="override tick count")
    parser.add_argument("--runs", type=int, default=3, help="repetitions per scarcity value")
    parser.add_argument("--tag", default="exp", help="label included in each run_id and the manifest filename")
    args = parser.parse_args()

    RUNS_DIR.mkdir(exist_ok=True)
    batch = f"{args.scenario}_{args.tag}_{int(time.time())}"
    scarcities = args.scarcity if args.scarcity else [None]

    total = len(scarcities) * args.runs
    print(f"Experiment batch: {batch}")
    print(f"  scenario={args.scenario}  scarcity={scarcities}  runs={args.runs}  total={total}\n")

    manifest: list[dict] = []
    done = 0
    for scarcity in scarcities:
        sc_label = "default" if scarcity is None else str(scarcity).replace(".", "p")
        for rep in range(args.runs):
            run_id = f"{batch}_sc{sc_label}_r{rep}"
            done += 1
            print(f"  [{done}/{total}] running {run_id} ...", end=" ", flush=True)
            info = await run_one(args.scenario, scarcity, args.ticks, run_id)
            manifest.append(info)
            print(f"done ({info['seconds']}s)")

    manifest_path = RUNS_DIR / f"{batch}_manifest.json"
    manifest_path.write_text(json.dumps({"batch": batch, "runs": manifest}, indent=2), encoding="utf-8")
    print(f"\nBatch complete. {total} runs written to simulation.db.")
    print(f"Manifest: {manifest_path}")
    print(f"Analyse with: python analyse.py --batch {batch}")


if __name__ == "__main__":
    asyncio.run(main())
