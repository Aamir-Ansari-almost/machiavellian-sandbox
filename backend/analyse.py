"""
Analyse an experiment batch: read its manifest, compute metrics across every run,
print a summary, and write figures to runs/<batch>_plots/.

    python analyse.py --batch axelrod_moral_1781946392

The manifest (written by run_experiment.py) lists the run_ids and conditions; this
script never re-runs simulations, it only reads simulation.db.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from infra.db import get_connection
from analysis import metrics as M
from analysis import plots as P

RUNS_DIR = Path(__file__).parent / "runs"


def find_manifest(batch: str) -> Path:
    exact = RUNS_DIR / f"{batch}_manifest.json"
    if exact.exists():
        return exact
    matches = sorted(RUNS_DIR.glob(f"*{batch}*manifest.json"))
    if not matches:
        raise FileNotFoundError(f"No manifest found for batch '{batch}' in {RUNS_DIR}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse an experiment batch and generate plots.")
    parser.add_argument("--batch", required=True, help="batch name (or substring) of the manifest")
    parser.add_argument("--resource", default=None, help="resource to score by (default: first in run)")
    parser.add_argument("--title", default=None,
                        help="clean title prefix for figures (default: the scenario name). "
                             "e.g. --title \"Moral vs Self-interested (iterated PD)\"")
    args = parser.parse_args()

    manifest = json.loads(find_manifest(args.batch).read_text(encoding="utf-8"))
    batch = manifest["batch"]
    runs = manifest["runs"]
    conn = get_connection()

    out_dir = RUNS_DIR / f"{batch}_plots"
    out_dir.mkdir(exist_ok=True)

    # Determine resource + agents from the first run.
    first_rid = runs[0]["run_id"]
    agents = M.agents_in_run(conn, first_rid)
    resource = args.resource or _infer_resource(conn, first_rid)
    max_ticks = max(M.n_ticks(conn, r["run_id"]) for r in runs)
    title_prefix = args.title or runs[0].get("scenario", batch).replace("_", " ").title()
    subtitle = f"{title_prefix}  ({len(runs)} runs)"

    # Collect per-run metrics.
    per_tick_coop: dict[str, list[float]] = {}
    first_betrayals: list[int | None] = []
    agent_strategies: dict[str, list[str]] = {a: [] for a in agents}
    scores_per_run: dict[str, dict[str, float]] = {}
    scarcity_betrayal: list[tuple[float, float]] = []

    print(f"\n{'=' * 78}")
    print(f"  BATCH: {batch}   ({len(runs)} runs, {len(agents)} agents, ~{max_ticks} ticks)")
    print(f"{'=' * 78}")
    header = f"  {'run':>4}  {'scarcity':>8}  {'first_betray':>12}  {'coop_rate':>9}  {'betray_rate':>11}  " + "  ".join(f"{a[:10]:>10}" for a in agents)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in runs:
        rid = r["run_id"]
        label = rid.split("_r")[-1]
        scarcity = r.get("scarcity", 1.0)

        per_tick_coop[rid] = M.cooperation_per_tick(conn, rid)
        fb = M.first_betrayal_tick(conn, rid)
        first_betrayals.append(fb)
        coop = M.cooperation_rate(conn, rid)
        br = M.betrayal_rate(conn, rid)
        scores = M.final_scores(conn, rid, resource)
        scores_per_run[rid] = scores
        scarcity_betrayal.append((scarcity, br))

        strat_cells = []
        for a in agents:
            s = M.classify_strategy(conn, rid, a)
            agent_strategies[a].append(s)
            strat_cells.append(f"{scores.get(a, 0):>4.0f}")

        print(f"  {label:>4}  {scarcity:>8}  {str(fb):>12}  {coop:>9.2f}  {br:>11.2f}  " + "  ".join(f"{c:>10}" for c in strat_cells))

    # Aggregate summary.
    landed = [t for t in first_betrayals if t is not None]
    print("\n  SUMMARY")
    print(f"    runs where cooperation broke:   {len(landed)}/{len(runs)}")
    if landed:
        print(f"    first betrayal: mean tick {np.mean(landed):.1f}, range {min(landed)}-{max(landed)}")
    print(f"    runs of sustained cooperation:  {len(runs) - len(landed)}/{len(runs)}")
    for a in agents:
        wins = sum(1 for rid in scores_per_run if scores_per_run[rid].get(a, 0) == max(scores_per_run[rid].values()))
        mean_score = np.mean([scores_per_run[rid].get(a, 0) for rid in scores_per_run])
        strat_mix = ", ".join(f"{s}:{agent_strategies[a].count(s)}" for s in sorted(set(agent_strategies[a])))
        print(f"    {a:8} mean_score={mean_score:6.1f}  wins/ties={wins}/{len(runs)}  strategies=[{strat_mix}]")

    # Plots.
    P.cooperation_over_time(per_tick_coop, out_dir / "cooperation_over_time.png",
                            f"Cooperation over time\n{subtitle}")
    P.first_betrayal_histogram(first_betrayals, max_ticks, out_dir / "first_betrayal_hist.png",
                               f"When cooperation first broke\n{subtitle}")
    P.strategy_distribution(agent_strategies, out_dir / "strategy_distribution.png",
                            f"Emergent strategies\n{subtitle}")
    P.score_comparison(scores_per_run, out_dir / "score_comparison.png",
                       f"Final scores by run\n{subtitle}")
    if len({s for s, _ in scarcity_betrayal}) > 1:
        P.betrayal_vs_scarcity(scarcity_betrayal, out_dir / "betrayal_vs_scarcity.png",
                               f"Betrayal rate vs scarcity\n{subtitle}")

    conn.close()
    print(f"\n  Plots written to: {out_dir}")
    print(f"{'=' * 78}\n")


def _infer_resource(conn, run_id: str) -> str:
    import json as _json
    row = conn.execute(
        "SELECT resources_snapshot FROM events WHERE run_id=? LIMIT 1", (run_id,)
    ).fetchone()
    snap = _json.loads(row[0])
    return next(iter(snap.keys()))


if __name__ == "__main__":
    main()
