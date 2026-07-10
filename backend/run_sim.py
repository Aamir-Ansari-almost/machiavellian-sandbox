import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime

from core.scenario_loader import load_scenario
from core.tick_engine import TickEngine

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
RUNS_DIR = Path(__file__).parent / "runs"


class _Tee:
    """Write to multiple streams at once (console + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Machiavellian Sandbox simulation."
    )
    parser.add_argument(
        "--scenario",
        default="senate",
        help="scenario name (file in scenarios/, no .yaml)",
    )
    parser.add_argument("--ticks", type=int, default=None, help="override tick count")
    parser.add_argument(
        "--scarcity", type=float, default=None, help="override scarcity multiplier"
    )
    parser.add_argument(
        "--run-id", default=None, help="explicit run_id (defaults to auto-generated)"
    )
    parser.add_argument(
        "--log-file",
        nargs="?",
        const="AUTO",
        default=None,
        help="write the transcript to a file. Pass a path, or use the flag alone "
        "to auto-name it runs/<run_id>.log",
    )
    args = parser.parse_args()

    scenario = load_scenario(SCENARIOS_DIR / f"{args.scenario}.yaml")
    if args.ticks is not None:
        scenario.ticks = args.ticks
    if args.scarcity is not None:
        scenario.scarcity = args.scarcity

    engine = TickEngine(scenario, run_id=args.run_id, verbose=True)

    log_handle = None
    original_stdout = sys.stdout
    if args.log_file is not None:
        if args.log_file == "AUTO":
            RUNS_DIR.mkdir(exist_ok=True)
            log_path = (
                RUNS_DIR
                / f"{engine.run_id}_{datetime.now().strftime("%Y%m%d%H%M%S")}.log"
            )
        else:
            log_path = Path(args.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "w", encoding="utf-8")
        sys.stdout = _Tee(original_stdout, log_handle)
        print(f"(transcript -> {log_path})")

    try:
        run_id = await engine.run()
        print(f"\nData written to simulation.db under run_id: {run_id}")
    finally:
        engine.close()
        sys.stdout = original_stdout
        if log_handle:
            log_handle.close()


if __name__ == "__main__":
    asyncio.run(main())
