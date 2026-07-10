"""
Static figures for the paper / presentation. Each function takes already-computed
data and writes a PNG; nothing here touches the database directly (analyse.py does
the querying and passes results in). Headless backend so it runs without a display.
"""

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")

# Presentation defaults — bigger type, consistent palette across every figure so
# the audience learns the colour code once. Stable per-agent colours by index.
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "legend.fontsize": 11,
})

_AGENT_PALETTE = ["#2c7fb8", "#d95f0e", "#31a354", "#756bb1", "#c51b8a"]


def _agent_color(agents: list[str]) -> dict[str, str]:
    return {a: _AGENT_PALETTE[i % len(_AGENT_PALETTE)] for i, a in enumerate(sorted(agents))}


def _label_bars(ax, fmt="{:.0f}") -> None:
    """Annotate each bar with its value — audiences can't read heights off an axis."""
    for p in ax.patches:
        h = p.get_height()
        if h == 0:
            continue
        ax.annotate(fmt.format(h), (p.get_x() + p.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=10, xytext=(0, 2),
                    textcoords="offset points")


def cooperation_over_time(per_tick_coop: dict[str, list[float]], out: Path, title: str) -> None:
    """
    Per-tick cooperation level, one line per run. In a 2-agent game the value is
    the share of players cooperating that tick: 1.0 = both cooperate (mutual
    benefit), 0.5 = one defects (exploitation), 0.0 = both defect (the trap).
    Shows how/when cooperation collapses and how variable that is across runs.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    n_ticks = max((len(s) for s in per_tick_coop.values()), default=0)

    # Shade the three regimes so the y-value is self-explanatory.
    ax.axhspan(0.75, 1.05, color="#e8f5e9", zorder=0)
    ax.axhspan(0.25, 0.75, color="#fff8e1", zorder=0)
    ax.axhspan(-0.05, 0.25, color="#ffebee", zorder=0)

    for run_id, series in per_tick_coop.items():
        ticks = range(1, len(series) + 1)
        label = run_id.split("_r")[-1]
        # tiny vertical jitter so overlapping flat lines stay distinguishable
        jitter = (int(label) - 2) * 0.012 if label.isdigit() else 0.0
        ax.plot(ticks, [v + jitter for v in series], marker="o", markersize=4,
                linewidth=2, alpha=0.9, label=f"run {label}")

    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["both defect", "one defects", "both cooperate"])
    ax.set_xlim(0.5, n_ticks + 0.5)
    ax.set_xticks(range(1, n_ticks + 1, max(1, n_ticks // 10)))
    ax.set_xlabel("tick")
    ax.set_title(title)
    ax.legend(title="run", ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def first_betrayal_histogram(ticks: list[int | None], n_ticks: int, out: Path, title: str) -> None:
    """Distribution of when the first betrayal landed across runs (None = never)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    landed = [t for t in ticks if t is not None]
    never = len(ticks) - len(landed)

    bins = np.arange(0.5, n_ticks + 1.5, 1)
    ax.hist(landed, bins=bins, color="#c0392b", alpha=0.85, edgecolor="white",
            label=f"cooperation broke ({len(landed)} runs)")
    if never:
        ax.bar([n_ticks + 1], [never], color="#27ae60", alpha=0.85, edgecolor="white",
               label=f"never broke ({never} runs)")

    step = max(1, n_ticks // 10)
    xt = list(range(1, n_ticks + 1, step))
    xl = [str(t) for t in xt]
    if never:
        xt.append(n_ticks + 1)
        xl.append("never")
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if landed:
        ax.axvline(float(np.mean(landed)), color="#2c3e50", linestyle="--", linewidth=1.5)
        ax.text(float(np.mean(landed)), ax.get_ylim()[1] * 0.95,
                f" mean tick {np.mean(landed):.1f}", color="#2c3e50", fontsize=10, va="top")
    ax.set_xlabel("tick of first betrayal")
    ax.set_ylabel("number of runs")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def strategy_distribution(agent_strategies: dict[str, list[str]], out: Path, title: str) -> None:
    """Grouped bar chart: how often each agent landed on each emergent strategy."""
    agents = list(agent_strategies.keys())
    all_strats = sorted({s for strats in agent_strategies.values() for s in strats})
    counts = {a: Counter(agent_strategies[a]) for a in agents}
    colors = _agent_color(agents)
    n_runs = max((len(s) for s in agent_strategies.values()), default=1)

    x = np.arange(len(all_strats))
    width = 0.8 / max(1, len(agents))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, a in enumerate(agents):
        heights = [counts[a].get(s, 0) for s in all_strats]
        ax.bar(x + i * width, heights, width, label=a, color=colors[a], alpha=0.9)

    pretty = [s.replace("_", " ") for s in all_strats]
    ax.set_xticks(x + width * (len(agents) - 1) / 2)
    ax.set_xticklabels(pretty, rotation=15, ha="right")
    ax.set_ylabel("number of runs")
    ax.set_ylim(0, n_runs + 0.6)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # counts are integers
    ax.set_title(title)
    ax.legend(title="agent")
    _label_bars(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def score_comparison(scores_per_run: dict[str, dict[str, float]], out: Path, title: str) -> None:
    """Final score per agent across runs — the 'who wins' / alignment-tax picture."""
    runs = list(scores_per_run.keys())
    agents = sorted({a for s in scores_per_run.values() for a in s})
    colors = _agent_color(agents)
    x = np.arange(len(runs))
    width = 0.8 / max(1, len(agents))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, a in enumerate(agents):
        heights = [scores_per_run[r].get(a, 0.0) for r in runs]
        ax.bar(x + i * width, heights, width, label=a, color=colors[a], alpha=0.9)
        mean = float(np.mean(heights))
        ax.axhline(mean, color=colors[a], linestyle="--", linewidth=1.4, alpha=0.7)
        ax.text(-0.42, mean, f"{a} mean {mean:.0f}", color=colors[a],
                va="bottom", ha="left", fontsize=10, fontweight="bold")

    ax.set_xticks(x + width * (len(agents) - 1) / 2)
    ax.set_xticklabels([f"run {r.split('_r')[-1]}" for r in runs])
    ax.set_ylabel("final score")
    ax.set_title(title)
    ax.legend(title="agent", loc="upper right")
    ax.margins(x=0.04)
    _label_bars(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ── Senate-oriented plots (used later for H1) ─────────────────────────────────

def betrayal_vs_scarcity(points: list[tuple[float, float]], out: Path, title: str) -> None:
    """Scatter + trend of betrayal rate vs scarcity — the H1 figure."""
    if not points:
        return
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys, s=60, color="#c0392b", alpha=0.8, zorder=3)
    if len(set(xs)) > 1:
        m, b = np.polyfit(xs, ys, 1)
        xline = np.linspace(xs.min(), xs.max(), 50)
        plt.plot(xline, m * xline + b, "--", color="#2c3e50", alpha=0.7)
        r = np.corrcoef(xs, ys)[0, 1]
        plt.text(0.05, 0.92, f"r = {r:.2f}", transform=plt.gca().transAxes, fontsize=11)
    plt.xlabel("scarcity")
    plt.ylabel("betrayal rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def trust_heatmap(matrix: np.ndarray, ticks: list[int], pair_labels: list[str], out: Path, title: str) -> None:
    """Trust over time: rows = directed pairs, cols = ticks, color = trust."""
    plt.figure(figsize=(11, max(4, len(pair_labels) * 0.4)))
    sns.heatmap(matrix, cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                xticklabels=ticks, yticklabels=pair_labels,
                cbar_kws={"label": "trust"})
    plt.xlabel("tick")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
