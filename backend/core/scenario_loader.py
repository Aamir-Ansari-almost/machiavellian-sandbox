from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ResourceDef:
    name: str
    initial: float
    decay_per_tick: float


@dataclass
class AgentConfig:
    name: str
    persona: str
    hidden_agenda: str
    initial_resources: dict[str, float]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoalitionThresholds:
    form: float = 0.6
    dissolve: float = 0.3


@dataclass
class PayoffMatrix:
    """
    Joint payoff table for two-action (cooperate/defect) scenarios.
    Each entry is {actor: float, target: float} — points added to primary resource.

    Classic prisoner's dilemma:
        both_cooperate  (3, 3)
        actor_defects   (5, 0)   — actor defects, target cooperated
        target_defects  (0, 5)   — actor cooperated, target defected
        both_defect     (1, 1)
    """

    both_cooperate: dict[str, float]
    actor_defects: dict[str, float]
    target_defects: dict[str, float]
    both_defect: dict[str, float]


@dataclass
class ScenarioConfig:
    name: str
    ticks: int
    scarcity: float
    extra: dict[str, Any]
    resources: list[ResourceDef]
    agents: list[AgentConfig]
    thresholds: CoalitionThresholds
    payoff_matrix: PayoffMatrix | None = (
        None  # None → use transfer-based _ACTION_EFFECTS
    )


def load_scenario(path: str | Path) -> ScenarioConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    scenario_block = raw["scenario"]

    resources = [
        ResourceDef(
            name=r["name"],
            initial=float(r["initial"]),
            decay_per_tick=float(r["decay_per_tick"]),
        )
        for r in raw.get("resources", [])
    ]

    agents = [
        AgentConfig(
            name=a["name"],
            persona=str(a["persona"]).strip(),
            hidden_agenda=str(a["hidden_agenda"]).strip(),
            initial_resources={
                k: float(v) for k, v in a.get("initial_resources", {}).items()
            },
            extra=a.get("extra", {}),
        )
        for a in raw.get("agents", [])
    ]

    thresholds_raw = raw.get("coalition_thresholds", {})
    thresholds = CoalitionThresholds(
        form=float(thresholds_raw.get("form", 0.6)),
        dissolve=float(thresholds_raw.get("dissolve", 0.3)),
    )

    payoff_matrix: PayoffMatrix | None = None
    if "payoff_matrix" in raw:
        pm = raw["payoff_matrix"]
        payoff_matrix = PayoffMatrix(
            both_cooperate={k: float(v) for k, v in pm["both_cooperate"].items()},
            actor_defects={k: float(v) for k, v in pm["actor_defects"].items()},
            target_defects={k: float(v) for k, v in pm["target_defects"].items()},
            both_defect={k: float(v) for k, v in pm["both_defect"].items()},
        )

    return ScenarioConfig(
        name=str(scenario_block["name"]),
        ticks=int(scenario_block.get("ticks", 50)),
        scarcity=float(scenario_block.get("scarcity", 1.0)),
        extra=scenario_block.get("extra", {}),
        resources=resources,
        agents=agents,
        thresholds=thresholds,
        payoff_matrix=payoff_matrix,
    )
