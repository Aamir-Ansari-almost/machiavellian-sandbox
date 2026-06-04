from typing import Optional

from core.scenario_loader import ScenarioConfig, ResourceDef

# Resource transfers applied to the primary resource (first in scenario list).
# actor = what the acting agent gains/loses
# target = what the target agent gains/loses
_ACTION_EFFECTS: dict[str, dict[str, float]] = {
    "cooperate": {"actor": -2.0, "target": +2.0},
    "defect":    {"actor": +5.0, "target": -5.0},
    "betray":    {"actor": +8.0, "target": -8.0},
    "negotiate": {"actor":  0.0, "target":  0.0},
    "ignore":    {"actor":  0.0, "target":  0.0},
}


class WorldState:
    def __init__(self, scenario: ScenarioConfig) -> None:
        self.tick: int = 0
        self.scarcity: float = scenario.scarcity
        self.scenario_name: str = scenario.name
        self.scenario_extra: dict = scenario.extra
        self._resource_defs: list[ResourceDef] = scenario.resources
        self._primary: str | None = scenario.resources[0].name if scenario.resources else None

        # resources[agent_name][resource_name] = current value
        self.resources: dict[str, dict[str, float]] = {
            agent.name: dict(agent.initial_resources)
            for agent in scenario.agents
        }

    # ── Tick progression ──────────────────────────────────────────────────────

    def advance_tick(self) -> None:
        """Apply per-tick resource decay (scarcity-scaled) and increment tick."""
        self.tick += 1
        for agent_resources in self.resources.values():
            for rd in self._resource_defs:
                decay = rd.decay_per_tick * self.scarcity
                agent_resources[rd.name] = max(0.0, agent_resources[rd.name] - decay)

    # ── Action resolution ─────────────────────────────────────────────────────

    def apply_action(self, actor: str, action: str, target: Optional[str]) -> None:
        """Transfer primary-resource units between actor and target based on action."""
        if self._primary is None:
            return
        effects = _ACTION_EFFECTS.get(action, {"actor": 0.0, "target": 0.0})

        self.resources[actor][self._primary] = max(
            0.0, self.resources[actor][self._primary] + effects["actor"]
        )
        if target and target in self.resources and effects["target"] != 0.0:
            self.resources[target][self._primary] = max(
                0.0, self.resources[target][self._primary] + effects["target"]
            )

    # ── Payoff preview (used by payoff.py to build the prompt matrix) ─────────

    def compute_payoff(self, actor: str, action: str, target: Optional[str]) -> dict:
        """
        Return the numeric payoffs that will be shown in the agent prompt.
        Does NOT mutate state — pure computation only.
        """
        effects = _ACTION_EFFECTS.get(action, {"actor": 0.0, "target": 0.0})
        return {
            "self_gain": effects["actor"],
            "target_gain": effects["target"] if target else 0.0,
        }

    # ── Read helpers ──────────────────────────────────────────────────────────

    def get_agent_resources(self, agent_name: str) -> dict[str, float]:
        return dict(self.resources[agent_name])

    def agent_names(self) -> list[str]:
        return list(self.resources.keys())

    def snapshot(self) -> dict:
        """Full state snapshot for logging."""
        return {
            "tick": self.tick,
            "scenario": self.scenario_name,
            "scarcity": self.scarcity,
            "resources": {k: dict(v) for k, v in self.resources.items()},
        }
