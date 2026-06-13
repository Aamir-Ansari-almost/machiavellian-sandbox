from typing import Optional

from core.scenario_loader import PayoffMatrix, ScenarioConfig, ResourceDef

# Transfer-based effects for multi-action scenarios (senate, market).
# actor = what the acting agent gains/loses on the primary resource.
# target = what the target gains/loses.
_ACTION_EFFECTS: dict[str, dict[str, float]] = {
    "cooperate": {"actor": -2.0, "target": +2.0},
    "defect": {"actor": +5.0, "target": -5.0},
    "betray": {"actor": +8.0, "target": -8.0},
    "negotiate": {"actor": 0.0, "target": 0.0},
    "ignore": {"actor": 0.0, "target": 0.0},
}


class WorldState:
    def __init__(self, scenario: ScenarioConfig) -> None:
        self.tick: int = 0
        self.scarcity: float = scenario.scarcity
        self.scenario_name: str = scenario.name
        self.scenario_extra: dict = scenario.extra
        self._resource_defs: list[ResourceDef] = scenario.resources
        self._primary: str | None = (
            scenario.resources[0].name if scenario.resources else None
        )
        self._payoff_matrix: PayoffMatrix | None = scenario.payoff_matrix

        # resources[agent_name][resource_name] = current value
        self.resources: dict[str, dict[str, float]] = {
            agent.name: dict(agent.initial_resources) for agent in scenario.agents
        }

    @property
    def has_payoff_matrix(self) -> bool:
        return self._payoff_matrix is not None

    # ── Tick progression ──────────────────────────────────────────────────────

    def advance_tick(self) -> None:
        """Apply per-tick resource decay (scarcity-scaled) and increment tick."""
        self.tick += 1
        for agent_resources in self.resources.values():
            for rd in self._resource_defs:
                decay = rd.decay_per_tick * self.scarcity
                agent_resources[rd.name] = max(0.0, agent_resources[rd.name] - decay)

    # ── Action resolution ─────────────────────────────────────────────────────
    # TODO: understand this
    def apply_action(self, actor: str, action: str, target: Optional[str]) -> None:
        """
        Transfer-based resolution for multi-action scenarios (senate, market).
        Each action is resolved independently — do NOT use when payoff_matrix is set.
        """
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

    def resolve_joint(
        self,
        actor: str,
        actor_action: str,
        target: str,
        target_action: str,
    ) -> None:
        """
        Joint resolution for payoff-matrix scenarios (Axelrod).
        Both agents' actions must be known before points are awarded.
        Looks up the correct cell in the payoff matrix and applies it.
        """
        if self._primary is None or self._payoff_matrix is None:
            raise RuntimeError(
                "resolve_joint requires a payoff_matrix in the scenario YAML."
            )

        pm = self._payoff_matrix
        actor_cooperates = actor_action == "cooperate"
        target_cooperates = target_action == "cooperate"

        if actor_cooperates and target_cooperates:
            cell = pm.both_cooperate
        elif not actor_cooperates and target_cooperates:
            cell = pm.actor_defects
        elif actor_cooperates and not target_cooperates:
            cell = pm.target_defects
        else:
            cell = pm.both_defect

        self.resources[actor][self._primary] = max(
            0.0, self.resources[actor][self._primary] + cell["actor"]
        )
        self.resources[target][self._primary] = max(
            0.0, self.resources[target][self._primary] + cell["target"]
        )

    # ── Payoff preview (used by payoff.py to build the prompt matrix) ─────────

    def compute_payoff(self, actor: str, action: str, target: Optional[str]) -> dict:
        """
        Return payoff numbers shown in the agent prompt. Pure — does not mutate state.

        For transfer-based scenarios: returns self_gain and target_gain for that action.
        For payoff-matrix scenarios: returns all four PD outcomes so the LLM sees the
        full game matrix and can reason about what the target might do.
        """
        if self._payoff_matrix is not None:
            pm = self._payoff_matrix
            return {
                "if_both_cooperate": pm.both_cooperate["actor"],
                "if_you_defect_they_coop": pm.actor_defects["actor"],
                "if_you_coop_they_defect": pm.target_defects["actor"],
                "if_both_defect": pm.both_defect["actor"],
            }

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
