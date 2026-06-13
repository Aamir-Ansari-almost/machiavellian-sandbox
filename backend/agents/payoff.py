from typing import Optional

from core.world_state import WorldState

# Trust deltas applied per action — actor's feeling toward target, and target's
# back toward actor. Shown in the prompt as relationship_change; the same table
# is applied by social/trust.py (Week 2). Victims always move more than actors.
TRUST_DELTAS: dict[str, dict[str, float]] = {
    "cooperate": {"actor_to_target": +0.10, "target_to_actor": +0.05},
    "defect": {"actor_to_target": -0.20, "target_to_actor": -0.30},
    "betray": {"actor_to_target": -0.50, "target_to_actor": -0.80},
    "negotiate": {"actor_to_target": +0.20, "target_to_actor": +0.20},
    "ignore": {"actor_to_target": 0.00, "target_to_actor": 0.00},
}


def build_payoff_block(
    world: WorldState,
    actor: str,
    others: list[str],
    allies: Optional[set[str]] = None,
) -> str:
    """
    Build the AVAILABLE ACTIONS AND PAYOFFS text block for the agent prompt.

    Payoff-matrix scenarios (Axelrod) show the full joint game matrix so the LLM
    can reason about the opponent. Transfer-based scenarios (senate) show per-action
    self_gain and relationship_change for each target.
    """
    allies = allies or set()
    if world.has_payoff_matrix:
        return _payoff_matrix_block(world, actor, others)
    return _transfer_block(world, actor, others, allies)


def _payoff_matrix_block(world: WorldState, actor: str, others: list[str]) -> str:
    lines = [
        "AVAILABLE ACTIONS: cooperate or defect (both players choose simultaneously)."
    ]
    for other in others:
        p = world.compute_payoff(actor, "cooperate", other)
        lines.append(f"\nAgainst {other} — points you gain depending on both choices:")
        lines.append(
            f"  - both cooperate:                  {_fmt(p['if_both_cooperate'])}"
        )
        lines.append(
            f"  - you defect, {other} cooperates:  {_fmt(p['if_you_defect_they_coop'])}"
        )
        lines.append(
            f"  - you cooperate, {other} defects:  {_fmt(p['if_you_coop_they_defect'])}"
        )
        lines.append(
            f"  - both defect:                     {_fmt(p['if_both_defect'])}"
        )
    return "\n".join(lines)


def _transfer_block(
    world: WorldState, actor: str, others: list[str], allies: set[str]
) -> str:
    lines = ["AVAILABLE ACTIONS AND PAYOFFS:"]
    for other in others:
        coop = world.compute_payoff(actor, "cooperate", other)
        defe = world.compute_payoff(actor, "defect", other)
        nego = world.compute_payoff(actor, "negotiate", other)

        lines.append(
            f"  - cooperate with {other}:  self_gain={_fmt(coop['self_gain'])}, "
            f"relationship_change={_fmt(TRUST_DELTAS['cooperate']['actor_to_target'])}"
        )
        lines.append(
            f"  - defect against {other}:  self_gain={_fmt(defe['self_gain'])}, "
            f"relationship_change={_fmt(TRUST_DELTAS['defect']['actor_to_target'])}"
        )
        if other in allies:
            betr = world.compute_payoff(actor, "betray", other)
            lines.append(
                f"  - betray {other}:          self_gain={_fmt(betr['self_gain'])}, "
                f"relationship_change={_fmt(TRUST_DELTAS['betray']['actor_to_target'])}, "
                f"visibility=high  [allies will notice]"
            )
        lines.append(
            f"  - negotiate with {other}:  self_gain={_fmt(nego['self_gain'])}, "
            f"relationship_change={_fmt(TRUST_DELTAS['negotiate']['actor_to_target'])}"
        )
    lines.append("  - ignore:                  self_gain=0, relationship_change=0")
    return "\n".join(lines)


def _fmt(x: float) -> str:
    return "0" if x == 0 else f"{x:+g}"
