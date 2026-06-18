from social.graph import SocialGraph

# Trust deltas applied per action. actor_to_target = how the acting agent's trust
# toward the target shifts; target_to_actor = how the target's trust back shifts.
# Victims always move more than actors — people update their own feelings faster
# than others update toward them. This table is the single source of truth: it is
# both shown in the prompt (agents/payoff.py imports it) and applied here.
TRUST_DELTAS: dict[str, dict[str, float]] = {
    "cooperate": {"actor_to_target": +0.10, "target_to_actor": +0.05},
    "defect": {"actor_to_target": -0.20, "target_to_actor": -0.30},
    "betray": {"actor_to_target": -0.50, "target_to_actor": -0.80},
    "negotiate": {"actor_to_target": +0.20, "target_to_actor": +0.20},
    "ignore": {"actor_to_target": 0.00, "target_to_actor": 0.00},
}

# Witnesses to a betrayal update their trust toward the betrayer by this fraction
# of the betrayal's salience. Depth-1 only — no cascading — to avoid loops.
BETRAYAL_PROPAGATION_FACTOR = 0.3
BETRAYAL_SALIENCE = 0.9  # mirrors AgentMemory.SALIENCE["betray"]


def apply_action(
    graph: SocialGraph,
    actor: str,
    action: str,
    target: str | None,
) -> None:
    """
    Apply the trust deltas for one resolved action to the social graph.
    Betrayal propagation is handled separately by propagate_betrayal(), which the
    tick engine calls only after confirming a betrayal actually fired.
    """
    if target is None or action not in TRUST_DELTAS:
        return
    deltas = TRUST_DELTAS[action]
    graph.adjust(actor, target, deltas["actor_to_target"])
    graph.adjust(target, actor, deltas["target_to_actor"])


def propagate_betrayal(
    graph: SocialGraph,
    betrayer: str,
    victim: str,
    witnesses: list[str],
) -> dict[str, float]:
    """
    A betrayal between `betrayer` and `victim` is observed by `witnesses`
    (depth-1 allies of either party). Each witness's trust toward the betrayer
    drops by BETRAYAL_PROPAGATION_FACTOR * BETRAYAL_SALIENCE.

    Returns the applied delta per witness, for logging. Single depth — witnesses
    do not themselves trigger further propagation.
    """
    shock = BETRAYAL_PROPAGATION_FACTOR * BETRAYAL_SALIENCE
    applied: dict[str, float] = {}
    for w in witnesses:
        if w in (betrayer, victim):
            continue
        graph.adjust(w, betrayer, -shock)
        applied[w] = -shock
    return applied
