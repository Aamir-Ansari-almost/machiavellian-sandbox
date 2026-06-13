from typing import Optional

from core.scenario_loader import AgentConfig
from core.world_state import WorldState
from agents.payoff import build_payoff_block


def build_system_prompt(config: AgentConfig) -> str:
    """
    Assembled once at agent init. Carries the agent's identity, persona,
    hidden agenda, and free-form extra traits, plus the output contract.
    """
    lines = [f"You are {config.name}."]
    lines.append(f"\nPERSONA: {config.persona}")
    lines.append(
        f"\nHIDDEN AGENDA (this is private — never state it directly in speech, "
        f"but let it drive every decision): {config.hidden_agenda}"
    )

    if config.extra:
        lines.append("\nADDITIONAL TRAITS:")
        for key, value in config.extra.items():
            lines.append(f"- {key}: {value}")

    lines.append(
        "\nEvery turn you must respond with valid JSON only, exactly this schema:\n"
        '{"action": "<one of the available actions>", '
        '"target": "<another agent\'s name, or null>", '
        '"speech": "<what you say aloud — others may hear it>", '
        '"reasoning": "<your private reasoning>"}\n'
        "Choose only from the actions listed in the turn prompt. "
        "Stay in character. Pursue your hidden agenda without revealing it."
    )
    return "\n".join(lines)


def build_user_prompt(
    agent_name: str,
    world: WorldState,
    others: list[str],
    trust_view: dict[str, float],
    allies: set[str],
    memories: list[dict],
    recent: Optional[list[dict]] = None,
) -> str:
    """
    Assembled fresh each tick. Shows the agent its situation: resources, world
    state, alliances, trust levels, the payoff matrix, and two memory channels:

      - `recent`:   working memory — what happened in the last few ticks,
                    unconditionally, so the agent always has fresh context.
      - `memories`: long-term recall — salience-weighted semantic retrieval of
                    older events relevant to this tick's situation.
    """
    res = world.get_agent_resources(agent_name)
    res_str = ", ".join(f"{k}={v:.0f}" for k, v in res.items())

    lines = [f"TICK {world.tick} | Your resources: {res_str}"]

    if world.scenario_extra:
        lines.append("\nWORLD:")
        for key, value in world.scenario_extra.items():
            lines.append(f"  {key}: {str(value).strip()}")

    lines.append(f"\nSCARCITY LEVEL: {world.scarcity}")

    if allies:
        lines.append(f"YOUR CURRENT ALLIANCES: {', '.join(sorted(allies))}")
    else:
        lines.append("YOUR CURRENT ALLIANCES: none")

    trust_str = ", ".join(f"{o}={trust_view.get(o, 0.0):+.2f}" for o in others)
    lines.append(f"TRUST LEVELS (your trust toward each): {trust_str}")

    lines.append("")
    lines.append(build_payoff_block(world, agent_name, others, allies))

    recent = recent or []
    recent_texts = {m["text"] for m in recent}

    lines.append("")
    if recent:
        lines.append("WHAT JUST HAPPENED (most recent ticks):")
        for m in recent:
            lines.append(f"  - [tick {m['tick']}] {m['text']}")
    else:
        lines.append("WHAT JUST HAPPENED: (nothing yet)")

    # Long-term recall — exclude anything already shown in the recent buffer.
    salient = [m for m in memories if m["text"] not in recent_texts]

    lines.append("")
    if salient:
        lines.append("RELEVANT PAST EVENTS:")
        for m in salient:
            lines.append(f"  - {m['text']} (salience {m['salience']})")
    else:
        lines.append("RELEVANT PAST EVENTS: (none)")

    lines.append("\nWhat do you do this tick? Respond with JSON only.")
    return "\n".join(lines)
