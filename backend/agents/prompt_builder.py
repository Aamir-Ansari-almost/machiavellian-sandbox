from typing import Optional

from core.scenario_loader import AgentConfig
from core.world_state import WorldState
from agents.payoff import build_payoff_block

# Keys in scenario.extra that are engine control flags, NOT world flavor — they
# must not be rendered into the WORLD block where agents would read them literally.
_CONTROL_KEYS = {"communication", "horizon_disclosure"}


def _horizon_disclosure(world: WorldState) -> Optional[str]:
    """
    Time-horizon framing, controlled by scenario.extra['horizon_disclosure']:

      "known"  -> tell the agent exactly how many rounds remain (enables backward
                  induction and end-game defection).
      "hidden" -> explicitly state the end is unknown (Folk Theorem conditions).
      absent   -> say nothing (current default; agent sees only the tick number).
    """
    mode = str(world.scenario_extra.get("horizon_disclosure", "")).strip().lower()
    if mode == "known":
        remaining = max(0, world.total_ticks - world.tick)
        return (
            f"TIME: This game lasts exactly {world.total_ticks} rounds. "
            f"This is round {world.tick} of {world.total_ticks} — {remaining} round(s) remain."
        )
    if mode == "hidden":
        return (
            "TIME: The game may end at any round without warning. "
            "You do not know how many rounds remain."
        )
    return None


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
    overheard: Optional[dict[str, str]] = None,
) -> str:
    """
    Assembled fresh each tick. Shows the agent its situation: resources, world
    state, alliances, trust levels, the payoff matrix, and two memory channels:

      - `recent`:   working memory — what happened in the last few ticks,
                    unconditionally, so the agent always has fresh context.
      - `memories`: long-term recall — salience-weighted semantic retrieval of
                    older events relevant to this tick's situation.

    `overheard` is the cheap-talk channel: what each other agent SAID aloud last
    tick. Only populated when the scenario enables communication — otherwise the
    agents are deaf to each other's speech and react to actions alone.
    """
    res = world.get_agent_resources(agent_name)
    res_str = ", ".join(f"{k}={v:.0f}" for k, v in res.items())

    lines = [f"TICK {world.tick} | Your resources: {res_str}"]

    horizon_line = _horizon_disclosure(world)
    if horizon_line:
        lines.append(horizon_line)

    # WORLD flavor — skip control flags, they are engine config, not world state.
    flavor = {k: v for k, v in world.scenario_extra.items() if k not in _CONTROL_KEYS}
    if flavor:
        lines.append("\nWORLD:")
        for key, value in flavor.items():
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

    # Cheap-talk channel — what others said aloud last tick (communication on).
    overheard = overheard or {}
    if overheard:
        lines.append("")
        lines.append(
            "WHAT OTHERS SAID TO YOU LAST TURN (they may be sincere, or lying):"
        )
        for speaker, said in overheard.items():
            lines.append(f'  - {speaker} said: "{said}"')

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
