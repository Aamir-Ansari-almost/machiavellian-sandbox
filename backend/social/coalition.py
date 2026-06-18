from dataclasses import dataclass

from social.graph import SocialGraph


@dataclass(frozen=True)
class CoalitionEvent:
    """A change in alliance state, emitted by CoalitionTracker each tick."""

    agent_a: str
    agent_b: str
    event_type: str  # "formed" | "dissolved" | "betrayed"


def _pair(a: str, b: str) -> tuple[str, str]:
    """Canonical unordered key for an agent pair."""
    return (a, b) if a < b else (b, a)


class CoalitionTracker:
    """
    Derives alliances from the trust graph. An alliance is NOT a stored object —
    it is the emergent fact that two agents mutually trust each other above the
    formation threshold. The tracker remembers which pairs are currently allied so
    it can emit formed/dissolved transitions when thresholds are crossed.

        form:     both directions  >  form_threshold      → "formed"
        dissolve: either direction  <  dissolve_threshold  → "dissolved"

    Between the two thresholds an existing alliance persists (hysteresis), which
    prevents an alliance from flickering on and off around a single boundary.
    """

    def __init__(
        self, graph: SocialGraph, form_threshold: float, dissolve_threshold: float
    ) -> None:
        self._graph = graph
        self._form = form_threshold
        self._dissolve = dissolve_threshold
        self._active: set[tuple[str, str]] = set()

    # ── Queries ─────────────────────────────────────────────────────────────

    def are_allied(self, a: str, b: str) -> bool:
        return _pair(a, b) in self._active

    def allies_of(self, agent: str) -> set[str]:
        result: set[str] = set()
        for x, y in self._active:
            if x == agent:
                result.add(y)
            elif y == agent:
                result.add(x)
        return result

    def active_alliances(self) -> set[tuple[str, str]]:
        return set(self._active)

    # ── Transition detection ─────────────────────────────────────────────────

    def update(self) -> list[CoalitionEvent]:
        """
        Recompute alliance state from current trust and emit transition events.
        Call once per tick AFTER all trust deltas (including propagation) are applied.
        Betrayal events are emitted separately via record_betrayal().
        """
        events: list[CoalitionEvent] = []
        agents = self._graph.agents()

        for i, a in enumerate(agents):
            for b in agents[i + 1 :]:
                key = _pair(a, b)
                t_ab = self._graph.trust(a, b)
                t_ba = self._graph.trust(b, a)
                allied = key in self._active

                if not allied and t_ab > self._form and t_ba > self._form:
                    self._active.add(key)
                    events.append(CoalitionEvent(key[0], key[1], "formed"))
                elif allied and (t_ab < self._dissolve or t_ba < self._dissolve):
                    self._active.discard(key)
                    events.append(CoalitionEvent(key[0], key[1], "dissolved"))

        return events

    def record_betrayal(self, betrayer: str, victim: str) -> CoalitionEvent:
        """
        Mark an alliance as betrayed. Removes it from the active set so update()
        won't also emit a "dissolved" for the same pair. The tick engine calls this
        when a `betray` action resolves against a current ally.
        """
        self._active.discard(_pair(betrayer, victim))
        return CoalitionEvent(betrayer, victim, "betrayed")

    def witnesses(self, betrayer: str, victim: str) -> list[str]:
        """
        Depth-1 witnesses to a betrayal: agents currently allied with either the
        betrayer or the victim (excluding the two parties themselves).
        """
        seen = self.allies_of(betrayer) | self.allies_of(victim)
        seen.discard(betrayer)
        seen.discard(victim)
        return sorted(seen)
