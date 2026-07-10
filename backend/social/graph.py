from typing import Iterable

import networkx as nx

TRUST_MIN = -1.0
TRUST_MAX = 1.0


class SocialGraph:
    """
    Directed trust graph over agents.

    Edge A→B carries a single float `trust` in [-1, 1] — how much A trusts B.
    A→B and B→A are independent: A may trust B more than B trusts A.

    This is the queryable "who trusts who right now" structure. Alliances are
    NOT stored here as a separate object — a coalition is an emergent property
    of the trust edges (see social/coalition.py), which keeps the graph the
    single source of truth.
    """

    def __init__(self, agents: Iterable[str]) -> None:
        self._g = nx.DiGraph()
        names = list(agents)
        self._g.add_nodes_from(names)
        # Every ordered pair starts at neutral trust 0.0.
        for a in names:
            for b in names:
                if a != b:
                    self._g.add_edge(a, b, trust=0.0)

    # ── Read ──────────────────────────────────────────────────────────────────

    def trust(self, a: str, b: str) -> float:
        """How much a trusts b (directed)."""
        return self._g[a][b]["trust"]

    def trust_view(self, a: str) -> dict[str, float]:
        """a's trust toward every other agent — used to build a's prompt."""
        return {b: self._g[a][b]["trust"] for b in self._g.successors(a)}

    def agents(self) -> list[str]:
        return list(self._g.nodes)

    # ── Write ──────────────────────────────────────────────────────────────────

    def adjust(self, a: str, b: str, delta: float) -> float:
        """Add delta to trust[a→b], clamped to [-1, 1]. Returns the new value."""
        if a == b or not self._g.has_edge(a, b):
            return 0.0
        new = max(TRUST_MIN, min(TRUST_MAX, self._g[a][b]["trust"] + delta))
        self._g[a][b]["trust"] = new
        return new

    def set_trust(self, a: str, b: str, value: float) -> None:
        if self._g.has_edge(a, b):
            self._g[a][b]["trust"] = max(TRUST_MIN, min(TRUST_MAX, value))

    # ── Metrics / introspection (used by analysis/metrics.py) ─────────────────

    def all_edges(self) -> list[tuple[str, str, float]]:
        """Every directed trust edge as (a, b, trust)."""
        return [(a, b, d["trust"]) for a, b, d in self._g.edges(data=True)]

    def mutual_trust_graph(self, threshold: float) -> nx.Graph:
        """
        Undirected graph where an edge exists iff BOTH directions exceed
        `threshold`. This is the structure coalitions and clustering are read from.
        """
        ug = nx.Graph()
        ug.add_nodes_from(self._g.nodes)
        for a, b in self._g.edges:
            if a < b:  # consider each unordered pair once
                if (
                    self._g[a][b]["trust"] > threshold
                    and self._g[b][a]["trust"] > threshold
                ):
                    ug.add_edge(a, b)
        return ug

    def clustering_coefficient(self, threshold: float) -> float:
        """Average clustering of the mutual-trust graph — coalition density."""
        return nx.average_clustering(self.mutual_trust_graph(threshold))
