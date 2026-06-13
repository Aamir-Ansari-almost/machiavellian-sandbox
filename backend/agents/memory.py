from typing import Optional

import chromadb
from chromadb.config import Settings


class AgentMemory:
    """
    Per-agent episodic memory backed by ChromaDB.

    Each memory is one event stored as the embedding of a natural-language
    sentence plus metadata. Retrieval ranks candidates by:

        score = cosine_similarity * salience * recency_decay

    so a betrayal (high salience) stays retrievable far longer than idle chat
    (low salience), which is exactly what hypothesis H3 tests.
    """

    # Deterministic salience per action type — set by the engine, never the LLM.
    SALIENCE: dict[str, float] = {
        "betray": 0.9,
        "defect": 0.6,
        "negotiate": 0.5,
        "cooperate": 0.2,
        "ignore": 0.05,
    }
    DEFAULT_SALIENCE = 0.1

    def __init__(
        self,
        agent_name: str,
        run_id: str,
        client: Optional["chromadb.api.ClientAPI"] = None,
        recency_halflife: float = 10.0,
    ) -> None:
        self.agent_name = agent_name
        self.recency_halflife = recency_halflife
        self._client = client or chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False)
        )
        # One collection per agent per run; cosine space so distance = 1 - cos_sim.
        name = f"mem_{run_id}_{agent_name}".replace(" ", "_").lower()
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        self._counter = 0

    # ── Write ──────────────────────────────────────────────────────────────────

    def write(
        self,
        text: str,
        tick: int,
        action: str,
        tags: Optional[list[str]] = None,
    ) -> None:
        """Store one event. Salience is derived from the action type."""
        salience = self.SALIENCE.get(action, self.DEFAULT_SALIENCE)
        self._counter += 1
        self._collection.add(
            documents=[text],
            metadatas=[
                {
                    "tick": int(tick),
                    "salience": float(salience),
                    "action": action,
                    "tags": ",".join(tags or []),
                }
            ],
            ids=[f"{self.agent_name}_{self._counter}"],
        )

    # ── Retrieve ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, current_tick: int, k: int = 5) -> list[dict]:
        """
        Return the top-k memories for the query, re-ranked by
        similarity * salience * recency. Empty list if no memories yet.
        """
        count = self._collection.count()
        if count == 0:
            return []

        n_candidates = min(count, max(k * 3, 15))
        res = self._collection.query(query_texts=[query], n_results=n_candidates)

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        scored: list[dict] = []
        for doc, meta, dist in zip(docs, metas, dists):
            similarity = max(0.0, 1.0 - dist)  # cosine distance → similarity, clamped
            age = current_tick - meta["tick"]
            recency = (
                0.5 ** (age / self.recency_halflife)
                if self.recency_halflife > 0
                else 1.0
            )
            score = similarity * meta["salience"] * recency
            scored.append(
                {
                    "text": doc,
                    "tick": meta["tick"],
                    "salience": meta["salience"],
                    "similarity": round(similarity, 3),
                    "recency": round(recency, 3),
                    "score": round(score, 4),
                }
            )

        scored.sort(key=lambda m: m["score"], reverse=True)
        return scored[:k]

    # ── Query template ──────────────────────────────────────────────────────

    @staticmethod
    def build_query(target: Optional[str], situation: str) -> str:
        """
        Template-based retrieval query. Free-form queries are unpredictable;
        this keeps retrieval meaningful and reproducible.
        """
        if target:
            return f"What do I know about {target} and {situation}?"
        return f"What do I know about {situation}?"

    def count(self) -> int:
        return self._collection.count()
