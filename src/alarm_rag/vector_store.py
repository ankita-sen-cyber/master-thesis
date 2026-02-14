from __future__ import annotations

import json
from dataclasses import dataclass, field

from .types import KnowledgeDoc


def _dot(lhs: list[float], rhs: list[float]) -> float:
    return sum(a * b for a, b in zip(lhs, rhs))


def _matches_filter(doc: KnowledgeDoc, metadata_filter: dict[str, object] | None) -> bool:
    if not metadata_filter:
        return True
    for key, value in metadata_filter.items():
        doc_value = getattr(doc, key, doc.metadata.get(key))
        if value is None:
            continue
        if isinstance(doc_value, list):
            if isinstance(value, list):
                if not set(value).intersection(doc_value):
                    return False
            elif value not in doc_value:
                return False
        elif doc_value != value:
            return False
    return True


@dataclass(slots=True)
class SearchResult:
    doc: KnowledgeDoc
    score: float


@dataclass(slots=True)
class InMemoryVectorStore:
    docs: list[KnowledgeDoc] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)

    def add(self, docs: list[KnowledgeDoc], vectors: list[list[float]]) -> None:
        if len(docs) != len(vectors):
            raise ValueError("docs and vectors must have equal length")
        self.docs.extend(docs)
        self.vectors.extend(vectors)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        metadata_filter: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        scored: list[SearchResult] = []
        for doc, vec in zip(self.docs, self.vectors):
            if not _matches_filter(doc, metadata_filter):
                continue
            scored.append(SearchResult(doc=doc, score=_dot(query_vector, vec)))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def save_json(self, path: str) -> None:
        payload = [
            {
                "doc": doc.to_dict(),
                "vector": vector,
            }
            for doc, vector in zip(self.docs, self.vectors)
        ]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "InMemoryVectorStore":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        docs = [KnowledgeDoc.from_dict(item["doc"]) for item in payload]
        vectors = [list(item["vector"]) for item in payload]
        return cls(docs=docs, vectors=vectors)
