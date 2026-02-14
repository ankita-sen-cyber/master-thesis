from __future__ import annotations

import json
from dataclasses import dataclass

from .embeddings import HashingEmbedder
from .types import AlarmFloodQuery, KnowledgeDoc
from .vector_store import InMemoryVectorStore, SearchResult


def _load_docs(path: str) -> list[KnowledgeDoc]:
    if path.endswith(".jsonl"):
        rows: list[dict] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
    return [KnowledgeDoc.from_dict(item) for item in rows]


def _build_query_text(query: AlarmFloodQuery) -> str:
    components = [
        "alarm flood diagnosis",
        " ".join(query.active_alarm_tags),
    ]
    for key, value in query.process_state.items():
        components.append(f"{key}:{value}")
    if query.fault_hint:
        components.append(f"fault_hint:{query.fault_hint}")
    return " ".join(components)


@dataclass(slots=True)
class AlarmRAGPipeline:
    embedder: HashingEmbedder
    store: InMemoryVectorStore

    @classmethod
    def from_documents(cls, docs: list[KnowledgeDoc], embedding_dim: int = 384) -> "AlarmRAGPipeline":
        embedder = HashingEmbedder(dim=embedding_dim)
        vectors = embedder.encode([_doc_to_embedding_text(doc) for doc in docs])
        store = InMemoryVectorStore()
        store.add(docs, vectors)
        return cls(embedder=embedder, store=store)

    @classmethod
    def from_file(cls, path: str, embedding_dim: int = 384) -> "AlarmRAGPipeline":
        docs = _load_docs(path)
        return cls.from_documents(docs, embedding_dim=embedding_dim)

    def retrieve(
        self,
        query: AlarmFloodQuery,
        metadata_filter: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        query_text = _build_query_text(query)
        query_vec = self.embedder.encode([query_text])[0]
        base = self.store.search(query_vec, top_k=max(query.top_k * 3, query.top_k), metadata_filter=metadata_filter)
        reranked = [
            SearchResult(doc=item.doc, score=_rerank_score(item, query))
            for item in base
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[: query.top_k]


def _doc_to_embedding_text(doc: KnowledgeDoc) -> str:
    tags = " ".join(doc.alarm_tags)
    return " ".join(
        [
            doc.doc_type,
            doc.title,
            doc.text,
            doc.fault_type or "",
            doc.operating_region or "",
            doc.time_scale or "",
            tags,
        ]
    )


def _rerank_score(result: SearchResult, query: AlarmFloodQuery) -> float:
    score = result.score
    doc = result.doc
    query_tags = set(query.active_alarm_tags)
    if query_tags and doc.alarm_tags:
        overlap = len(query_tags.intersection(doc.alarm_tags))
        score += 0.15 * (overlap / max(len(query_tags), 1))
    query_time_scale = query.process_state.get("time_scale")
    if query_time_scale and doc.time_scale == query_time_scale:
        score += 0.1
    query_region = query.process_state.get("operating_region")
    if query_region and doc.operating_region == query_region:
        score += 0.1
    if query.fault_hint and doc.fault_type and query.fault_hint.lower() in doc.fault_type.lower():
        score += 0.2
    return score
