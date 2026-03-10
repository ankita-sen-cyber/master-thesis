from __future__ import annotations

from .llm import OllamaClient
from .types import AlarmFloodQuery
from .vector_store import SearchResult


def build_rag_prompt(
    query: AlarmFloodQuery,
    retrieved: list[SearchResult],
    question: str | None = None,
    max_doc_text_chars: int = 1200,
) -> str:
    lines: list[str] = []
    lines.append("You are an industrial alarm-flood diagnosis assistant for the Tennessee Eastman process.")
    lines.append("Use only the retrieved context. If uncertain, say explicitly what is uncertain.")
    lines.append("")
    lines.append("Current alarm flood context:")
    lines.append(f"- Active alarm tags: {', '.join(query.active_alarm_tags) if query.active_alarm_tags else 'none'}")
    lines.append(f"- Process state: {query.process_state if query.process_state else '{}'}")
    lines.append(f"- Fault hint: {query.fault_hint or 'none'}")
    lines.append(f"- User question: {question or 'Provide diagnosis and operator guidance.'}")
    lines.append("")
    lines.append("Retrieved documents:")

    for idx, item in enumerate(retrieved, start=1):
        doc = item.doc
        text = doc.text
        if len(text) > max_doc_text_chars:
            text = f"{text[:max_doc_text_chars]}..."
        lines.append(
            f"[{idx}] id={doc.doc_id} score={item.score:.4f} type={doc.doc_type} "
            f"fault_type={doc.fault_type} alarms={doc.alarm_tags} "
            f"region={doc.operating_region} time_scale={doc.time_scale}"
        )
        lines.append(f"[{idx}] text: {text}")

    lines.append("")
    lines.append("Return output in this exact structure:")
    lines.append("1) Most likely fault(s): short ranked list with confidence 0-1")
    lines.append("2) Alarm-sequence rationale: why these alarms and timing fit")
    lines.append("3) Immediate operator actions: max 5 actions, safety-first")
    lines.append("4) Evidence used: list document IDs used in reasoning")
    lines.append("5) Uncertainty and missing signals: what extra measurements would disambiguate")
    return "\n".join(lines)


def generate_rag_answer(
    client: OllamaClient,
    query: AlarmFloodQuery,
    retrieved: list[SearchResult],
    question: str | None = None,
    temperature: float = 0.1,
    max_doc_text_chars: int = 1200,
) -> str:
    prompt = build_rag_prompt(
        query=query,
        retrieved=retrieved,
        question=question,
        max_doc_text_chars=max_doc_text_chars,
    )
    return client.generate(prompt=prompt, temperature=temperature)
