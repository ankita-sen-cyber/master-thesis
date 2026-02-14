from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class KnowledgeDoc:
    doc_id: str
    doc_type: str
    title: str
    text: str
    fault_type: str | None = None
    alarm_tags: list[str] = field(default_factory=list)
    operating_region: str | None = None
    time_scale: str | None = None
    simulator_version: str | None = None
    unit_area: str | None = None
    priority_level: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KnowledgeDoc":
        reserved = {
            "doc_id",
            "id",
            "doc_type",
            "type",
            "title",
            "text",
            "fault_type",
            "alarm_tags",
            "operating_region",
            "time_scale",
            "simulator_version",
            "unit_area",
            "priority_level",
        }
        doc_id = str(payload.get("doc_id") or payload.get("id"))
        return cls(
            doc_id=doc_id,
            doc_type=str(payload.get("doc_type") or payload.get("type") or "unknown"),
            title=str(payload.get("title") or doc_id),
            text=str(payload.get("text") or ""),
            fault_type=payload.get("fault_type"),
            alarm_tags=[str(tag) for tag in payload.get("alarm_tags", [])],
            operating_region=payload.get("operating_region"),
            time_scale=payload.get("time_scale"),
            simulator_version=payload.get("simulator_version"),
            unit_area=payload.get("unit_area"),
            priority_level=payload.get("priority_level"),
            metadata={k: v for k, v in payload.items() if k not in reserved},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "title": self.title,
            "text": self.text,
            "fault_type": self.fault_type,
            "alarm_tags": self.alarm_tags,
            "operating_region": self.operating_region,
            "time_scale": self.time_scale,
            "simulator_version": self.simulator_version,
            "unit_area": self.unit_area,
            "priority_level": self.priority_level,
            **self.metadata,
        }


@dataclass(slots=True)
class AlarmFloodQuery:
    active_alarm_tags: list[str]
    process_state: dict[str, Any] = field(default_factory=dict)
    fault_hint: str | None = None
    top_k: int = 5
