from __future__ import annotations

import argparse
import json
from urllib.error import URLError

from .llm import OllamaClient
from .pipeline import AlarmRAGPipeline
from .rag import generate_rag_answer
from .types import AlarmFloodQuery


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TE Alarm RAG retrieval CLI")
    parser.add_argument("--data", required=True, help="Path to JSON/JSONL knowledge documents")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--alarms", default="", help="Comma-separated active alarm tags")
    parser.add_argument("--fault-hint", default=None, help="Optional fault hint")
    parser.add_argument("--region", default=None, help="Operating region filter/context")
    parser.add_argument("--time-scale", default=None, help="Time scale filter/context")
    parser.add_argument("--metadata-filter", default=None, help="JSON object for metadata filtering")
    parser.add_argument("--mode", choices=["retrieve", "rag"], default="retrieve", help="Run retrieval only or full RAG")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--ollama-model", default="llama3.1:8b", help="Ollama model name")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature for RAG mode")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    metadata_filter = json.loads(args.metadata_filter) if args.metadata_filter else None
    tags = [token.strip() for token in args.alarms.split(",") if token.strip()]

    process_state = {}
    if args.region:
        process_state["operating_region"] = args.region
    if args.time_scale:
        process_state["time_scale"] = args.time_scale

    query = AlarmFloodQuery(
        active_alarm_tags=tags,
        process_state=process_state,
        fault_hint=args.fault_hint,
        top_k=args.top_k,
    )

    pipeline = AlarmRAGPipeline.from_file(args.data)
    results = pipeline.retrieve(query=query, metadata_filter=metadata_filter)

    for idx, result in enumerate(results, start=1):
        doc = result.doc
        print(f"[{idx}] score={result.score:.4f} id={doc.doc_id} type={doc.doc_type} title={doc.title}")
        print(f"    fault_type={doc.fault_type} tags={','.join(doc.alarm_tags)}")
        print(f"    region={doc.operating_region} time_scale={doc.time_scale}")
        print(f"    text={doc.text[:180]}{'...' if len(doc.text) > 180 else ''}")

    if args.mode == "rag":
        client = OllamaClient(base_url=args.ollama_url, model=args.ollama_model)
        try:
            answer = generate_rag_answer(client=client, query=query, retrieved=results, temperature=args.temperature)
            print("\n===== RAG ANSWER =====")
            print(answer)
        except URLError as exc:
            print("\n===== RAG ERROR =====")
            print(
                "Could not reach Ollama at "
                f"{args.ollama_url}. Start Ollama and pull model {args.ollama_model}. "
                f"Details: {exc}"
            )


if __name__ == "__main__":
    main()
