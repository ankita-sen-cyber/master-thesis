# master-thesis

Alarm flood management system.

## Specifications

- `/Users/ankitasen/Desktop/MASTER THESIS/alarm-flood/master-thesis/docs/te_alarm_rag_spec.md`: TE alarm/fault knowledge base and retrieval design requirements.

## TE Alarm RAG Pipeline

Implemented under:

- `/Users/ankitasen/Desktop/MASTER THESIS/alarm-flood/master-thesis/src/alarm_rag`

Starter dataset:

- `/Users/ankitasen/Desktop/MASTER THESIS/alarm-flood/master-thesis/data/te_knowledge_base.jsonl`

### Run Query

```bash
PYTHONPATH=src python3 -m alarm_rag.cli \
  --data data/te_knowledge_base.jsonl \
  --alarms AH_P_REACTOR_HIGH,AH_COMP_DISCH_PRESS \
  --region upset \
  --time-scale seconds \
  --top-k 3
```

### Metadata Filtering Example

```bash
PYTHONPATH=src python3 -m alarm_rag.cli \
  --data data/te_knowledge_base.jsonl \
  --alarms AH_P_REACTOR_HIGH \
  --metadata-filter '{"doc_type":"fault_library","simulator_version":"revised"}'
```

## Ollama Pod + Full RAG

Kubernetes manifests:

- `/Users/ankitasen/Desktop/MASTER THESIS/alarm-flood/master-thesis/deploy/k8s/ollama-pod.yaml`
- `/Users/ankitasen/Desktop/MASTER THESIS/alarm-flood/master-thesis/deploy/k8s/ollama-service.yaml`

Deploy Ollama:

```bash
kubectl apply -f deploy/k8s/ollama-pod.yaml
kubectl apply -f deploy/k8s/ollama-service.yaml
kubectl port-forward svc/ollama 11434:11434
```

Pull a model into Ollama:

```bash
curl http://localhost:11434/api/pull -d '{"name":"llama3.1:8b"}'
```

Run full RAG (retrieve + generate):

```bash
PYTHONPATH=src python3 -m alarm_rag.cli \
  --mode rag \
  --data data/te_knowledge_base.jsonl \
  --alarms AH_P_REACTOR_HIGH,AH_COMP_DISCH_PRESS \
  --region upset \
  --time-scale seconds \
  --top-k 4 \
  --ollama-url http://localhost:11434 \
  --ollama-model llama3.1:8b
```
