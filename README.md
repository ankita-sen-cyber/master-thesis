# master-thesis

Alarm flood management system.

## Environment Setup

Use a local virtual environment so installs are not global:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you only run retrieval/RAG and do not ingest `.RData`, you can skip `pyreadr`:

```bash
pip install -r requirements-min.txt
```

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
  --ollama-model gemma3:4b
```

## Build KB From RData (Faulty + Fault-Free)

If your cleaned TE training data is in RData files:

```bash
pip install -r requirements.txt
PYTHONPATH=src python3 -m alarm_rag.build_kb_from_rdata \
  --faulty-rdata data/dataverse_files/TEP_Faulty_Training.RData \
  --faultfree-rdata data/dataverse_files/TEP_FaultFree_Training.RData \
  --output data/te_knowledge_base.generated.jsonl
```

Then query the generated KB:

```bash
PYTHONPATH=src python3 -m alarm_rag.cli \
  --mode retrieve \
  --data data/te_knowledge_base.generated.jsonl \
  --alarms AH_P_REACTOR_HIGH \
  --top-k 5
```
