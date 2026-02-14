# TE Alarm Knowledge Base and RAG Retrieval Specification

## 1. Content Scope

### 1.1 Alarm Philosophy for TE

Define and maintain:

- Alarm tag definitions.
- Setpoints and alarm limits.
- Expected operator responses.
- Priority levels and escalation logic.

### 1.2 TE Fault Library

Maintain a library of 20 standard Tennessee Eastman (TE) faults (original and revised simulators), with:

- Fault description.
- Typical symptoms in process variables.
- Alarm signature(s) and sequence patterns.
- Root cause and affected process units.

### 1.3 Cause-Effect Matrices

Build matrices mapping:

- Fault -> triggered alarm tags.
- Fault -> temporal alarm sequence.
- Fault interactions and cross-coupling effects.

### 1.4 Synthetic Incident Reports

Curate synthetic or literature-based incident narratives combining:

- Normal operation baseline.
- Fault injection and time of occurrence.
- Alarm flood evolution.
- Operator actions/interventions.
- Final outcomes and lessons learned.

### 1.5 PtX Context (If Applicable)

Include offshore PtX operational context where relevant:

- Standard operating procedures.
- Common disturbances and upset conditions.
- Safety constraints tied to alarm response.

## 2. Indexing and Retrieval Architecture

### 2.1 Embeddings

Embed the following artifacts:

- Alarm descriptions.
- Fault signatures.
- Operating procedures and response guidance.

Model guidance:

- Prefer a domain-tuned embedding model.
- Alternatively, fine-tune a strong general embedding model on TE/PtX process text.

### 2.2 Vector Index

Store embeddings in a vector database (e.g., FAISS, Weaviate, Pinecone) with metadata fields:

- `fault_type`
- `alarm_tags` (multi-valued)
- `operating_region` (e.g., normal, startup, upset, shutdown)
- `time_scale` (seconds/minutes/hours)
- Optional: `simulator_version`, `unit_area`, `priority_level`

### 2.3 RAG Query Logic

Given current alarm flood context:

- Active alarm tag set.
- Alarm timing/sequence information.
- Current process state.

Retrieve top-K relevant documents using:

- Primary ranking: cosine similarity in embedding space.
- Secondary constraints: metadata filtering and/or weighted reranking.

Suggested retrieval pipeline:

1. Build query object from live alarm flood context.
2. Apply metadata pre-filter (if known constraints exist).
3. Execute vector similarity search.
4. Rerank candidates using temporal consistency and alarm-tag overlap.
5. Return top-K context documents for diagnosis and response recommendation.

## 3. Minimum Deliverables

- TE alarm philosophy dataset.
- TE 20-fault structured library.
- Cause-effect mapping tables/matrices.
- Annotated incident scenario set.
- Vector index with searchable metadata.
- RAG retrieval module accepting alarm flood state as input.
