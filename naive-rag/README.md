# Naive RAG — local & zero-cost (Phase 1, Track B)

Offline: Ingest → Chunk → Embed → Store. Online: Query → Retrieve → Generate. No reranker, no hybrid search yet — that's Phase 2, and you'll understand why after this fails.

## Setup

```bash
ollama pull nomic-embed-text
ollama pull llama4:scout   # if unavailable, set LLM_MODEL="llama3.2" in config.py
pip install -r requirements.txt
```

## Run

```bash
python ingest.py
python query.py "How does the Jira cursor-ai automation work?"
```

### LangGraph on Windows

From the `naive-rag` folder in Command Prompt, use the project's Python environment. If you already have `.venv`, skip the first command.

```cmd
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe graph.py "What is the difference between Startup delivery ops and affiliate flow?"
```

## Corpus

`docs/` has 10 short docs from Med's real work — Startup, Jira → Cursor SDK, OpenClaw, and the 5 GenAI demos. Replace them with your real PDFs/Markdown later; the pipeline doesn't care.

Chunking is in `ingest.py`: 300 words, 50 overlap (~400 tokens, 10–15%). Retrieval is plain cosine similarity, top-5, in-memory `store.json`.

## Your exercise

Run the 10 questions in `questions.md`. For each, log: top score, right doc retrieved? answer faithful? Keep at least 3 failures — wrong chunk, missing context, low score. That list is the brief for Phase 2 (BM25 + reranker).
