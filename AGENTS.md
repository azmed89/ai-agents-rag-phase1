# AGENTS.md — ai-agents-rag-phase1

## Project
AI Agents + RAG, built from scratch. Two tracks:
- `react-agent/` — ReAct loop, Ollama local, then rebuilt with OpenAI Agents SDK / Codex SDK
- `naive-rag/` — naive RAG, in-memory cosine, chunk 300 words / 50 overlap, top-k 5

Stack: Python 3.12, Ollama (llama3.1:8b, nomic-embed-text), Codex (OpenAI).

## How to run
```bash
# Track A
cd react-agent
python agent.py "List the files in ./demo and summarise what this mini-project does"

# Track B
cd naive-rag
python ingest.py
python query.py "What is the difference between Startup delivery ops and affiliate flow?"
