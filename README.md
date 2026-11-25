# AI Agents + RAG — from scratch (Phase 1)

ReAct loop + naive RAG, built by hand in Python with Ollama only. Zero cost, no API key. Then compared with Codex.

## Results (20–22 Nov 2026)
- **Track A — ReAct:** 3/3 demo tasks in <10 steps. `write_file` in 4 steps. Weather question answered "Unknown" in 5 steps, no hallucination.
- **Track B — Naive RAG:** 9/10, then **10/10** after changing only the system prompt (retrieval 0.847 / 0.785 / 0.746 untouched). The bottleneck was the prompt, not retrieval.
- **Codex comparison:** same tasks in Codex passed in ~10–15s each, no drift, synthesis straight away. What Codex abstracts away: structured tool-calling, model strength, safety handling.

## Run it
```bash
cd react-agent
python agent.py "List the files in ./demo and summarise what this mini-project does"

cd ../naive-rag
python ingest.py
python query.py "What is the difference between Startup delivery ops and affiliate flow?"