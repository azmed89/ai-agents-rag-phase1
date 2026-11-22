# ReAct Agent — from scratch (Phase 1, Track A, refactored)

No Claude API, no Codex API, no Cursor, no framework. Ollama local only, text-parsed ReAct. Phase 1 is model-agnostic on purpose — swap Ollama for Codex / OpenAI later, the loop stays the same.

## Setup — from zero

```bash
ollama pull llama3.2
ollama serve   # keep running
pip install -r requirements.txt
python agent.py "List the files in ./demo and summarise what this mini-project does"
```

## Tools

- `read_file("path")` — read a file inside this folder
- `list_dir("path")` — list a folder
- `run_command("ls")` — short read-only shell command (blocked: rm, sudo, curl, installs, redirects)

## How the loop works

The model must write:
```
Thought: ...
Action: read_file("demo/README.md")
```
You parse that line, run the tool, feed back `Observation: ...`, repeat until `Final answer: ...` or 10 steps. All steps go to `session.json`.

Local models are weaker at following this format than Codex / Claude — if it drifts, that's the lesson. Real tools fix this with structured JSON tool-calling. You learn it raw first, then you appreciate the fix.

## Your exercise (Sat afternoon)

Add a 4th tool `write_file("path", "content")` yourself. Then test: a task with no possible tool, a bad path, Ollama stopped. Log what happens each time.
