# ReAct Agent — from scratch (Phase 1, Track A, refactored)

No Claude API, no Codex API, no Cursor, no framework. Ollama local only, text-parsed ReAct. Phase 1 is model-agnostic on purpose — swap Ollama for Codex / OpenAI later, the loop stays the same.

## Setup — from zero

[ollama_agent.py](ollama_agent.py) is the original Phase 1, Track A implementation. Ollama local only, text-parsed ReAct.

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

## Step 2: OpenAI Agents SDK

[openai_agent.py](openai_agent.py) uses structured calls to `read_file`, `list_dir`, `run_command`, and `write_file`. Tools are restricted to `demo/`. `run_command` only lists directories.

```cmd
.venv\Scripts\python.exe -X utf8 openai_agent.py
```

If `OPENAI_API_KEY` is not set, the agent prompts for your key with hidden input. The key is kept only in the running process.

The three demo tasks each run three times at temperature 0. Append a quoted task to run that task three times. Set `OPENAI_MODEL` to override the default `gpt-4.1`.

## Step 3: Codex SDK

[agent.py](agent.py) uses your signed-in Codex plan and calls the four tools through MCP.

```cmd
codex login status
.venv\Scripts\python.exe -X utf8 agent.py
```

If Codex asks you to sign in, run `codex login` and choose **Sign in with ChatGPT**.

The three demo tasks each run three times in fresh threads. The Codex SDK uses its default sampling settings and does not expose a temperature setting. Append a quoted task to the run command to run that task three times.

The MCP tools accept paths inside `demo/` and reject links and traversal. `run_command` supports directory listing through `ls` or `dir`. The Codex thread disables shell access and uses a restricted permissions profile.

## Demo tasks

1. List the files in ./demo and summarise what this mini-project does
2. Read README.md in ./demo and tell me which file handles orders
3. Check if there is a TODO file — if not, say so, don't invent one

