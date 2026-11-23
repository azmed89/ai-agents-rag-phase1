"""
ReAct agent from scratch — Phase 1, Track A. REFACTORED for true from-scratch.
No Claude API, no Cursor, no OpenClaw, no framework. Ollama local only.

Run:
  ollama serve   # in another terminal
  ollama pull llama3.2
  python agent.py "List the files in ./demo and summarise what this mini-project does"

How it works — text-parsed ReAct, the honest version:
  You prompt the model to write:
    Thought: <reasoning>
    Action: read_file("demo/README.md")
  You parse that line with a regex, run the tool, and feed back:
    Observation: <tool result>
  Repeat until the model writes "Final answer: ..." or hits MAX_STEPS.
  Every step is saved to session.json.

Why text parsing, not native tool-calling? Because you see the loop.
Claude / Cursor SDK hide this behind JSON tool calls. Learn it raw first.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import ollama

BASE = Path(__file__).parent.resolve()
MODEL = "llama3.1:8b"  # change to "llama4:scout" if your Ollama has it
MAX_STEPS = 10

SYSTEM = """You are a coding assistant that works in a ReAct loop.
You can only act with these tools, one per step:
  read_file("path") — read a text file inside the project, e.g. read_file("demo/README.md")
  list_dir("path") — list a folder, e.g. list_dir(".") or list_dir("demo")
  run_command("ls") — short read-only shell command, e.g. run_command("ls demo")
  write_file("path", "content") - write a text file inside the project, e.g. write_file("demo/notes.txt", "hello")

Every reply must follow this exact format, no extra text outside it:
Thought: <one sentence of reasoning>
Action: <one tool call, e.g. read_file("demo/README.md")>
OR, when done:
Thought: <why you are done>
Final answer: <short answer with what you found>

If a tool result says a file does not exist, say so. Never invent file contents.
Never call a tool that is not in the list above.
"""

ACTION_RE = re.compile(r'Action:\s*(\w+)\((.*)\)\s*$', re.DOTALL)
ARG_RE = re.compile(r'"([^"]*)"')


def _safe_path(rel: str) -> Path:
    p = (BASE / rel).resolve()
    if not str(p).startswith(str(BASE)):
        raise ValueError(f"Path outside project: {rel}")
    return p


def read_file(path: str) -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    return p.read_text(encoding="utf-8")[:4000]

def write_file(path: str, content: str) -> str:
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK: wrote {len(content)} chars to {path}"

def list_dir(path: str = ".") -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"ERROR: folder not found: {path}"
    return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir()))


def run_command(command: str) -> str:
    blocked = ["rm ", "sudo", "curl", "wget", "pip install", "npm install", ">", "&&"]
    if any(b in command for b in blocked):
        return f"ERROR: command not allowed in Phase 1: {command}"
    try:
        r = subprocess.run(command, shell=True, cwd=BASE, capture_output=True, text=True, timeout=15)
        out = (r.stdout + r.stderr)[:4000]
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 15s"


TOOL_FUNCS = {"read_file": read_file, "list_dir": list_dir, "run_command": run_command, "write_file": write_file}


def parse_action(text: str):
    m = ACTION_RE.search(text)
    if not m:
        return None, None
    name = m.group(1)
    args = ARG_RE.findall(m.group(2))
    return name, args


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python agent.py "your task"')
        sys.exit(1)

    task = sys.argv[1]
    history = f"{SYSTEM}\n\nTask: {task}\n"
    steps = []

    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- step {step} ---")
        try:
            resp = ollama.chat(model=MODEL, messages=[{"role": "user", "content": history}])
        except Exception as e:  # noqa: BLE001 — Ollama not running is the #1 from-scratch error
            print(f"ERROR calling Ollama ({MODEL}): {e}")
            print("Is `ollama serve` running? Did you run `ollama pull llama3.2`?")
            sys.exit(1)

        text = resp["message"]["content"]
        print(text[:800])
        steps.append({"step": step, "model_reply": text})

        if "Final answer:" in text:
            print(f"\nDone in {step} steps.")
            break

        name, args = parse_action(text)
        if not name or name not in TOOL_FUNCS:
            history += f"\n{text}\nObservation: ERROR: no valid Action found. Use exactly: Action: read_file(\"demo/README.md\")\n"
            print("Observe: ERROR — could not parse Action, retrying with a hint.")
            continue

        print(f"Tool: {name} {args}")
        try:
            result = TOOL_FUNCS[name](*args) if args else TOOL_FUNCS[name]()
        except Exception as e:  # noqa: BLE001 — show the error to the model
            result = f"ERROR: {e}"
        print(f"Observe: {str(result)[:400]}")
        history += f"\n{text}\nObservation: {result}\n"
    else:
        print(f"\nStopped: hit MAX_STEPS={MAX_STEPS} without a final answer.")

    (BASE / "session.json").write_text(
        json.dumps({"task": task, "model": MODEL, "steps": steps}, indent=2), encoding="utf-8"
    )
    print("Session saved to session.json")


if __name__ == "__main__":
    main()
