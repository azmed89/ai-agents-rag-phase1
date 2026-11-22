# Phase 1 — Debrief: AI Agents + RAG from scratch

**Dates:** Fri 20 – Sun 22 Nov 2026
**Stack:** Python, Ollama only, zero cost, no API key
**Tracks:** `react-agent/` (ReAct loop, text-parsed) and `naive-rag/` (in-memory cosine, chunk 300 words / 50 overlap, top-k 5)

## Results

| Session | Result |
|---|---|
| Track A S1 — ReAct loop (Fri 20 Nov) | 3/3 demo tasks in < 10 steps. TODO task correctly said "not found". |
| Track B S1 — Naive RAG (Sat 21 Nov) | 9/10 questions answered, then **10/10** after a prompt-only fix. |
| Track A S2 — Break the agent (Sat 21 Nov) | `write_file` worked in 4 steps. Weather question answered "Unknown" in 5 steps, no hallucination. Bad paths blocked. Ollama-stopped error as expected. |
| Wrap-up (Sun 22 Nov) | ☐ Push both folders to GitHub · ✅ Codex comparison done 22 Nov 19:32 (see below) |

## Track A — what the loop actually did

- **Session 1, llama3.1:latest, TODO task:** 8 steps, correct final answer, but drift — repeated `list_dir("demo")`, model inventing its own `Observation:` lines inside its reply, misreading `ls` output.
- **Same task, llama3.1:8b:** 4 steps, clean — list root, list demo, `read_file("TODO.md")` errors, final answer "No TODO file found". No invented Observations, no loops.
- **Session 2, llama3.1:8b, write_file test:** 4 steps. Wrote `"Demo mini-project: Onessta order flow (sample) handling order creation and courier assignment"` to `demo/notes.txt` and read it back; final answer matches the file. Drift remained (two Actions in one reply, redundant re-reads, an invented "fix errors").
- **Session 2, weather in Casablanca (no possible tool):** 5 steps, final answer "Unknown". It tried README, `ls`, `list_dir`, even invented a `run_command("weather")` with a fake error — then stopped early instead of burning all 10 steps or inventing a temperature.

## Track B — the one failure, and the fix

Question: *"What is the difference between Onessta delivery ops and affiliate flow?"*

Retrieval was perfect, first try:

```
0.847  02-onessta-affiliate-flow.md
0.785  03-onessta-delivery-ops.md
0.746  01-onessta-overview.md
0.477  07-smart-invoice-processor.md
0.466  04-cursor-sdk-jira-automation.md
```

Generation refused anyway: *"I don't know from these documents."*

Cause: a comparison question needs synthesis across two docs, and no sentence states the difference. The strict prompt — *"Answer ONLY from the context below"* — made the small local model over-refuse.

Fix (Sun 22 Nov, one variable only — chunk 300 and top-k 5 unchanged):

```python
"Answer from the context below. You may compare and combine facts from different documents to answer. Use only facts present in the context, do not invent new facts. If the context truly contains no relevant facts, say 'I don't know from these documents.' Cite the source file name."
```

Same scores, correct synthesized answer citing `02-onessta-affiliate-flow.md` + `03-onessta-delivery-ops.md`. **9/10 → 10/10. The bottleneck was the prompt, not the retrieval.**

## Failure log — the brief for Phase 2

1. **Text-parsed ReAct is fragile by design.** Small local models drift: multiple `Thought:`/`Action:` per reply, invented `Observation:` lines, redundant re-reads, invented "fix errors". Fix: structured tool-calling (what Codex / Claude Code do with JSON).
2. **Over-refusal on synthesis questions.** A strict "answer ONLY from context" prompt fails comparison questions even when retrieval is perfect. Fix: allow comparing/combining facts across docs.
3. **Run-to-run variance is large, even in the same model family.** llama3.1:latest drifted for 8 steps; llama3.1:8b was clean in 4. Evaluate on multiple runs; consider temperature 0 for determinism.
4. **Argument parsing is quote-fragile.** `ARG_RE` splits on `"` — a `"` inside file content breaks the parse. Another reason for structured tool-calling.
5. **The guardrail gap.** `_safe_path` blocks escaping the project, but nothing stops the model overwriting `agent.py` or `session.json` inside it. Before giving an agent shell access in production: allow `demo/` only (or block code files), on top of the existing `run_command` blocklist.

## Codex comparison (Sun 22 Nov 19:32, Med ran it)

Same tasks, same folder (`C:/laragon/www/my-stack/ai-agents-rag-phase1/`), in Codex instead of the 120-line loop.

**ReAct tasks — all passed, fast, no drift:**
- Task 1, list + summarise `./demo`: **one run, ~15s**. Correctly listed README.md, orders.py, delivery.py, notes.txt (empty), and said "There is no TODO file" unprompted. Summary: "small demonstration project showing a simple order workflow: create an affiliate order, then assign a courier for delivery. It is sample code for the ReAct agent, not production code."
- Task 2, which file handles orders: **orders.py handles order creation for affiliates.** (19:13)
- Task 3, TODO file: **"There is no TODO file in ./demo."** (19:14)

**RAG questions — all passed, quick:**
- Jira cursor-ai automation: **11s**. Correct: planned for Phase 3, not built or running yet. Flow 1–5 (Jira task tagged cursor-ai → webhook → Cursor agent codes → commit → PR), OpenClaw as conversational entry point reporting back the PR link. Source: `04-cursor-sdk-jira-automation.md`.
- Difference delivery ops vs affiliate flow: **10s**. Correct synthesis: affiliates are seller-facing (register, browse, submit orders, no stock/confirmation/delivery, margin commission only on delivery); delivery ops handle everything after (confirmation, preparation, stock/CRM, courier assignment, delivery, staff/partner/platform management, Shopify/Youcan/WooCommerce/Google Sheets integrations, commission after delivery). "In short: affiliates generate orders; delivery operations process and fulfil them." Sources: `02-onessta-affiliate-flow.md` + `03-onessta-delivery-ops.md`.

**What the comparison shows:**
- My loop: 4–8 steps, local llama3.1:8b, text-parsed, drift (multiple Actions per reply, invented Observations, redundant re-reads), free, offline.
- Codex: 1 run per task, ~10–15s, no visible drift, synthesis on the comparison question straight away (my loop needed a prompt fix to do the same), stronger model, needs API / account.
- What Codex abstracts away: structured tool-calling instead of regex parsing, model strength that handles synthesis without a special prompt, and permission / safety handling I only approximated with `_safe_path` + a `run_command` blocklist.

## GitHub

**Repo:** https://github.com/azmed89/ai-agents-rag-phase1.git — pushed Sun 22 Nov 2026, HEAD `12f5230bb66b8b7e610745e0dd53e166e473a61e`, verified public and reachable.

Phase 1 is complete: all 4 sessions done, debrief + Codex comparison in this repo.

## How to run

```bash
# Track A
cd react-agent
ollama serve            # separate terminal, llama3.2 / llama3.1:8b
python agent.py "List the files in ./demo and summarise what this mini-project does"

# Track B
cd naive-rag
python ingest.py
python query.py "What is the difference between Onessta delivery ops and affiliate flow?"
```
