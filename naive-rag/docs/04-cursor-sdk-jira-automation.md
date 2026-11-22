# Cursor SDK — Jira automation (planned, not built yet)

This is the target automation for Phase 3, not something already running.

The idea: when a new task is created in Jira and tagged cursor-ai, a webhook fires. A new agent starts coding on that task, then creates a new commit and opens a pull request. The flow is: Jira webhook, new agent start coding, new commit, pull request.

This is event-driven orchestration: Jira webhook to agent to commit to PR. It is the automation Phase 1 is building towards — first learn the ReAct loop, then automate with the Cursor SDK.
