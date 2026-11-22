# OpenClaw — target setup (not installed yet)

OpenClaw is not installed yet. This doc describes what it will do in Phase 3.

When a message comes in, the OpenClaw Gateway normalizes it, assembles context, runs inference, executes tools in a ReAct loop, and persists memory.

OpenClaw is the conversational entry point. In Phase 3, an OpenClaw skill will trigger the Cursor SDK automation: message OpenClaw, it creates a Jira task, tags it cursor-ai, the Cursor agent codes, commits, opens a PR, and OpenClaw reports back with the PR link.
