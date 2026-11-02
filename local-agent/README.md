# Local AI Agent

A local, private AI agent with a web GUI. It runs on your own model (Ollama, LM Studio,
llama.cpp, vLLM, or anything with an OpenAI-compatible API). It uses only the Python
standard library, so there is nothing to `pip install`.

## What it can do
- **Local files**: list, read, write, edit and delete files inside a workspace folder you choose.
  It cannot touch anything outside that folder.
- **Shell**: run commands in the workspace folder.
- **Web**: search the web (DuckDuckGo) and read web pages.
- **Agents**: the agent can hand a sub-task to a helper agent with `spawn_agent`. Helpers can
  start their own helpers, up to 2 levels deep.

## Modes
| Mode | Behaviour |
|---|---|
| **Full autonomy** | Keeps working until it calls `finish`, you press **Stop**, or it hits the step limit (set it to 0 for no limit). |
| **Planning** | Looks around without changing anything, then shows a written plan. You can edit the plan, then approve it to run with full autonomy or with approvals, or reject it. |
| **Manual** | Every file write, edit or delete and every shell command waits for you to click Approve or Reject. Reading files and browsing the web don't need approval. |

## Quick start
1. Install [Ollama](https://ollama.com) and pull a model that supports tool calling:
   `ollama pull qwen2.5:7b` (other good options: `llama3.1:8b`, `qwen2.5:14b`, `mistral-nemo`).
2. `cd local-agent && python server.py --verbose`
3. Open http://localhost:8765. Use **Settings** to change the model, server URL or workspace
   folder (the default is `~/agent-workspace`).

## Logs
- **Show log** in the GUI opens a live panel with every event.
- `--verbose` also prints every event, colour-coded, in the terminal you started it from.
- Everything is always written to `logs/agent.log`.

Options: `--port`, `--host`, `--workspace PATH`, `--model NAME`.

**Safety:** full autonomy mode with shell access can run any command as your user.
For risky tasks, use Manual or Planning mode.
