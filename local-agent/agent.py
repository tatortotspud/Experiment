"""The agent loop: talks to a local LLM, calls tools, spawns sub-agents."""
import itertools
import json
import threading
import time
import urllib.request

import tools as T

SYSTEM = """You are a capable autonomous assistant running on the user's own computer.
You can read/write files in the workspace ({root}), run shell commands, search the web,
fetch web pages, and delegate self-contained sub-tasks to helper agents with spawn_agent.

Work step by step. Use tools rather than guessing. Check your work.
When the objective is complete, call the `finish` tool with a clear summary of what you did.
If you truly cannot continue, call `finish` and explain what blocked you."""

PLAN_SYSTEM = """You are in PLANNING mode. Do NOT change anything. You may only look around
(list_dir, read_file, web_search, fetch_url) to understand the task.
Then call `finish` with a proposal in Markdown containing:
1. Goal (one line)  2. Numbered steps, naming files/commands/sites you will touch
3. Risks or questions for the user  4. How you will verify success."""

EXTRA_TOOLS = {
    "finish": T._fn("finish", "Call when the objective is complete (or impossible). Give a summary.",
                    {"summary": T.S}, ["summary"]),
    "spawn_agent": T._fn("spawn_agent",
                         "Delegate a self-contained sub-task to a helper agent. It has the same tools "
                         "and returns its final summary. Give it full context; it cannot see your history.",
                         {"task": T.S, "role": {"type": "string",
                                                 "description": "e.g. researcher, coder, reviewer"}},
                         ["task"]),
}

_ids = itertools.count(1)


class Stopped(Exception):
    pass


class LLM:
    """Minimal OpenAI-compatible chat client (works with Ollama, LM Studio, llama.cpp, vLLM)."""

    def __init__(self, base_url, model, api_key="local", temperature=0.2):
        self.base_url = base_url.rstrip("/")
        self.model, self.api_key, self.temperature = model, api_key, temperature

    def chat(self, messages, tools):
        body = json.dumps({"model": self.model, "messages": messages, "tools": tools,
                           "temperature": self.temperature, "stream": False}).encode()
        req = urllib.request.Request(self.base_url + "/chat/completions", data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["choices"][0]["message"]


class Session:
    """Shared state for one task: event stream, stop flag, pending approvals."""

    def __init__(self, emit, workspace, llm, mode, max_steps):
        self.emit, self.ws, self.llm = emit, workspace, llm
        self.mode = mode            # "auto" | "manual" | "plan"
        self.max_steps = max_steps  # 0 = unlimited
        self.stop_flag = threading.Event()
        self.pending = {}           # approval id -> {"event", "approved"}
        self.lock = threading.Lock()

    def check_stop(self):
        if self.stop_flag.is_set():
            raise Stopped()

    def request_approval(self, agent_name, name, args):
        aid = str(next(_ids))
        ev = threading.Event()
        with self.lock:
            self.pending[aid] = {"event": ev, "approved": False}
        self.emit("approval", {"id": aid, "agent": agent_name, "tool": name, "args": args})
        while not ev.wait(0.5):
            self.check_stop()
        with self.lock:
            return self.pending.pop(aid)["approved"]

    def resolve(self, aid, approved):
        with self.lock:
            p = self.pending.get(aid)
            if p:
                p["approved"] = approved
                p["event"].set()
        return bool(p)


class Agent:
    def __init__(self, session, name="main", depth=0, planning=False):
        self.s, self.name, self.depth, self.planning = session, name, depth, planning
        allowed = T.READ_ONLY if planning else set(T.TOOLS)
        self.tool_schemas = [T.TOOLS[k][1] for k in T.TOOLS if k in allowed]
        self.tool_schemas.append(EXTRA_TOOLS["finish"])
        if not planning and depth < 2:
            self.tool_schemas.append(EXTRA_TOOLS["spawn_agent"])
        self.allowed = allowed | {"finish", "spawn_agent"}

    def log(self, kind, **data):
        self.s.emit(kind, {"agent": self.name, **data})

    def run(self, task, extra_context=""):
        system = (PLAN_SYSTEM if self.planning else SYSTEM).format(root=self.s.ws.root)
        if extra_context:
            system += "\n\n" + extra_context
        messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
        self.log("agent_start", task=task)
        step = 0
        while True:
            self.s.check_stop()
            step += 1
            if self.s.max_steps and step > self.s.max_steps:
                self.log("info", text=f"Reached the step limit ({self.s.max_steps}).")
                return "Stopped: step limit reached."
            self.log("thinking", step=step)
            t0 = time.time()
            try:
                msg = self.s.llm.chat(messages, self.tool_schemas)
            except Exception as e:  # network / model errors
                self.log("error", text=f"LLM request failed: {e}")
                raise
            self.s.check_stop()
            self.log("llm", seconds=round(time.time() - t0, 1), content=msg.get("content") or "")
            calls = msg.get("tool_calls") or []
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             **({"tool_calls": calls} if calls else {})})
            if not calls:
                text = msg.get("content") or ""
                if self.planning or step > 1:  # a plain answer counts as done
                    self.log("agent_done", summary=text)
                    return text
                messages.append({"role": "user", "content":
                                 "Continue working using tools. Call `finish` when the objective is done."})
                continue
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    args = {}
                if name == "finish":
                    summary = args.get("summary", "")
                    self.log("agent_done", summary=summary)
                    return summary
                result = self.call_tool(name, args)
                messages.append({"role": "tool", "tool_call_id": call.get("id", name),
                                 "name": name, "content": result})

    def call_tool(self, name, args):
        self.log("tool_call", tool=name, args=args)
        if name not in self.allowed:
            result = f"Error: tool '{name}' is not available in this mode."
        elif self.s.mode == "manual" and name in T.MUTATING and \
                not self.s.request_approval(self.name, name, args):
            result = "The user REJECTED this action. Do not retry it; choose another approach or ask via finish."
        elif name == "spawn_agent":
            child = Agent(self.s, name=f"{self.name}>{args.get('role') or 'helper'}#{next(_ids)}",
                          depth=self.depth + 1)
            result = child.run(args.get("task", ""))
        else:
            try:
                result = str(T.TOOLS[name][0](self.s.ws, **args))
            except Stopped:
                raise
            except Exception as e:
                result = f"Error: {type(e).__name__}: {e}"
        self.log("tool_result", tool=name, result=result[:4000])
        return result
