"""Local AI Agent - web GUI server (Python standard library only).

Run:  python server.py [--verbose] [--port 8765] [--workspace ~/agent-workspace]
Then open http://localhost:8765
"""
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import agent as A
from tools import Workspace

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(HERE, "settings.json")
DEFAULTS = {
    "base_url": "http://localhost:11434/v1",   # Ollama's OpenAI-compatible endpoint
    "model": "qwen2.5:7b",
    "api_key": "local",
    "workspace": os.path.join(os.path.expanduser("~"), "agent-workspace"),
    "max_steps": 50,                           # 0 = unlimited
    "temperature": 0.2,
}

COLORS = {"error": "31", "approval": "33", "tool_call": "36", "tool_result": "90",
          "agent_done": "32", "plan": "35", "state": "34"}


class App:
    def __init__(self, verbose):
        self.verbose = verbose
        self.settings = dict(DEFAULTS)
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE) as f:
                self.settings.update(json.load(f))
        self.events, self.cond = [], threading.Condition()
        self.session, self.state, self.plan, self.task = None, "idle", None, None
        os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
        self.logfile = open(os.path.join(HERE, "logs", "agent.log"), "a", encoding="utf-8")

    # -- events ---------------------------------------------------------
    def emit(self, kind, data):
        ev = {"id": None, "t": time.strftime("%H:%M:%S"), "type": kind, "data": data}
        with self.cond:
            ev["id"] = len(self.events)
            self.events.append(ev)
            self.cond.notify_all()
        line = f"[{ev['t']}] {kind:<12} {json.dumps(data, ensure_ascii=False)}"
        self.logfile.write(line + "\n")
        self.logfile.flush()
        if self.verbose:
            print(f"\033[{COLORS.get(kind, '0')}m{line[:2000]}\033[0m", flush=True)

    def set_state(self, state):
        self.state = state
        self.emit("state", {"state": state})

    # -- task control ---------------------------------------------------
    def new_session(self, mode):
        s = self.settings
        llm = A.LLM(s["base_url"], s["model"], s["api_key"], float(s["temperature"]))
        return A.Session(self.emit, Workspace(s["workspace"]), llm, mode, int(s["max_steps"]))

    def start(self, task, mode):
        if self.state in ("running", "planning"):
            return "A task is already running."
        self.task, self.plan = task, None
        self.session = self.new_session(mode)
        self.emit("task", {"task": task, "mode": mode})
        threading.Thread(target=self._run, args=(task, mode), daemon=True).start()

    def _run(self, task, mode, plan=None):
        try:
            if mode == "plan":
                self.set_state("planning")
                self.plan = A.Agent(self.session, planning=True).run(task)
                self.emit("plan", {"plan": self.plan})
                self.set_state("awaiting_plan")
                return
            self.set_state("running")
            ctx = f"The user approved this plan. Follow it:\n{plan}" if plan else ""
            result = A.Agent(self.session).run(task, ctx)
            self.emit("result", {"summary": result})
            self.set_state("done")
        except A.Stopped:
            self.emit("info", {"text": "Stopped by user."})
            self.set_state("stopped")
        except Exception as e:
            self.emit("error", {"text": f"{type(e).__name__}: {e}"})
            self.set_state("error")

    def approve_plan(self, exec_mode, edited_plan=None):
        if self.state != "awaiting_plan":
            return "No plan is waiting for approval."
        plan = edited_plan or self.plan
        self.session = self.new_session(exec_mode)
        threading.Thread(target=self._run, args=(self.task, exec_mode, plan), daemon=True).start()

    def stop(self):
        if self.session:
            self.session.stop_flag.set()
        if self.state == "awaiting_plan":
            self.set_state("stopped")


class Handler(BaseHTTPRequestHandler):
    app: App = None

    def log_message(self, *a):  # silence default access log
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        app = self.app
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "static", "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/settings":
            self._json({**app.settings, "state": app.state})
        elif self.path.startswith("/api/events"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            i = int(self.headers.get("Last-Event-ID") or -1) + 1
            try:
                while True:
                    with app.cond:
                        app.cond.wait_for(lambda: len(app.events) > i, timeout=15)
                        batch = app.events[i:]
                    i += len(batch)
                    out = "".join(f"id: {e['id']}\ndata: {json.dumps(e)}\n\n" for e in batch) or ": ping\n\n"
                    self.wfile.write(out.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        app, b = self.app, self._body()
        err = None
        if self.path == "/api/task":
            if not b.get("task", "").strip():
                err = "Task is empty."
            else:
                err = app.start(b["task"], b.get("mode", "auto"))
        elif self.path == "/api/stop":
            app.stop()
        elif self.path == "/api/approve":
            if not (app.session and app.session.resolve(str(b.get("id")), bool(b.get("approved")))):
                err = "Unknown approval id."
        elif self.path == "/api/plan":
            if b.get("action") == "reject":
                app.set_state("idle")
            else:
                err = app.approve_plan(b.get("exec_mode", "auto"), b.get("plan"))
        elif self.path == "/api/settings":
            app.settings.update({k: v for k, v in b.items() if k in DEFAULTS})
            with open(SETTINGS_FILE, "w") as f:
                json.dump(app.settings, f, indent=2)
        else:
            return self._json({"error": "not found"}, 404)
        self._json({"ok": not err, "error": err}, 400 if err else 200)


def main():
    p = argparse.ArgumentParser(description="Local AI Agent")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--verbose", "-v", action="store_true", help="print every agent event to this terminal")
    p.add_argument("--workspace", help="folder the agent may read/write")
    p.add_argument("--model")
    a = p.parse_args()
    Handler.app = app = App(a.verbose)
    if a.workspace:
        app.settings["workspace"] = a.workspace
    if a.model:
        app.settings["model"] = a.model
    print(f"Local AI Agent running at http://{a.host}:{a.port}")
    print(f"  model={app.settings['model']} @ {app.settings['base_url']}")
    print(f"  workspace={app.settings['workspace']}  log=logs/agent.log  verbose={a.verbose}")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
