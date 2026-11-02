"""Tools the agent can call: local files, shell, web search/fetch."""
import html
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) LocalAgent/1.0"
MAX_OUT = 12000


def _clip(text, n=MAX_OUT):
    return text if len(text) <= n else text[:n] + f"\n...[truncated {len(text) - n} chars]"


class Workspace:
    """All file access is confined to this root directory."""

    def __init__(self, root):
        self.root = os.path.realpath(os.path.expanduser(root))
        os.makedirs(self.root, exist_ok=True)

    def resolve(self, path):
        full = os.path.realpath(os.path.join(self.root, os.path.expanduser(path or ".")))
        if full != self.root and not full.startswith(self.root + os.sep):
            raise PermissionError(f"Path '{path}' is outside the workspace {self.root}")
        return full


# ---- tool implementations -------------------------------------------------

def list_dir(ws, path="."):
    full = ws.resolve(path)
    out = []
    for name in sorted(os.listdir(full)):
        p = os.path.join(full, name)
        out.append(f"{name}/" if os.path.isdir(p) else f"{name}  ({os.path.getsize(p)} bytes)")
    return "\n".join(out) or "(empty)"


def read_file(ws, path):
    with open(ws.resolve(path), "r", encoding="utf-8", errors="replace") as f:
        return _clip(f.read())


def write_file(ws, path, content):
    full = ws.resolve(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to {path}"


def edit_file(ws, path, old, new):
    full = ws.resolve(path)
    with open(full, "r", encoding="utf-8") as f:
        text = f.read()
    count = text.count(old)
    if count != 1:
        return f"Error: 'old' text found {count} times; it must match exactly once."
    with open(full, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new))
    return f"Edited {path}"


def delete_file(ws, path):
    full = ws.resolve(path)
    if os.path.isdir(full):
        os.rmdir(full)  # only empty dirs, deliberately
    else:
        os.remove(full)
    return f"Deleted {path}"


def run_command(ws, command, timeout=120):
    try:
        r = subprocess.run(command, shell=True, cwd=ws.root, capture_output=True,
                           text=True, timeout=int(timeout))
        return _clip(f"exit code: {r.returncode}\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"


def _get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        charset = r.headers.get_content_charset() or "utf-8"
        return r.read(2_000_000).decode(charset, errors="replace")


def html_to_text(raw):
    raw = re.sub(r"(?is)<(script|style|noscript|svg|head).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h\d>|</li>|</tr>", "\n", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
    return re.sub(r"\n\s*\n+", "\n\n", raw).strip()


def web_search(ws, query, max_results=8):
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    page = _get(url)
    results = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)'
                         r'(?:class="result__snippet"[^>]*>(.*?)</a>)?', page, re.S):
        href = html.unescape(m.group(1))
        q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        if "uddg" in q:
            href = q["uddg"][0]
        title = html_to_text(m.group(2))
        snippet = html_to_text(m.group(4) or "")
        results.append(f"- {title}\n  {href}\n  {snippet}")
        if len(results) >= int(max_results):
            break
    return "\n".join(results) or "No results found."


def fetch_url(ws, url):
    if not url.startswith(("http://", "https://")):
        return "Error: only http(s) URLs are allowed"
    return _clip(html_to_text(_get(url)))


# ---- schema ---------------------------------------------------------------

def _fn(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


S = {"type": "string"}
TOOLS = {
    "list_dir": (list_dir, _fn("list_dir", "List files in a workspace directory.",
                               {"path": S}, [])),
    "read_file": (read_file, _fn("read_file", "Read a text file from the workspace.",
                                 {"path": S}, ["path"])),
    "write_file": (write_file, _fn("write_file", "Create or overwrite a file with the full content.",
                                   {"path": S, "content": S}, ["path", "content"])),
    "edit_file": (edit_file, _fn("edit_file", "Replace one exact occurrence of 'old' with 'new' in a file.",
                                 {"path": S, "old": S, "new": S}, ["path", "old", "new"])),
    "delete_file": (delete_file, _fn("delete_file", "Delete a file (or empty folder).",
                                     {"path": S}, ["path"])),
    "run_command": (run_command, _fn("run_command", "Run a shell command in the workspace folder.",
                                     {"command": S, "timeout": {"type": "integer"}}, ["command"])),
    "web_search": (web_search, _fn("web_search", "Search the web. Returns titles, URLs and snippets.",
                                   {"query": S}, ["query"])),
    "fetch_url": (fetch_url, _fn("fetch_url", "Download a web page and return its readable text.",
                                 {"url": S}, ["url"])),
}

# Tools that change the user's machine -> need approval in manual mode.
MUTATING = {"write_file", "edit_file", "delete_file", "run_command"}
READ_ONLY = {"list_dir", "read_file", "web_search", "fetch_url"}


def describe_call(name, args):
    return f"{name}({json.dumps(args, ensure_ascii=False)[:300]})"
