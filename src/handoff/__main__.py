#!/usr/bin/env python3
"""
handoff.py — claude <-> codex session handoff tool
lists sessions, shows token estimates for 3 context tiers, launches target tool
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── token estimation ──────────────────────────────────────────────────────────

_tokenizer = None

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        import tiktoken
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer

def est_tokens(text: str) -> int:
    """token count via tiktoken cl100k_base (used by both gpt-4/codex and claude)"""
    return len(_get_tokenizer().encode(text))

# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class Message:
    role: str   # 'user' | 'assistant'
    content: str

@dataclass
class ToolCall:
    name: str
    summary: str

@dataclass
class ThinkingBlock:
    text: str

@dataclass
class SessionData:
    id: str
    source: str           # 'claude' | 'codex'
    cwd: str
    summary: str          # first user message
    updated_at: datetime
    path: Path
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: list[ThinkingBlock] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    model: str = ""
    branch: str = ""
    token_usage: tuple[int, int] = (0, 0)  # (input, output)
    size_bytes: int = 0

# ── claude parser ─────────────────────────────────────────────────────────────

def _extract_text_blocks(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    pass  # skip in text extraction
        return "\n".join(p for p in parts if p)
    return ""

_HANDOFF_MARKERS = (
    "Session Handoff",
    "i was working with claude on a coding task",
    "i was working with codex on a coding task",
    "I'm continuing a coding session",
    "continuing a coding session",
    "previous session (Claude Code",
    "previous session (OpenAI Codex",
)

def _is_real_user_message(text: str) -> bool:
    """filter out system-injected content, slash commands, xml tags, handoff summaries"""
    t = text.strip()
    if not t:
        return False
    if t.startswith("<") or t.startswith("/") or t.startswith("# AGENTS.md"):
        return False
    if any(marker in t for marker in _HANDOFF_MARKERS):
        return False
    return True

def parse_claude_sessions() -> list[SessionData]:
    base = Path.home() / ".claude" / "projects"
    if not base.exists():
        return []

    uuid_re = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.jsonl$", re.I)
    sessions = []

    for jsonl_file in base.rglob("*.jsonl"):
        if not uuid_re.match(jsonl_file.name):
            continue
        if "debug" in jsonl_file.name:
            continue
        try:
            lines = jsonl_file.read_text(errors="replace").splitlines()
            if not lines:
                continue

            session_id = jsonl_file.stem
            cwd = ""
            first_user = ""
            model = ""
            branch = ""

            # quick scan for metadata
            for line in lines[:50]:
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                if msg.get("sessionId") and not session_id:
                    session_id = msg["sessionId"]
                if msg.get("cwd") and not cwd:
                    cwd = msg["cwd"]
                if msg.get("model") and not model:
                    model = msg["model"]
                if msg.get("gitBranch") and not branch:
                    branch = msg["gitBranch"]
                if not first_user and msg.get("type") == "user":
                    content = msg.get("message", {}).get("content", "")
                    text = _extract_text_blocks(content)
                    if _is_real_user_message(text):
                        first_user = text[:100]

            stat = jsonl_file.stat()
            if stat.st_size < 200:
                continue

            sessions.append(SessionData(
                id=session_id,
                source="claude",
                cwd=cwd,
                summary=first_user.strip(),
                updated_at=datetime.fromtimestamp(stat.st_mtime),
                path=jsonl_file,
                model=model,
                branch=branch,
                size_bytes=stat.st_size,
            ))
        except Exception:
            continue

    sessions.sort(key=lambda s: s.updated_at, reverse=True)
    return sessions

def load_claude_context(session: SessionData) -> None:
    """populate messages, tool_calls, thinking from the full jsonl"""
    lines = session.path.read_text(errors="replace").splitlines()
    raw_msgs = []
    for line in lines:
        try:
            raw_msgs.append(json.loads(line))
        except Exception:
            continue

    # collect tool results by id
    tool_results: dict[str, str] = {}
    for msg in raw_msgs:
        content = msg.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result" and block.get("tool_use_id"):
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    result_text = " ".join(b.get("text", "") for b in result_content if isinstance(b, dict))
                else:
                    result_text = str(result_content)
                tool_results[block["tool_use_id"]] = result_text[:200]

    skip_tools = {"TodoRead", "TodoWrite"}
    tool_calls: list[ToolCall] = []
    thinking: list[ThinkingBlock] = []
    files_modified: set[str] = set()
    messages: list[Message] = []
    model = session.model

    for msg in raw_msgs:
        if msg.get("type") in ("queue-operation", "system"):
            continue
        if msg.get("isCompactSummary"):
            continue
        if not model and msg.get("model"):
            model = msg["model"]

        content = msg.get("message", {}).get("content", "")
        if not isinstance(content, list):
            text = _extract_text_blocks(content)
            if text and _is_real_user_message(text):
                if msg.get("type") == "user":
                    messages.append(Message("user", text))
                elif msg.get("type") == "assistant":
                    messages.append(Message("assistant", text))
            continue

        # extract text for conversation
        text_parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")

            if btype == "text":
                text_parts.append(block.get("text", ""))

            elif btype == "thinking":
                t = block.get("thinking") or block.get("text", "")
                if t and len(t) > 20:
                    first_line = re.split(r"[.\n]", t)[0].strip()
                    if first_line:
                        thinking.append(ThinkingBlock(first_line[:300]))

            elif btype == "tool_use" and block.get("name") not in skip_tools:
                name = block.get("name", "")
                inp = block.get("input", {})
                result = tool_results.get(block.get("id", ""), "")
                summary = _format_tool_call(name, inp, result)
                fp = inp.get("file_path") or inp.get("path", "")
                if fp and name in ("Write", "Edit", "NotebookEdit"):
                    files_modified.add(fp)
                tool_calls.append(ToolCall(name, summary))

        full_text = "\n".join(p for p in text_parts if p)
        if full_text:
            role = "user" if msg.get("type") == "user" else "assistant"
            if role == "user" and not _is_real_user_message(full_text):
                continue
            messages.append(Message(role, full_text))

    session.messages = messages
    session.tool_calls = tool_calls
    session.thinking = thinking
    session.files_modified = sorted(files_modified)
    session.model = model

def _format_tool_call(name: str, inp: dict, result: str = "") -> str:
    shell_tools = {"Bash", "exec_command", "shell_command"}
    read_tools = {"Read"}
    write_tools = {"Write", "NotebookEdit"}
    edit_tools = {"Edit"}
    grep_tools = {"Grep"}
    glob_tools = {"Glob"}

    def with_exit(cmd: str, res: str) -> str:
        m = re.search(r"exit(?:ed with)? code[:\s]+(\d+)", res, re.I)
        if m:
            return f"$ {cmd[:80]} -> exit {m.group(1)}"
        return f"$ {cmd[:80]}"

    if name in shell_tools:
        cmd = inp.get("command") or inp.get("cmd", "")
        return with_exit(cmd, result)
    if name in read_tools:
        return f"read {inp.get('file_path', '')}"
    if name in write_tools:
        return f"write {inp.get('file_path', '')}"
    if name in edit_tools:
        return f"edit {inp.get('file_path', '')}"
    if name in grep_tools:
        return f"grep \"{inp.get('pattern', '')}\" {inp.get('path', '')}"
    if name in glob_tools:
        return f"glob \"{inp.get('pattern', '')}\""
    if name.startswith("mcp__") or "___" in name:
        return f"{name}({json.dumps(inp)[:80]})"
    return f"{name}({json.dumps(inp)[:60]})"

# ── codex parser ──────────────────────────────────────────────────────────────

def parse_codex_sessions() -> list[SessionData]:
    base = Path.home() / ".codex" / "sessions"
    if not base.exists():
        return []

    filename_re = re.compile(
        r"rollout-(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})-(.+)\.jsonl$"
    )
    sessions = []

    for jsonl_file in base.rglob("rollout-*.jsonl"):
        m = filename_re.match(jsonl_file.name)
        if not m:
            continue
        try:
            yr, mo, dy, hr, mn, sc, sid = m.groups()
            ts = datetime(int(yr), int(mo), int(dy), int(hr), int(mn), int(sc))

            lines = jsonl_file.read_text(errors="replace").splitlines()
            cwd = ""
            first_user = ""
            model = ""
            branch = ""

            for line in lines[:150]:
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                if msg.get("type") == "session_meta":
                    payload = msg.get("payload", {})
                    if not cwd:
                        cwd = payload.get("cwd", "")
                    if not model:
                        model = payload.get("model", "")
                    if not branch:
                        branch = payload.get("git_branch") or payload.get("gitBranch", "")
                if not first_user and msg.get("type") == "event_msg":
                    payload = msg.get("payload", {})
                    if payload.get("type") == "user_message":
                        text = (payload.get("message") or "")[:200]
                        if _is_real_user_message(text):
                            first_user = text[:100]
                if not first_user and msg.get("type") == "message" and msg.get("role") == "user":
                    text = str(msg.get("content", ""))[:200]
                    if _is_real_user_message(text):
                        first_user = text[:100]

            stat = jsonl_file.stat()
            sessions.append(SessionData(
                id=sid,
                source="codex",
                cwd=cwd,
                summary=first_user.strip(),
                updated_at=datetime.fromtimestamp(stat.st_mtime),
                path=jsonl_file,
                model=model,
                branch=branch,
                size_bytes=stat.st_size,
            ))
        except Exception:
            continue

    sessions.sort(key=lambda s: s.updated_at, reverse=True)
    return sessions

def load_codex_context(session: SessionData) -> None:
    lines = session.path.read_text(errors="replace").splitlines()
    raw_msgs = []
    for line in lines:
        try:
            raw_msgs.append(json.loads(line))
        except Exception:
            continue

    # collect outputs by call_id
    outputs: dict[str, str] = {}
    for msg in raw_msgs:
        if msg.get("type") != "response_item":
            continue
        payload = msg.get("payload", {})
        if payload.get("type") in ("function_call_output", "custom_tool_call_output"):
            call_id = payload.get("call_id")
            output = payload.get("output", "")
            if call_id and output:
                outputs[call_id] = str(output)[:200]

    event_messages: list[Message] = []
    response_messages: list[Message] = []
    tool_calls: list[ToolCall] = []
    thinking: list[ThinkingBlock] = []
    files_modified: set[str] = set()
    model = session.model
    token_input = 0
    token_output = 0

    skip_content_prefixes = ("<environment_context>", "<permissions", "# AGENTS.md")

    for msg in raw_msgs:
        mtype = msg.get("type")

        if mtype == "turn_context":
            if msg.get("payload", {}).get("model") and not model:
                model = msg["payload"]["model"]

        elif mtype == "event_msg":
            payload = msg.get("payload", {})
            etype = payload.get("type")
            if etype == "user_message":
                content = payload.get("message") or msg.get("message", "")
                if content:
                    event_messages.append(Message("user", content))
            elif etype in ("agent_message", "assistant_message"):
                content = payload.get("message", "")
                if content:
                    event_messages.append(Message("assistant", content))
            elif etype == "agent_reasoning":
                text = payload.get("message", "")
                if text and len(text) > 20:
                    first_line = re.split(r"[.\n]", text)[0].strip()
                    if first_line:
                        thinking.append(ThinkingBlock(first_line[:300]))
            elif etype == "token_count":
                token_input = payload.get("input_tokens", 0)
                token_output = payload.get("output_tokens", 0)

        elif mtype == "response_item":
            payload = msg.get("payload", {})
            ptype = payload.get("type")

            if ptype == "message":
                role = payload.get("role")
                content_parts = payload.get("content", [])

                if role == "user":
                    text = " ".join(
                        p.get("text", "") for p in content_parts
                        if isinstance(p, dict) and p.get("type") == "input_text" and p.get("text")
                    )
                    if text and not any(text.startswith(s) for s in skip_content_prefixes):
                        response_messages.append(Message("user", text))

                elif role == "assistant":
                    text = " ".join(
                        p.get("text", "") for p in content_parts
                        if isinstance(p, dict) and p.get("type") in ("output_text", "text") and p.get("text")
                    )
                    if text:
                        response_messages.append(Message("assistant", text))

            elif ptype == "function_call" and payload.get("arguments"):
                try:
                    name = payload.get("name", "")
                    args = json.loads(payload["arguments"])
                    result = outputs.get(payload.get("call_id", ""), "")
                    summary = _format_tool_call(name, args, result)
                    tool_calls.append(ToolCall(name, summary))
                except Exception:
                    pass

            elif ptype == "custom_tool_call" and payload.get("name") == "apply_patch":
                patch_input = payload.get("input", "")
                file_matches = re.findall(r"\*\*\* (?:Add|Update|Delete) File: (.+)", patch_input)
                file_list = ", ".join(file_matches[:3]) if file_matches else "(patch)"
                tool_calls.append(ToolCall("apply_patch", f"patch: {file_list[:70]}"))
                for f in file_matches:
                    files_modified.add(f)

            elif ptype == "web_search_call":
                action = payload.get("action", {})
                query = action.get("query") or (action.get("queries") or [""])[0]
                tool_calls.append(ToolCall("web_search", f"search \"{query[:60]}\""))

    # prefer response_item messages (richer format); fall back to event_msg
    has_response_user = any(m.role == "user" for m in response_messages)
    messages = response_messages if has_response_user else event_messages

    session.messages = messages
    session.tool_calls = tool_calls
    session.thinking = thinking
    session.files_modified = sorted(files_modified)
    session.model = model
    session.token_usage = (token_input, token_output)

# ── handoff markdown builders ─────────────────────────────────────────────────

DEFAULT_HANDOFF_PROMPT = "read through this context and let me know once you're up to speed. don't start coding until i say so."

def build_tool_activity(tool_calls: list[ToolCall]) -> str:
    if not tool_calls:
        return ""
    groups: dict[str, list[str]] = {}
    for tc in tool_calls:
        groups.setdefault(tc.name, []).append(tc.summary)
    lines = ["tools used:"]
    for name, summaries in groups.items():
        samples = ", ".join(summaries[:3])
        lines.append(f"  {name} (x{len(summaries)}): {samples}")
    return "\n".join(lines)

def build_conversation(messages: list[Message]) -> str:
    if not messages:
        return ""
    lines = ["conversation:"]
    for msg in messages:
        role = "User" if msg.role == "user" else "Agent"
        lines.append(f"\n{role}: {msg.content}")
    return "\n".join(lines)

def build_thinking(thinking: list[ThinkingBlock]) -> str:
    if not thinking:
        return ""
    lines = ["key decisions:"]
    for t in thinking[:5]:
        lines.append(f"  - {t.text}")
    return "\n".join(lines)

def build_header(session: SessionData) -> str:
    source_label = "Claude Code" if session.source == "claude" else "OpenAI Codex"
    summary = clean_summary(session.summary or "")[:120]
    return f"previous session ({source_label}, {session.cwd}): {summary}"

def build_files(files: list[str]) -> str:
    if not files:
        return ""
    return "files modified: " + ", ".join(files)

def _trim_conversation(messages: list[Message], budget: int) -> list[Message]:
    """return as many recent messages as fit within token budget, trimming from the front"""
    if not messages:
        return messages
    # try all messages first
    total = est_tokens(build_conversation(messages))
    if total <= budget:
        return messages
    # binary search: find max suffix that fits
    lo, hi = 0, len(messages)
    while lo < hi:
        mid = (lo + hi) // 2
        if est_tokens(build_conversation(messages[mid:])) <= budget:
            hi = mid
        else:
            lo = mid + 1
    return messages[lo:]

def make_tier1(session: SessionData, max_tokens: int = 100_000) -> str:
    """full: header + tools + thinking + conversation + files"""
    non_convo = "\n\n".join(p for p in [
        build_header(session),
        build_tool_activity(session.tool_calls),
        build_thinking(session.thinking),
        build_files(session.files_modified),
    ] if p)
    convo_budget = max_tokens - est_tokens(non_convo)
    msgs = _trim_conversation(session.messages, convo_budget)
    convo = build_conversation(msgs)
    parts = [non_convo, convo]
    return "\n\n".join(p for p in parts if p)

def make_tier2(session: SessionData, max_tokens: int = 100_000) -> str:
    """focused: header + tools + conversation + files"""
    non_convo = "\n\n".join(p for p in [
        build_header(session),
        build_tool_activity(session.tool_calls),
        build_files(session.files_modified),
    ] if p)
    convo_budget = max_tokens - est_tokens(non_convo)
    msgs = _trim_conversation(session.messages, convo_budget)
    convo = build_conversation(msgs)
    parts = [non_convo, convo]
    return "\n\n".join(p for p in parts if p)

def make_tier3(session: SessionData, max_tokens: int = 100_000) -> str:
    """minimal: header + conversation only"""
    header = build_header(session)
    convo_budget = max_tokens - est_tokens(header)
    msgs = _trim_conversation(session.messages, convo_budget)
    convo = build_conversation(msgs)
    parts = [header, convo]
    return "\n\n".join(p for p in parts if p)

# ── launch ────────────────────────────────────────────────────────────────────

def launch_with_context(target: str, source: str, markdown: str, cwd: str, handoff_prompt: str) -> None:
    handoff_path = Path(cwd) / ".handoff.md" if cwd else Path.cwd() / ".handoff.md"
    try:
        handoff_path.write_text(markdown)
        print(f"  wrote {handoff_path}")
    except Exception as e:
        print(f"  warning: could not write handoff file: {e}")

    intro = f"i was working with {source} on a coding task and am continuing that session here. here's the context from that conversation:\n\n{markdown}\n\n---\n\n{handoff_prompt}"
    work_dir = cwd if cwd and Path(cwd).exists() else str(Path.cwd())

    if target == "codex":
        cmd = ["codex", intro]
    elif target == "claude":
        cmd = ["claude", intro]
    else:
        print(f"unknown target: {target}")
        sys.exit(1)

    print(f"  launching {target} in {work_dir}\n")
    try:
        subprocess.run(cmd, cwd=work_dir)
    except FileNotFoundError:
        print(f"error: '{target}' binary not found in PATH")
        sys.exit(1)

# ── helpers ───────────────────────────────────────────────────────────────────

def fmt_age(dt: datetime) -> str:
    delta = datetime.now() - dt
    s = int(delta.total_seconds())
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h"
    return f"{s // 86400}d"

def fmt_cwd(cwd: str) -> str:
    home = str(Path.home())
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]
    parts = cwd.split("/")
    return "/".join(parts[-2:]) if len(parts) > 2 else cwd

def clean_summary(s: str) -> str:
    return s.replace("\n", " ").replace("\r", "").strip()

# ── ansi helpers ──────────────────────────────────────────────────────────────

DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
ORANGE = "\033[38;5;208m"   # claude
CYAN   = "\033[36m"         # codex

def _hide_cursor():
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

def _show_cursor():
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()

def _move_up(n: int):
    if n > 0:
        sys.stdout.write(f"\033[{n}A")

def _clear_line():
    sys.stdout.write("\033[2K\r")

def _read_key() -> str:
    """read a single keypress, handling arrow keys and j/k"""
    import tty, termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                if ch3 == "A": return "up"
                if ch3 == "B": return "down"
            return "esc"
        if ch == "j": return "down"
        if ch == "k": return "up"
        if ch in ("\r", "\n"): return "enter"
        if ch in ("q", "\x03"): return "quit"  # q or ctrl-c
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def pick(label: str, items: list, fmt_fn=None, max_visible: int = 20) -> Optional[int]:
    """
    inline picker. prints items, handles j/k/arrows, returns index or None.
    fmt_fn(index, item, is_selected) -> str for custom formatting.
    """
    if not items:
        return None
    if fmt_fn is None:
        fmt_fn = lambda i, item, sel: f"{'>' if sel else ' '} {item}"

    idx = 0
    scroll_offset = 0
    n = len(items)
    visible = min(n, max_visible)

    # print label + initial items
    sys.stdout.write(f"{BOLD}{label}{RESET}\n")
    total_lines = visible  # lines we'll redraw
    for i in range(visible):
        sys.stdout.write(fmt_fn(i, items[i], i == idx) + "\n")
    sys.stdout.flush()

    _hide_cursor()
    try:
        while True:
            key = _read_key()
            if key == "quit" or key == "esc":
                return None
            elif key == "enter":
                return idx
            elif key == "up":
                if idx > 0:
                    idx -= 1
                    if idx < scroll_offset:
                        scroll_offset = idx
            elif key == "down":
                if idx < n - 1:
                    idx += 1
                    if idx >= scroll_offset + visible:
                        scroll_offset = idx - visible + 1

            # redraw
            _move_up(total_lines)
            for vi in range(visible):
                _clear_line()
                ri = scroll_offset + vi
                sys.stdout.write(fmt_fn(ri, items[ri], ri == idx) + "\n")
            sys.stdout.flush()
    except (EOFError, KeyboardInterrupt):
        return None
    finally:
        _show_cursor()

# ── input helper ──────────────────────────────────────────────────────────────

def editable_input(label: str, default: str) -> str:
    """show a default value the user can edit or just press enter to accept"""
    sys.stdout.write(f"{BOLD}{label}{RESET} {DIM}(enter to accept, type to replace){RESET}\n")
    sys.stdout.write(f"  {DIM}{default}{RESET}\n")
    sys.stdout.write(f"  > ")
    sys.stdout.flush()
    try:
        val = input().strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return val if val else default

# ── subcommands ───────────────────────────────────────────────────────────────

def _scan_all() -> tuple[list[SessionData], list[SessionData]]:
    return parse_claude_sessions(), parse_codex_sessions()

def _fmt_size(b: int) -> str:
    if b < 1024:
        return f"{b}B"
    if b < 1024 * 1024:
        return f"{b // 1024}K"
    return f"{b / (1024 * 1024):.1f}M"

def cmd_scan() -> None:
    """show session discovery stats"""
    claude_sessions, codex_sessions = _scan_all()
    all_s = claude_sessions + codex_sessions

    print(f"{BOLD}handoff scan{RESET}\n")
    print(f"  {ORANGE}claude{RESET}  {len(claude_sessions)} sessions")
    print(f"  {CYAN}codex{RESET}   {len(codex_sessions)} sessions")
    print(f"  total   {len(all_s)} sessions\n")

    if not all_s:
        return

    # directories with most sessions
    dir_counts: dict[str, int] = {}
    branch_counts: dict[str, int] = {}
    total_bytes = 0
    for s in all_s:
        dir_counts[s.cwd] = dir_counts.get(s.cwd, 0) + 1
        if s.branch:
            branch_counts[s.branch] = branch_counts.get(s.branch, 0) + 1
        total_bytes += s.size_bytes

    top_dirs = sorted(dir_counts.items(), key=lambda x: -x[1])[:5]
    print(f"  total size  {_fmt_size(total_bytes)}")
    print(f"  directories {len(dir_counts)}\n")

    print(f"  {DIM}top directories:{RESET}")
    for d, c in top_dirs:
        print(f"    {fmt_cwd(d).ljust(30)}  {c} sessions")

    if branch_counts:
        top_branches = sorted(branch_counts.items(), key=lambda x: -x[1])[:5]
        print(f"\n  {DIM}top branches:{RESET}")
        for b, c in top_branches:
            print(f"    {b.ljust(30)}  {c} sessions")

def cmd_list(source_filter: str = "", limit: int = 20) -> None:
    """list sessions in tabular format"""
    claude_sessions, codex_sessions = _scan_all()
    all_s = claude_sessions + codex_sessions
    all_s.sort(key=lambda s: s.updated_at, reverse=True)

    if source_filter:
        all_s = [s for s in all_s if s.source == source_filter]

    cwd = os.getcwd()
    for s in all_s[:limit]:
        src_color = ORANGE if s.source == "claude" else CYAN
        src = s.source.ljust(6)
        age = fmt_age(s.updated_at).ljust(4)
        loc = fmt_cwd(s.cwd).ljust(24)
        br = (s.branch[:16].ljust(16) + "  ") if s.branch else ""
        here_mark = "*" if s.cwd == cwd else " "
        summary = clean_summary(s.summary or "")[:40]
        print(f"  {here_mark} {src_color}{src}{RESET}  {DIM}{age}{RESET}  {loc}  {br}{summary}")

    if len(all_s) > limit:
        print(f"\n  {DIM}({len(all_s)} total, showing {limit}. use --limit to see more){RESET}")

# ── interactive handoff (default) ─────────────────────────────────────────────

def cmd_handoff() -> None:
    print(f"{DIM}scanning sessions...{RESET}")
    claude_sessions = parse_claude_sessions()
    codex_sessions  = parse_codex_sessions()
    all_sessions = claude_sessions + codex_sessions
    all_sessions.sort(key=lambda s: s.updated_at, reverse=True)

    if not all_sessions:
        print("no sessions found")
        sys.exit(1)

    cwd = os.getcwd()
    here = [s for s in all_sessions if s.cwd == cwd]
    print(f"  {ORANGE}{len(claude_sessions)} claude{RESET}  {CYAN}{len(codex_sessions)} codex{RESET}\n")

    # ── step 1: scope ─────────────────────────────────────────────────────────
    scope_items = [
        f"this directory  {DIM}{fmt_cwd(cwd)}  ({len(here)}){RESET}",
        f"all sessions  {DIM}({len(all_sessions)}){RESET}",
    ]
    def fmt_scope(i, item, sel):
        ptr = f"{BOLD}>{RESET}" if sel else " "
        hl = BOLD if sel else ""
        return f"  {ptr} {hl}{item}{RESET}"

    si = pick("scope", scope_items, fmt_fn=fmt_scope)
    if si is None:
        sys.exit(0)

    pool = here if si == 0 else all_sessions
    if not pool:
        print(f"  no sessions in {cwd}")
        sys.exit(1)
    show_cwd = si == 1

    # ── step 2: session picker ────────────────────────────────────────────────
    display = pool[:40]

    def fmt_session(i, s, sel):
        ptr = f"{BOLD}>{RESET}" if sel else " "
        src_color = ORANGE if s.source == "claude" else CYAN
        src = s.source.ljust(6)
        age = fmt_age(s.updated_at).ljust(4)
        loc = (fmt_cwd(s.cwd).ljust(18) + "  ") if show_cwd else ""
        br = (DIM + s.branch[:12] + RESET + "  ") if s.branch else ""
        summary = clean_summary(s.summary or "(no summary)")[:48]
        if sel:
            return f"  {ptr} {BOLD}{src_color}{src}{RESET}  {BOLD}{age}{RESET}  {BOLD}{loc}{RESET}{br}{BOLD}{summary}{RESET}"
        return f"  {ptr} {src_color}{src}{RESET}  {DIM}{age}{RESET}  {loc}{br}{summary}"

    print()
    col_hdr = ("     " + "tool".ljust(6) + "  " + "age".ljust(4) + "  " +
               ("dir".ljust(18) + "  " if show_cwd else "") + "summary")
    sys.stdout.write(f"  {DIM}{col_hdr}{RESET}\n")

    si = pick("session", display, fmt_fn=fmt_session)
    if si is None:
        sys.exit(0)
    session = display[si]

    # ── step 3: load context ──────────────────────────────────────────────────
    print(f"\n  {DIM}loading {session.source} {session.id[:8]}...{RESET}")
    if session.source == "claude":
        load_claude_context(session)
    else:
        load_codex_context(session)

    target = "codex" if session.source == "claude" else "claude"
    print(f"  {session.source} -> {target}  {DIM}|  {len(session.messages)} msgs  {len(session.tool_calls)} tools  {len(session.thinking)} thinking{RESET}\n")

    # ── step 4: tier ──────────────────────────────────────────────────────────
    t1 = make_tier1(session)
    t2 = make_tier2(session)
    t3 = make_tier3(session)
    tok1, tok2, tok3 = est_tokens(t1), est_tokens(t2), est_tokens(t3)

    tier_items = [
        (t1, f"full     {tok1:>6,} tok  {DIM}conversation + tools + thinking + files{RESET}"),
        (t2, f"focused  {tok2:>6,} tok  {DIM}conversation + tools + files{RESET}"),
        (t3, f"minimal  {tok3:>6,} tok  {DIM}conversation only{RESET}"),
    ]

    def fmt_tier(i, item, sel):
        ptr = f"{BOLD}>{RESET}" if sel else " "
        hl = BOLD if sel else ""
        return f"  {ptr} {hl}{item[1]}{RESET}"

    ti = pick("context", tier_items, fmt_fn=fmt_tier)
    if ti is None:
        sys.exit(0)
    markdown = tier_items[ti][0]

    # ── step 5: handoff prompt ────────────────────────────────────────────────
    print()
    handoff_prompt = editable_input("handoff prompt", DEFAULT_HANDOFF_PROMPT)

    print()
    launch_with_context(target, session.source, markdown, session.cwd, handoff_prompt)

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    if not args:
        cmd_handoff()
        return

    cmd = args[0]

    if cmd in ("list", "ls"):
        source = ""
        limit = 20
        for i, a in enumerate(args[1:], 1):
            if a in ("claude", "codex"):
                source = a
            elif a == "--limit" and i < len(args) - 1:
                try:
                    limit = int(args[i + 1])
                except ValueError:
                    pass
            elif a.isdigit():
                limit = int(a)
        cmd_list(source_filter=source, limit=limit)

    elif cmd == "scan":
        cmd_scan()

    elif cmd in ("-h", "--help", "help"):
        print(f"{BOLD}handoff{RESET} — claude <-> codex session handoff\n")
        print(f"  {BOLD}handoff{RESET}              interactive session picker + handoff")
        print(f"  {BOLD}handoff list{RESET}          list sessions  {DIM}[claude|codex] [--limit N]{RESET}")
        print(f"  {BOLD}handoff scan{RESET}          show session discovery stats")
        print(f"  {BOLD}handoff help{RESET}          show this help")

    else:
        # unknown subcommand — treat as default
        cmd_handoff()

if __name__ == "__main__":
    main()
