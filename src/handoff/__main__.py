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
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── token estimation ──────────────────────────────────────────────────────────

def est_tokens(text: str) -> int:
    """rough token estimate: chars / 4"""
    return max(1, len(text) // 4)

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
    token_usage: tuple[int, int] = (0, 0)  # (input, output)

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

def _is_real_user_message(text: str) -> bool:
    skip = ("<environment_details>", "<system>", "# AGENTS.md", "<permissions", "<environment_context>")
    return bool(text.strip()) and not any(text.startswith(s) for s in skip)

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
                if not first_user and msg.get("type") == "event_msg":
                    payload = msg.get("payload", {})
                    if payload.get("type") == "user_message":
                        first_user = (payload.get("message") or "")[:100]
                if not first_user and msg.get("type") == "message" and msg.get("role") == "user":
                    first_user = str(msg.get("content", ""))[:100]

            stat = jsonl_file.stat()
            sessions.append(SessionData(
                id=sid,
                source="codex",
                cwd=cwd,
                summary=first_user.strip(),
                updated_at=datetime.fromtimestamp(stat.st_mtime),
                path=jsonl_file,
                model=model,
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

def build_tool_activity_section(tool_calls: list[ToolCall]) -> str:
    if not tool_calls:
        return ""
    # group by name
    groups: dict[str, list[str]] = {}
    for tc in tool_calls:
        groups.setdefault(tc.name, []).append(tc.summary)
    lines = ["## Tool Activity", ""]
    for name, summaries in groups.items():
        samples = " | ".join(f"`{s}`" for s in summaries[:3])
        lines.append(f"- **{name}** (x{len(summaries)}): {samples}")
    return "\n".join(lines)

def build_conversation_section(messages: list[Message], full: bool = False) -> str:
    if not messages:
        return ""
    lines = ["## Recent Conversation", ""]
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"### {role}")
        lines.append("")
        content = msg.content if full else (msg.content[:800] + ("..." if len(msg.content) > 800 else ""))
        lines.append(content)
        lines.append("")
    return "\n".join(lines)

def build_thinking_section(thinking: list[ThinkingBlock]) -> str:
    if not thinking:
        return ""
    lines = ["## Key Decisions", ""]
    for t in thinking[:5]:
        lines.append(f"- {t.text}")
    return "\n".join(lines)

def build_header(session: SessionData) -> str:
    source_label = "Claude Code" if session.source == "claude" else "OpenAI Codex"
    lines = [
        "# Session Handoff Context",
        "",
        "## Session Overview",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| **Source** | {source_label} |",
        f"| **Session ID** | `{session.id[:12]}` |",
        f"| **Working Directory** | `{session.cwd}` |",
    ]
    if session.model:
        lines.append(f"| **Model** | {session.model} |")
    lines.append(f"| **Last Active** | {session.updated_at.strftime('%Y-%m-%d %H:%M')} |")
    if session.token_usage != (0, 0):
        inp, out = session.token_usage
        lines.append(f"| **Tokens Used** | {inp:,} in / {out:,} out |")
    if session.files_modified:
        lines.append(f"| **Files Modified** | {len(session.files_modified)} |")
    lines.append(f"| **Messages** | {len(session.messages)} |")
    lines.append("")

    if session.summary:
        lines += ["## Summary", "", f"> {session.summary[:200]}", ""]

    return "\n".join(lines)

def build_footer() -> str:
    return "\n---\n\n**You are continuing this session. Pick up exactly where it left off — review the context above and keep going.**"

def build_files_section(files: list[str]) -> str:
    if not files:
        return ""
    lines = ["## Files Modified", ""]
    for f in files:
        lines.append(f"- `{f}`")
    return "\n".join(lines)

def make_tier1(session: SessionData, n_messages: int = 20) -> str:
    """full: header + tool activity + thinking + conversation (last N, full text) + files"""
    parts = [
        build_header(session),
        build_tool_activity_section(session.tool_calls),
        build_thinking_section(session.thinking),
        build_conversation_section(session.messages[-n_messages:], full=True),
        build_files_section(session.files_modified),
        build_footer(),
    ]
    return "\n\n".join(p for p in parts if p)

def make_tier2(session: SessionData, n_messages: int = 20) -> str:
    """focused: header + tool activity + conversation (last N, full text) + files — no thinking"""
    parts = [
        build_header(session),
        build_tool_activity_section(session.tool_calls),
        build_conversation_section(session.messages[-n_messages:], full=True),
        build_files_section(session.files_modified),
        build_footer(),
    ]
    return "\n\n".join(p for p in parts if p)

def make_tier3(session: SessionData, n_messages: int = 20) -> str:
    """minimal: header + conversation only (last N, full text)"""
    parts = [
        build_header(session),
        build_conversation_section(session.messages[-n_messages:], full=True),
        build_footer(),
    ]
    return "\n\n".join(p for p in parts if p)

# ── launch ────────────────────────────────────────────────────────────────────

def launch_with_context(target: str, markdown: str, cwd: str) -> None:
    # write handoff file to project dir
    handoff_path = Path(cwd) / ".handoff.md" if cwd else Path.cwd() / ".handoff.md"
    try:
        handoff_path.write_text(markdown)
        print(f"wrote handoff file: {handoff_path}")
    except Exception as e:
        print(f"warning: could not write handoff file: {e}")

    intro = f"I'm continuing a coding session. Here's the full context:\n\n---\n\n{markdown}"

    if target == "codex":
        # codex takes the prompt as a positional argument
        cmd = ["codex", intro]
    elif target == "claude":
        cmd = ["claude", "-p", intro]
    else:
        print(f"unknown target: {target}")
        sys.exit(1)

    work_dir = cwd if cwd and Path(cwd).exists() else str(Path.cwd())
    print(f"launching {target} in {work_dir}...\n")

    try:
        subprocess.run(cmd, cwd=work_dir)
    except FileNotFoundError:
        print(f"error: '{target}' binary not found in PATH")
        sys.exit(1)

# ── display helpers ───────────────────────────────────────────────────────────

def fmt_age(dt: datetime) -> str:
    delta = datetime.now() - dt
    s = int(delta.total_seconds())
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"

def fmt_cwd(cwd: str) -> str:
    home = str(Path.home())
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]
    parts = cwd.split("/")
    return "/".join(parts[-2:]) if len(parts) > 2 else cwd

def session_label(s: SessionData) -> str:
    tag = f"[{s.source}]".ljust(8)
    age = fmt_age(s.updated_at).ljust(8)
    cwd = fmt_cwd(s.cwd).ljust(22)
    summary = (s.summary or "(no summary)")[:52]
    return f"{tag}  {age}  {cwd}  {summary}"

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import questionary
    from questionary import Style

    theme = Style([
        ("qmark",        "fg:#6c71c4 bold"),
        ("question",     "bold"),
        ("answer",       "fg:#268bd2 bold"),
        ("pointer",      "fg:#6c71c4 bold"),
        ("highlighted",  "fg:#268bd2 bold"),
        ("selected",     "fg:#2aa198"),
        ("separator",    "fg:#586e75"),
        ("instruction",  "fg:#586e75"),
        ("text",         ""),
    ])

    print("scanning sessions...")
    claude_sessions = parse_claude_sessions()
    codex_sessions = parse_codex_sessions()

    all_sessions = claude_sessions + codex_sessions
    all_sessions.sort(key=lambda s: s.updated_at, reverse=True)

    if not all_sessions:
        print("no sessions found")
        sys.exit(1)

    display = all_sessions[:40]
    print(f"found {len(claude_sessions)} claude + {len(codex_sessions)} codex sessions")

    # session picker
    choices = [questionary.Choice(title=session_label(s), value=s) for s in display]
    session = questionary.select(
        "pick a session:",
        choices=choices,
        style=theme,
        use_shortcuts=False,
        use_indicator=True,
    ).ask()

    if session is None:
        sys.exit(0)

    # load full context
    print(f"\nloading {session.source} session {session.id[:12]}...")
    if session.source == "claude":
        load_claude_context(session)
    else:
        load_codex_context(session)

    target = "codex" if session.source == "claude" else "claude"
    print(f"{session.source} -> {target}  |  {len(session.messages)} messages  {len(session.tool_calls)} tool calls  {len(session.thinking)} thinking blocks")

    # build tiers
    t1 = make_tier1(session)
    t2 = make_tier2(session)
    t3 = make_tier3(session)

    tok1 = est_tokens(t1)
    tok2 = est_tokens(t2)
    tok3 = est_tokens(t3)

    tier_choices = [
        questionary.Choice(
            title=f"full      {tok1:>6,} tokens  (conversation + tools + thinking + files)",
            value=t1,
        ),
        questionary.Choice(
            title=f"focused   {tok2:>6,} tokens  (conversation + tools + files)",
            value=t2,
        ),
        questionary.Choice(
            title=f"minimal   {tok3:>6,} tokens  (conversation only)",
            value=t3,
        ),
    ]

    markdown = questionary.select(
        "pick context tier:",
        choices=tier_choices,
        style=theme,
        use_indicator=True,
    ).ask()

    if markdown is None:
        sys.exit(0)

    launch_with_context(target, markdown, session.cwd)

if __name__ == "__main__":
    main()
