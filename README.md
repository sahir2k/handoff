# handoff

session handoff tool for claude code and openai codex. picks up where one ai coding session left off and hands the full conversation context to the other tool, so you can switch between claude and codex without losing track of what was being worked on.

## install

```
uv tool install git+https://github.com/sahir2k/handoff
```

## usage

- `handoff` — interactive picker to select a session and hand it off to the other tool
- `handoff list` — list recent sessions across claude and codex
- `handoff scan` — show session discovery stats (counts, top dirs, branches, total size)
