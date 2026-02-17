# miniClaw

A single-file AI agent (~540 lines) that demonstrates the core agent loop pattern used by OpenClaw — stripped down to the essentials.

It talks to a **local Ollama model** (no API keys needed) and implements the same fundamental cycle that powers OpenClaw:

```
LLM call → tool dispatch → feed results back → repeat
```

## Prerequisites

1. Install [Ollama](https://ollama.com)
2. Pull a model:
   ```sh
   ollama pull qwen2.5-coder:7b-instruct
   ```

## Usage

```sh
# Terminal REPL
bun miniClaw/mini-agent.ts

# Telegram bot + REPL (both run simultaneously)
TELEGRAM_BOT_TOKEN=<token> bun miniClaw/mini-agent.ts

# Use a different model
OLLAMA_MODEL=llama3:8b bun miniClaw/mini-agent.ts

# Enable debug logging
DEBUG=1 bun miniClaw/mini-agent.ts
```

## Tools

miniClaw has three built-in tools:

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents (truncated at 10k chars) |
| `run_command` | Execute a shell command (10s timeout) |
| `current_time` | Return the current date/time |

The model decides when to call them. It can call multiple tools in a single turn and chain tool results across iterations (up to 10 rounds).

## Telegram Setup

1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and follow the prompts to pick a name and username
3. BotFather gives you a token like `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`
4. Run miniClaw with the token:
   ```sh
   TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11 bun miniClaw/mini-agent.ts
   ```
5. The REPL starts in your terminal and the Telegram bot runs simultaneously — message the bot on Telegram or type in the terminal, both work at the same time

No pairing or allowlist required — unlike the full OpenClaw bot, miniClaw responds to anyone who messages it.

## Examples (partially tested)

### Reading a file

```
you> what's in package.json?
[tool: read_file({"path":"package.json"})]
[result: { "name": "openclaw", "version": "2025.1.15", ...]
The package.json defines the "openclaw" package at version 2025.1.15. It uses
ESM modules, has scripts for build/test/lint, and depends on ...
```

### Running a command

```
you> how much disk space is free?
[tool: run_command({"command":"df -h /"})]
[result: Filesystem  Size  Used  Avail  Use%  Mounted on  /dev/disk1s1  460G  280G  150G  66%  /]
You have about 150 GB free on your main disk (66% used).
```

### Multi-tool chaining

```
you> what files are in miniClaw/ and what time is it?
[tool: run_command({"command":"ls miniClaw/"})]
[result: mini-agent.ts README.md]
[tool: current_time({})]
[result: 2025-06-15T14:32:01.000Z]
There are two files in miniClaw/: mini-agent.ts and README.md.
The current time is 2025-06-15 at 2:32 PM UTC.
```

### Multi-step reasoning

```
you> find all TODO comments in the codebase
[tool: run_command({"command":"grep -r 'TODO' src/ --include='*.ts' -l"})]
[result: src/cli/commands.ts src/agents/pi-tools.ts]
[tool: run_command({"command":"grep -n 'TODO' src/cli/commands.ts src/agents/pi-tools.ts"})]
[result: src/cli/commands.ts:42: // TODO: add --verbose flag ...]
Found 2 files with TODOs:
1. src/cli/commands.ts:42 — add --verbose flag
2. ...
```

## How it maps to OpenClaw

The file is annotated with comments showing where each section corresponds to the full OpenClaw codebase:

| miniClaw | OpenClaw |
|----------|----------|
| `Tool` interface | `AgentTool` in `src/agents/tools/common.ts` |
| `tools[]` array | `createOpenClawCodingTools()` in `src/agents/pi-tools.ts` |
| `SYSTEM_PROMPT` | `buildEmbeddedSystemPrompt()` in `src/agents/pi-embedded-runner/system-prompt.ts` |
| `callOllama()` | Provider session via `@mariozechner/pi-ai` |
| `agentLoop()` | `session.prompt()` inside `src/agents/pi-embedded-runner/run/attempt.ts` |
| `startTelegram()` | `src/telegram/` (Grammy bot with pairing/allowlist) |
| `startRepl()` | `src/cli/` |
| In-memory `history[]` | `SessionManager` with JSONL persistence |
