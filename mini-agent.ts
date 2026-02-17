/**
 * Mini OpenClaw Agent Loop (Ollama Edition)
 *
 * A standalone script that makes the core agent pattern explicit:
 *   LLM call → tool dispatch → feed results → repeat
 *
 * Uses a local Ollama model (qwen2.5-coder:7b-instruct) — no API keys needed.
 * Talks directly to Ollama's /api/chat endpoint with streaming and tool calling.
 *
 * Prerequisites:
 *   1. Install Ollama: https://ollama.com
 *   2. Pull the model: ollama pull qwen2.5-coder:7b-instruct
 *
 * Usage:
 *   bun miniClaw/mini-agent.ts                          # terminal REPL
 *   TELEGRAM_BOT_TOKEN=123:abc bun miniClaw/mini-agent.ts  # Telegram bot
 *
 * Then chat interactively. Try:
 *   "what time is it?"              → calls current_time
 *   "read package.json"             → calls read_file
 *   "list files in src/"            → calls run_command with `ls src/`
 *   "what files are in miniClaw/ and what time is it?"  → calls both tools
 *   "quit"                          → exits
 */

import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { networkInterfaces } from "node:os";
import { createInterface } from "node:readline";
import { Agent } from "undici";
import { Bot } from "grammy";

// ─── Section 1: Types ────────────────────────────────────────────────────────
//
// OpenClaw equivalent: `AgentTool` from pi-agent-core (see src/agents/tools/common.ts).
// Our version is intentionally minimal — just a name, description, JSON Schema
// for the input, and an execute function.

interface Tool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  execute: (input: Record<string, unknown>) => string;
}

// Ollama chat API types (subset we need)
interface OllamaMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_calls?: OllamaToolCall[];
}

interface OllamaToolCall {
  function: {
    name: string;
    arguments: Record<string, unknown>;
  };
}

// Ollama streaming response chunk
interface OllamaStreamChunk {
  message: OllamaMessage;
  done: boolean;
}

// Ollama non-streaming response
interface OllamaChatResponse {
  message: OllamaMessage;
  done: boolean;
}

// ─── Section 2: Tool Definitions ─────────────────────────────────────────────
//
// OpenClaw equivalent: `createOpenClawCodingTools()` in src/agents/pi-tools.ts
// builds ~20+ tools (file read/write, shell, browser, search, etc.).
// We define just three to show the pattern.

const tools: Tool[] = [
  {
    name: "read_file",
    description:
      "Read the contents of a file at the given path. Returns the file content as a string, truncated to 10,000 characters.",
    parameters: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "Absolute or relative file path to read",
        },
      },
      required: ["path"],
    },
    execute: (input) => {
      const path = input.path as string;
      try {
        const content = readFileSync(path, "utf-8");
        return content.length > 10_000
          ? content.slice(0, 10_000) + "\n\n... (truncated at 10,000 chars)"
          : content;
      } catch (err) {
        return `Error reading file: ${(err as Error).message}`;
      }
    },
  },
  {
    name: "run_command",
    description:
      "Run a shell command and return its stdout and stderr. Has a 10-second timeout.",
    parameters: {
      type: "object",
      properties: {
        command: {
          type: "string",
          description: "The shell command to execute",
        },
      },
      required: ["command"],
    },
    execute: (input) => {
      const command = input.command as string;
      try {
        const output = execSync(command, {
          timeout: 10_000,
          encoding: "utf-8",
          stdio: ["pipe", "pipe", "pipe"],
        });
        return output || "(no output)";
      } catch (err) {
        const e = err as { stdout?: string; stderr?: string; message: string };
        return `stdout: ${e.stdout ?? ""}\nstderr: ${e.stderr ?? ""}\nerror: ${e.message}`;
      }
    },
  },
  {
    name: "current_time",
    description: "Returns the current date and time in ISO 8601 format.",
    parameters: {
      type: "object",
      properties: {},
    },
    execute: () => new Date().toISOString(),
  },
];

// ─── Section 3: Ollama Setup ─────────────────────────────────────────────────
//
// OpenClaw equivalent: provider setup happens in src/agents/pi-embedded-runner/run/attempt.ts
// where a pi-ai session is created with the configured provider (Anthropic, OpenAI, etc.).
// Here we talk directly to Ollama's local HTTP API — no SDK needed.

const OLLAMA_BASE = process.env.OLLAMA_HOST ?? "http://localhost:11434";
const MODEL = process.env.OLLAMA_MODEL ?? "qwen2.5-coder:7b-instruct";
const DEBUG = process.env.DEBUG === "1";

// OpenClaw equivalent: `buildEmbeddedSystemPrompt()` in
// src/agents/pi-embedded-runner/system-prompt.ts — assembles a large system prompt
// from agent config, persona, tool descriptions, and context.
const SYSTEM_PROMPT =
  "You are a helpful coding assistant. You have access to tools for reading files, running commands, and checking the time. Use them when appropriate to answer the user's questions. Be concise.";

// Convert our Tool[] to Ollama's tool format (OpenAI-style function calling)
const ollamaTools = tools.map((t) => ({
  type: "function" as const,
  function: {
    name: t.name,
    description: t.description,
    parameters: t.parameters,
  },
}));

// ─── Section 4: The Agent Loop ───────────────────────────────────────────────
//
// This is the heart of every AI agent. The pattern:
//   1. Send conversation history + tools to the LLM
//   2. Stream the response to the terminal in real-time
//   3. If the LLM wants to call tools (response contains tool_calls):
//      a. Execute each tool call
//      b. Append results as "tool" role messages
//      c. Go back to step 1
//   4. If the LLM is done (no tool_calls):
//      return — the answer is already printed via streaming
//
// OpenClaw equivalent: this entire loop is hidden inside `session.prompt()` from
// @mariozechner/pi-ai. When you call `activeSession.prompt(effectivePrompt)` at
// src/agents/pi-embedded-runner/run/attempt.ts:815, pi-ai internally runs this
// exact loop — calling the LLM, dispatching tools, feeding results back — until
// the model emits a final text response.

function log(...args: unknown[]) {
  if (DEBUG) console.log("\x1b[90m[debug]\x1b[0m", ...args);
}

// ─── Text-based tool call parser (fallback) ──────────────────────────────────
//
// Some models (like qwen2.5-coder) emit tool calls as plain JSON text instead
// of using the structured tool_calls field. This parser extracts them from
// the text content so the agent loop can still dispatch them.
//
// Looks for patterns like:
//   {"name": "current_time", "arguments": {}}
//   {"name": "read_file", "arguments": {"path": "package.json"}}

const toolNames = new Set(tools.map((t) => t.name));

function parseToolCallsFromText(text: string): OllamaToolCall[] {
  const trimmed = text.trim();
  const results: OllamaToolCall[] = [];

  // Try to parse the entire text as a single tool call or array of tool calls
  try {
    const parsed = JSON.parse(trimmed);

    // Array of tool calls
    if (Array.isArray(parsed)) {
      for (const item of parsed) {
        if (item.name && toolNames.has(item.name)) {
          results.push({ function: { name: item.name, arguments: item.arguments ?? {} } });
        }
      }
      if (results.length > 0) return results;
    }

    // Single tool call object: {"name": "...", "arguments": {...}}
    if (parsed.name && toolNames.has(parsed.name)) {
      return [{ function: { name: parsed.name, arguments: parsed.arguments ?? {} } }];
    }
  } catch {
    // Not valid JSON as a whole — try regex extraction below
  }

  // Regex fallback: find JSON objects that look like tool calls embedded in text
  const jsonPattern = /\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}/g;
  let match;
  while ((match = jsonPattern.exec(trimmed)) !== null) {
    const name = match[1];
    if (toolNames.has(name)) {
      try {
        const args = JSON.parse(match[2]);
        results.push({ function: { name, arguments: args } });
      } catch {
        // skip malformed arguments
      }
    }
  }

  return results;
}

// ─── Call Ollama (non-streaming) ─────────────────────────────────────────────
//
// Used for tool-calling turns. Non-streaming mode gives some models a better
// chance of producing structured tool_calls. We also apply the text fallback
// parser if the model still emits tool calls as plain text.

async function callOllama(messages: OllamaMessage[]): Promise<{
  content: string;
  toolCalls: OllamaToolCall[];
}> {
  const payload = {
    model: MODEL,
    messages: [{ role: "system", content: SYSTEM_PROMPT }, ...messages],
    tools: ollamaTools,
    stream: false,
  };
  log(">>> request (non-streaming)");

  const res = await fetch(`${OLLAMA_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Ollama error ${res.status}: ${body}`);
  }

  const data: OllamaChatResponse = await res.json();
  const content = data.message.content ?? "";
  let toolCalls = data.message.tool_calls ?? [];

  log("<<< response content:", content.slice(0, 200));
  log("<<< structured tool_calls:", toolCalls.length);

  // Fallback: if no structured tool_calls, try parsing from text
  if (toolCalls.length === 0 && content.trim()) {
    toolCalls = parseToolCallsFromText(content);
    if (toolCalls.length > 0) {
      log("<<< parsed tool_calls from text:", JSON.stringify(toolCalls));
    }
  }

  return { content, toolCalls };
}

// ─── Stream Ollama response ──────────────────────────────────────────────────
//
// Used for the final text response (no tools). Streams tokens to stdout.

async function streamOllama(messages: OllamaMessage[]): Promise<string> {
  const payload = {
    model: MODEL,
    messages: [{ role: "system", content: SYSTEM_PROMPT }, ...messages],
    tools: ollamaTools,
    stream: true,
  };
  log(">>> request (streaming)");

  const res = await fetch(`${OLLAMA_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Ollama error ${res.status}: ${body}`);
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let fullContent = "";
  let toolCalls: OllamaToolCall[] = [];
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop()!;

    for (const line of lines) {
      if (!line.trim()) continue;
      const chunk: OllamaStreamChunk = JSON.parse(line);

      if (chunk.message.content) {
        process.stdout.write(chunk.message.content);
        fullContent += chunk.message.content;
      }
      if (chunk.message.tool_calls?.length) {
        toolCalls = chunk.message.tool_calls;
      }
    }
  }

  // Check for tool calls (structured or text-parsed) even in streaming mode
  if (toolCalls.length === 0 && fullContent.trim()) {
    toolCalls = parseToolCallsFromText(fullContent);
  }

  // If we got tool calls, store them so the caller knows
  if (toolCalls.length > 0) {
    log("<<< streaming response had tool_calls:", toolCalls.length);
  }

  return fullContent;
}

// ─── The Agent Loop ──────────────────────────────────────────────────────────

const MAX_TOOL_ITERATIONS = 10;

async function agentLoop(
  userMessage: string,
  history: OllamaMessage[],
): Promise<string> {
  history.push({ role: "user", content: userMessage });
  log(">>> user:", userMessage);

  let iteration = 0;

  while (true) {
    iteration++;
    log(`--- loop iteration ${iteration} ---`);

    if (iteration > MAX_TOOL_ITERATIONS) {
      const bail = "I've hit the tool-call limit. Here's what I have so far.";
      history.push({ role: "assistant", content: bail });
      return bail;
    }

    // Step 1: Call Ollama (non-streaming for reliable tool detection)
    const { content, toolCalls } = await callOllama(history);

    // Step 2: Append assistant response to history
    const assistantMessage: OllamaMessage = { role: "assistant", content };
    if (toolCalls.length > 0) {
      assistantMessage.tool_calls = toolCalls;
      // If the model emitted tool calls as text, clear content to avoid
      // showing raw JSON to the user
      if (!assistantMessage.tool_calls?.length) {
        assistantMessage.content = "";
      }
    }
    history.push(assistantMessage);
    log("<<< assistant:", JSON.stringify(assistantMessage).slice(0, 300));

    // Step 3: No tool calls — done
    if (toolCalls.length === 0) {
      // Print the response (wasn't streamed since we used non-streaming mode)
      process.stdout.write(content);
      log("<<< no tool calls — returning final text");
      return content;
    }

    // Step 4: Dispatch tool calls
    for (const toolCall of toolCalls) {
      const { name, arguments: args } = toolCall.function;
      const tool = tools.find((t) => t.name === name);

      if (!tool) {
        console.log(`\n[unknown tool: ${name}]`);
        history.push({ role: "tool", content: `Unknown tool: ${name}` });
        continue;
      }

      console.log(`[tool: ${name}(${JSON.stringify(args)})]`);
      const result = tool.execute(args);
      const preview = result.length > 200 ? result.slice(0, 200) + "..." : result;
      console.log(`[result: ${preview}]`);

      // Step 5: Feed tool result back
      history.push({ role: "tool", content: result });
      log(">>> tool result pushed, length:", result.length);
    }

    log("--- looping back with tool results ---");
    // Loop: LLM will see the tool results and either call more tools or respond
  }
}

// ─── Section 5: Telegram Channel ─────────────────────────────────────────────
//
// OpenClaw equivalent: src/telegram/ wraps the same session.prompt() call with
// Grammy's bot.on("message") handler. Each chat gets its own conversation
// history so multiple users can talk to the bot concurrently.

const TELEGRAM_MAX_LENGTH = 4096;

function splitMessage(text: string): string[] {
  if (text.length <= TELEGRAM_MAX_LENGTH) return [text];
  const chunks: string[] = [];
  for (let i = 0; i < text.length; i += TELEGRAM_MAX_LENGTH) {
    chunks.push(text.slice(i, i + TELEGRAM_MAX_LENGTH));
  }
  return chunks;
}

function getInterfaceAddress(name: string): string {
  const nets = networkInterfaces();
  const iface = nets[name];
  if (!iface) throw new Error(`Network interface ${name} not found`);
  const ipv4 = iface.find((i) => i.family === "IPv4" && !i.internal);
  if (!ipv4) throw new Error(`No IPv4 address found on ${name}`);
  return ipv4.address;
}

async function startTelegram(token: string) {
  const TELEGRAM_INTERFACE = process.env.TELEGRAM_INTERFACE ?? "en1";
  const localAddress = getInterfaceAddress(TELEGRAM_INTERFACE);
  console.log(`Binding Telegram traffic to ${TELEGRAM_INTERFACE} (${localAddress})`);

  const dispatcher = new Agent({ connect: { localAddress } });
  const bot = new Bot(token, {
    client: {
      baseFetchConfig: { dispatcher } as RequestInit,
    },
  });
  const chatHistories = new Map<number, OllamaMessage[]>();

  console.log(`Mini OpenClaw Telegram Bot (Ollama: ${MODEL})`);
  console.log(`Debug: ${DEBUG ? "ON (DEBUG=1)" : "off (set DEBUG=1 to enable)"}`);

  bot.command("start", (ctx) => ctx.reply("Hi! I'm a mini coding assistant powered by Ollama. Send me a message."));

  bot.on("message:text", async (ctx) => {
    const chatId = ctx.chat.id;
    const history = chatHistories.get(chatId) ?? [];

    try {
      const response = await agentLoop(ctx.message.text, history);
      chatHistories.set(chatId, history);

      for (const chunk of splitMessage(response)) {
        await ctx.reply(chunk);
      }
    } catch (err) {
      await ctx.reply(`Error: ${(err as Error).message}`);
    }
  });

  console.log("Starting Telegram bot (long-polling)...");
  await bot.start({
    onStart: (botInfo) =>
      console.log(`Bot @${botInfo.username} is live — send it a message!`),
  });
}

// ─── Section 6: Interactive REPL ─────────────────────────────────────────────
//
// OpenClaw equivalent: the CLI REPL lives in src/cli/ and routes user input
// through commands, routing, and eventually into the agent loop. The web UI
// and messaging channels (Telegram, Discord, etc.) each have their own input
// paths that all converge on the same `session.prompt()` call.

async function startRepl() {
  console.log(`Mini OpenClaw Agent Loop (Ollama: ${MODEL})`);
  console.log(`Debug: ${DEBUG ? "ON (DEBUG=1)" : "off (set DEBUG=1 to enable)"}`);
  console.log('Type a message to chat, "quit" to exit.\n');

  // Conversation history persists across turns — this is what gives the agent memory.
  //
  // OpenClaw equivalent: SessionManager persists this to JSONL files so conversations
  // survive across process restarts. We just keep it in memory.
  const conversationHistory: OllamaMessage[] = [];

  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const prompt = () =>
    new Promise<string>((resolve) => {
      rl.question("you> ", resolve);
    });

  while (true) {
    const input = await prompt();

    if (input.trim().toLowerCase() === "quit") {
      console.log("Goodbye!");
      rl.close();
      break;
    }

    if (!input.trim()) continue;

    // Run the agent loop — this may make multiple LLM calls if tools are involved
    try {
      await agentLoop(input, conversationHistory);
    } catch (err) {
      console.error(`\nError: ${(err as Error).message}`);
      console.error("Is Ollama running? Try: ollama serve");
    }
    console.log(); // newline after streamed response
  }
}

// ─── Section 7: Entry Point ──────────────────────────────────────────────────

async function main() {
  const telegramToken = process.env.TELEGRAM_BOT_TOKEN;
  if (telegramToken) {
    startTelegram(telegramToken); // fire-and-forget (long-polls in background)
  }
  await startRepl(); // always run the REPL
}

main().catch(console.error);
