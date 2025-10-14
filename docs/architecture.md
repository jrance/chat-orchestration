# Architecture

This document outlines the main components and how they fit together.

## Compiler
- Defines and validates declarative, codeless agent graphs.
- Compiles YAML/JSON specs into LangGraph artifacts.

## Providers
- Pluggable interfaces for model providers (OpenAI, Google, etc.).
- Centralized auth, retries, and rate limiting.

## Tools
- Built-in tools (e.g., web search) and function calling.
- Sandboxed execution and timeouts.

## Engine
- Execution runtime using LangGraph.
- State management, persistence hooks, and observability.
 - Telemetry: per-run/turn/tool metrics aggregated at runtime with pluggable sinks (in-memory by default; optional OTLP export).

### Context Pipeline
- Per turn, each agent assembles a prompt from the shared transcript using a history window (see `docs/context.md`).
- A single system message is built from optional org preamble + system instructions + style guide with simple variable substitution.
- Normalized model parameters are forwarded to the provider on every turn.

### Structured Output Pipeline
- When `structuredOutput.enabled` is true on an agent, the engine:
  - Adds a concise system hint to force JSON-only output.
  - Requests provider JSON mode; passes a JSON Schema when available.
  - Parses and validates the model output; on violation, retries with a repair instruction up to `maxRepairAttempts`.
  - Produces a compact JSON string in the assistant message content upon success.
  - Applies post-processing options (`normalizeWhitespace`, `ensureMarkdown`) for non-JSON content.
- For streaming, uses a non-stream call and flushes a minimal SSE sequence for predictability.

### Safety Pipeline
- Before provider calls: apply safety pre-filter per agent safety config (`policyRef`, `onBlock`, `piiRedaction`, `promptInjectionDefense`).
- PII redaction: messages sent to providers are redacted when enabled; transcript remains canonical but redacted for outputs.
- Injection defense: score recent user inputs; if above threshold, `onBlock=Refuse` emits a refusal message and stops; `Warn` continues with a warning marker.
- After provider returns: redact assistant content before appending to transcript and emitting SSE.

### Engine Core
- State shape: `messages`, `metadata`, `tool_calls`, `steps`, plus optional buckets (`agent_cursor`, `concurrent`, `group`).
- Graph: `START → agent`, `agent → (tools | END)`, `tools → agent`.
- Tool loop: after an assistant message with `tool_calls`, execute attached tools deterministically with parameter overrides and timeouts; append tool results as `tool` messages with `tool_call_id` and continue.
- Provider resolver: maps model params to an `LLMProvider` (default supports `openai` and `gemini`).
- Streaming: supports single-turn token streaming without tool-calls. If a turn ends with tool-calls, the engine ends the streaming turn and continues non-streaming to execute tools.

## Server
- FastAPI app exposing orchestration APIs.
- Optional SSE/WebSocket streaming for tokens and events.
 - Request IDs are propagated and included on SSE events; metrics are surfaced when enabled.

### E2E Validation
- End-to-end tests exercise the server endpoints with deterministic provider/tool stubs across all orchestrations (single, sequential, handoff, concurrent, group chat), structured output, safety, SSE sequencing, and telemetry.
- See `tests/e2e/` and `docs/examples.md` for runnable examples and verification steps.

### Orchestrations
- Detection builds an agent-only graph (ignoring tool edges) and classifies it as:
  - Sequential: exactly one root and a single path covering all agents (outdegree ≤ 1).
  - Handoff: exactly one root and at least one agent with outdegree > 1.
  - Concurrency: a single fan-out stage with leaf branches, or a fan-in to a common next agent with merge strategies.
  - Group chat: the agent-only graph contains a strongly connected component (SCC) with size ≥ 2, or a self-loop.
- Builders construct a single LangGraph app with per-agent `agent:{id}` and `tools:{id}` nodes.
- Handoff uses a lightweight LLM router to select the next agent when multiple candidates exist, with deterministic fallback on parse errors.
- Group chat uses a `select_next → speak → tools → select_next` loop; next speaker is chosen by round-robin or a moderator policy using a low-temperature router with robust parsing.
