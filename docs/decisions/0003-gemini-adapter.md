**Decision**

Adopt an adapter for Google Gemini that conforms to the internal OpenAI‑spec provider interface. The `GeminiProvider` translates OpenAI‑shaped messages, tools, tool choices, and generation options into `google-generativeai` calls, and maps responses (including function calls and streaming deltas) back to the unified `ChatResponse`/`ChatChunk` types.

**Context**

- The orchestrator standardizes on an OpenAI Chat‑Completions‑like contract for messages, tools, and streaming.
- Gemini supports similar capabilities (system instruction, tools/function calling, JSON responses) with a different wire format.
- A focused adapter avoids leaking provider‑specific shapes into the core and keeps multi‑provider orchestration simple.

**Details**

- System messages aggregate into `system_instruction` and are excluded from the `contents` history.
- Messages map to Gemini `contents=[{role, parts}]` with `user|model|tool` roles and parts for `text`, `function_call`, and `function_response`.
- Tool declarations become `tools=[{"function_declarations":[...]}]` and tool choice maps to `tool_config.function_calling_config`.
- Generation options map to Gemini `generation_config`, including JSON mode via `response_mime_type: application/json`.
- Non‑streaming returns a single `ChatResponse` with mapped finish reason and usage; streaming yields `ChatChunk`s for role, text deltas, tool call start/deltas, and a final `message.end`.

**Tradeoffs**

- Minor impedance mismatch:
  - OpenAI `system` messages vs Gemini `system_instruction`.
  - Tool result naming requires linking `tool_call_id` to prior assistant calls to populate Gemini `function_response.name`.
  - Finish reason enums differ; we normalize to OpenAI‑like reasons.
- Usage metadata fields differ slightly and may vary by model/region; we map best‑effort.

**Consequences**

- A consistent provider contract enables shared tooling (routers, logging, SSE) across OpenAI and Gemini.
- Tests rely on simple stubs and do not require live API keys.

**Future**

- Add schema‑driven structured output using Gemini `response_schema` once the project integrates schema support (planned in PR 12).
