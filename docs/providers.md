**Overview**
- Defines a provider-agnostic, OpenAI-spec interface for messages, tools, requests, responses, and streaming deltas.
- Implements `OpenAIProvider` using OpenAI v2 Chat Completions (sync + streaming) with function tools.

**Core Types**
- `ChatMessage`: `role` (`system|user|assistant|tool`), `content` (string or text parts), optional `tool_call_id` (for `tool` role), optional `tool_calls` (assistant outputs).
- `ToolSpec`: `name`, optional `description`, `parameters` (JSON Schema dict).
- `ToolCall`: `id`, `name`, `arguments_json` (raw JSON string from stream or response).
- `ChatRequest`: `model`, `messages`, optional `tools`, `tool_choice` (`auto|none|dict`), `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `json_mode_enabled`, optional `response_format_schema` (reserved), `metadata`, `timeout`.
- `ChatResponse`: `message`, `finish_reason`, `usage`, `raw` (provider response).
- `ChatChunk` streaming deltas: `type` in {`role`, `content.delta`, `tool_call.start`, `tool_call.delta`, `tool_call.end`, `message.end`} with fields for role, text, tool call info, finish_reason, usage.

**OpenAI Mapping**
- Messages → `[{role, content, tool_call_id?}]` (tool messages include `tool_call_id`).
- Tools → `[{type:"function", function:{name, description, parameters}}]`.
- Assistant `tool_calls[]` → `ToolCall[]` with `{id, function:{name, arguments}}` → `{id, name, arguments_json}`.
- Usage → `{prompt_tokens, completion_tokens, total_tokens}`.

**Usage**
- Non‑streaming:
  - Build a `ChatRequest` and call `OpenAIProvider().chat(req)`.
  - JSON mode: set `json_mode_enabled=True` to forward `response_format={"type":"json_object"}`.
- Streaming:
  - Iterate `OpenAIProvider().stream(req)` and handle `ChatChunk` cases to drive SSE.
  - Deltas include both text and function tool call assembly.

**Example**
- Non‑streaming:
  - `req = ChatRequest(model="gpt-4o-mini", messages=[ChatMessage(role="user", content="Hi")])`
  - `res = OpenAIProvider().chat(req)`
- Streaming:
  - `for chunk in OpenAIProvider().stream(req): ...` (handle `role`, `content.delta`, `tool_call.start`/`tool_call.delta`, `message.end`).

**Notes**
- Tool choice supports `"auto"`, `"none"`, or a full dict forwarded as‑is.
- Schema-driven JSON responses will be added in PR 12 with the Responses API.
- Streaming usage may be unavailable and is reported as `None`.

**Gemini Mapping**
- Roles:
  - Aggregate `system` messages into a single `system_instruction` string (omitted from history).
  - `user` → Gemini role `user` with `parts: [{text}]`.
  - `assistant` → Gemini role `model`. If tool calls are present, emit `parts: [{function_call:{name,args}}]`; otherwise text parts.
  - `tool` (tool result) → Gemini role `tool` with `parts: [{function_response:{name,response}}]`. The name is resolved by linking the tool message’s `tool_call_id` to the prior assistant tool call id.
- Tools: OpenAI function tools → `tools=[{"function_declarations":[{name, description, parameters}]}]`.
- Tool choice:
  - `"auto"` → `tool_config={function_calling_config:{mode:"AUTO"}}`
  - `"none"` → `tool_config={function_calling_config:{mode:"NONE"}}`
  - Specific selection (`{"type":"function","function":{"name":"x"}}`) → `tool_config={function_calling_config:{mode:"ANY", allowed_function_names:["x"]}}`.
- Generation config: `temperature`, `top_p`, `max_tokens→max_output_tokens`, `stop→stop_sequences`, `seed` (best‑effort), JSON mode → `response_mime_type:"application/json"`.
- Finish reasons: `STOP→stop`, `MAX_TOKENS→length`, safety/blocked/recitation→`content_filter`.
- Usage: `usage_metadata` mapped to `{prompt_tokens, completion_tokens, total_tokens}` when available.

**Gemini Usage**
- Non-streaming: `GeminiProvider().chat(req)` builds a stateless `GenerativeModel` per call with `system_instruction`, `tools`, and `tool_config`, then calls `generate_content`.
- Streaming: `GeminiProvider().stream(req)` yields `ChatChunk`s in order: role → content deltas → tool_call start/deltas → message.end with mapped finish reason. Usage may be present at end depending on model/region.

**Notes (Gemini)**
- System messages are aggregated; they do not appear in the history array.
- Tool result `function_response.name` is derived by linking to the referenced tool call id.
- Exact token usage reporting can vary by model/region; schema-driven JSON will arrive with response schema integration in PR 12.
