API Overview

- POST `/validate`: Validate a graph configuration. Query `returnNormalized` to include normalized config.
- POST `/compile`: Compile a graph and return orchestration metadata.
- POST `/execute`: Execute a run and return final messages.
- POST `/execute/stream`: Execute with SSE streaming events.
- GET `/healthz`: Health check.

Request/Response Schemas

- Validate
  - Request: raw graph JSON body.
  - Response: `{ valid, errors: [{code,message,path,severity}], warnings: [...], normalized? }`

- Compile
  - Request: raw graph JSON body.
  - Response: `{ ok, orchestration, agents: [{id,label,provider,model_id,tools}], tools: [{id,label,tool_id,version}], capabilities, notes }`

- Execute
  - Request: `{ config: Graph, input: { messages?: [ChatMessage], text?: string }, options?: { requestId?: string } }`
  - Response: `{ messages: [ChatMessage], finish_reason, usage, steps, request_id, metrics? }`

- Execute Stream (SSE)
  - Request: same as non-stream.
  - Response: `text/event-stream` with `data: {json}\n\n` frames. Event JSON:
    - `type`: `role | content.delta | tool_call.start | tool_call.delta | tool_call.end | message.end | error | end`
    - `role?`, `text_delta?`, `tool_call_index?`, `tool_call_id?`, `tool_name?`, `arguments_delta?`, `finish_reason?`, `usage?`, `request_id?`, `metrics?`, `error?`.
  - Safety: when a turn is blocked by safety (onBlock=Refuse), a normal assistant message is emitted with `finish_reason="content_filter"` and a short refusal. When onBlock=Warn, content proceeds and may be redacted.

Examples

- Validate
  - `curl -X POST http://localhost:8000/validate -H "Content-Type: application/json" --data @examples/configs/single_agent_with_tool.json`

- Compile
  - `curl -X POST http://localhost:8000/compile -H "Content-Type: application/json" --data @examples/configs/single_agent_with_tool.json`

- Execute
  - `curl -X POST http://localhost:8000/execute -H "Content-Type: application/json" -d "{\"config\":$(cat examples/configs/single_agent_with_tool.json),\"input\":{\"text\":\"hello\"}}"`

- Stream (Node/TS)
  - `new EventSource("http://localhost:8000/execute/stream", { method: 'POST', body: JSON.stringify({ config, input: { text: 'hello' } }) })`
  - Each `message` event contains a JSON payload as above.

Links

- Examples and curl/SSE walkthroughs: `docs/examples.md`
- Event ordering in real runs typically follows: `role → content.delta → ... → message.end → end`. Tool-call interleaving may emit tool events between content deltas when tools are enabled.
