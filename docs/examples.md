Examples

- Start the server:
  - `uvicorn codeless_orchestrator.server.app:app --reload`

- Execute (non-stream):
  - Bash/macOS: `curl -X POST http://localhost:8000/execute -H "Content-Type: application/json" -d "{\"config\":$(cat examples/configs/single_agent_with_tool.json),\"input\":{\"text\":\"hello\"}}"`
  - Windows PowerShell: `curl -Method POST http://localhost:8000/execute -Headers @{ 'Content-Type' = 'application/json' } -Body ("{`"config`":$((Get-Content examples/configs/single_agent_with_tool.json -Raw)),`"input`":{`"text`":`"hello`"}}")`

- Execute (SSE stream):
  - Bash/macOS: `curl -N -X POST http://localhost:8000/execute/stream -H "Content-Type: application/json" -d "{\"config\":$(cat examples/configs/single_agent_with_tool.json),\"input\":{\"text\":\"hello\"}}"`
  - Windows PowerShell: `curl -N -Method POST http://localhost:8000/execute/stream -Headers @{ 'Content-Type' = 'application/json' } -Body ("{`"config`":$((Get-Content examples/configs/single_agent_with_tool.json -Raw)),`"input`":{`"text`":`"hello`"}}")`

- Minimal Python SSE client:
  - `python examples/scripts/sse_client.py http://localhost:8000/execute/stream examples/configs/single_agent_with_tool.json hello`

Configs

- `examples/configs/single_agent_with_tool.json`: Single agent with built-in `tool:web-search` attached and parameter overrides.
- `examples/configs/sequential_two_agents.json`: Linear A → B orchestration.
- `examples/configs/handoff_branch.json`: A → {B, C} with LLM router choosing the next.
- `examples/configs/concurrent_fanout.json`: Fan-out to B and C branches, aggregated at join.
- `examples/configs/groupchat_moderator.json`: Moderator M coordinating participants X and Y.
- `examples/configs/structured_output.json`: Structured output enabled with a JSON schema.
- `examples/configs/safety_redaction.json`: Safety with PII redaction and injection defense.

What To Expect

- Non-stream responses: final transcript in `messages`, `steps`, optional `metrics` (token usage, timers).
- SSE stream: events `role → content.delta → message.end → end`; when telemetry is enabled, `message.end` includes `metrics`.
- Safety refusal (`onBlock=Refuse`): a refusal assistant message with `finish_reason=content_filter`.

