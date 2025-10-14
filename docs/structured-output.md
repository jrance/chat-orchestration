**Structured Output (JSON)**

- Purpose: guarantee that agent responses are valid JSON when enabled via `structuredOutput` in the agent config.
- JSON mode: the engine requests provider JSON mode and passes a JSON Schema when available.
- Validation: model outputs are parsed and validated against the schema; on violation, the engine retries with a repair instruction up to `maxRepairAttempts`.
- Policies:
  - `onViolation`:
    - `RetryAndRepair` (default): retry with a repair hint until success or attempts exhausted.
    - `Refuse`: return 422 (engine raises an error) if invalid after final attempt.
    - `Ignore`: keep original content; apply post-processing.
- Post-processing (only for non-JSON content):
  - `postProcess.normalizeWhitespace` collapses whitespace.
  - `postProcess.ensureMarkdown` wraps plain text in a code block for readability.

**Provider Mapping**

- OpenAI Chat Completions:
  - With schema: attempts `response_format={"type":"json_schema","json_schema":{"name":"structured_output","schema":<schema>,"strict":true}}` and falls back to `{"type":"json_object"}` if unsupported.
  - Without schema: uses `{"type":"json_object"}`.
- Gemini:
  - Sets `generation_config.response_mime_type="application/json"` when JSON mode is enabled.
  - Attempts `generation_config.response_schema=<schema>` when provided (ignored silently if unsupported).

**SSE Behavior**

- For structured turns, streaming uses a non-stream provider call, then emits:
  1) `role`, 2) one `content.delta` with the compact JSON string, 3) `message.end` (and final `end`).
  This ensures clients receive validated, compact JSON.

**Example Schema**

```
{
  "type": "object",
  "properties": { "answer": { "type": "string" } },
  "required": ["answer"],
  "additionalProperties": false
}
```

**Configuration**

- `structuredOutput.enabled: true`
- `structuredOutput.schema: <JSON Schema>`
- `structuredOutput.onViolation: "RetryAndRepair" | "Refuse" | "Ignore"`
- `structuredOutput.maxRepairAttempts: <int>`
- `structuredOutput.postProcess.normalizeWhitespace: true|false`
- `structuredOutput.postProcess.ensureMarkdown: true|false`

