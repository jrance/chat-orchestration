**Config Schema**

- Strongly typed Pydantic v2 models for the Agent Builder JSON graph.
- Accepts camelCase from UI and snake_case for Python; exposes normalized snake_case fields.
- Cross-reference validation for edges, tool attachments, and duplicates.

**Top-level**
- `GraphConfig`: `meta`, `nodes`, `edges`
- `Meta`: `id`, `name`, `version`, `metadata.show_tool_nodes_on_canvas`
- `Node` (discriminated by `kind`)
  - `AgentNode` (`kind="agent.codeless"`): `label`, `data=AgentNodeData`
  - `ToolNode` (`kind="tool"`): `label`, `data=ToolNodeData`
- `Edge`: `id`, `from` (aliased to `from_`), `to`

**Agent Data**
- `system_instructions`, `style_guide`
- `model=ModelParams` (with alias normalization)
- `context=AgentContext`
- `tools=ToolsConfig`
- `structured_output=StructuredOutputConfig`
- `safety=SafetyConfig`
- `telemetry=TelemetryConfig`

**Normalization Mappings**
- Model params: `topP` → `top_p`, `maxTokens` → `max_tokens`, `jsonModeEnabled` → `json_mode_enabled`
- Context: `historyWindow` → `history_window`, `injectOrgPreamble` → `inject_org_preamble`
- Tools: `timeoutMs` → `timeout_ms`, `maxCallsPerTurn` → `max_calls_per_turn`
- Safety: `policyRef` → `policy_ref`, `onBlock` → `on_block`, `piiRedaction` → `pii_redaction`, `promptInjectionDefense` → `prompt_injection_defense`
- Structured output post-process: `normalizeWhitespace` → `normalize_whitespace`, `ensureMarkdown` → `ensure_markdown`
- Tool data: `toolId` → `tool_id`, `parameterOverrides` → `parameter_overrides`
- Edge: `from` → `from_` (Python attribute)

**Enumerations**
- `ModelParams.provider`: `openai` | `gemini`
- `AgentContext.history_window.mode`: `LastN`
- `StructuredOutputConfig.on_violation`: `RetryAndRepair` | `Refuse` | `Ignore`
- `SafetyConfig.on_block`: `Refuse` | `Warn`
- `ToolsConfig.policy`: `Auto` | `None` | `Required`

**Validation Rules**
- Field-level:
  - `temperature` in `[0.0, 2.0]`
  - `top_p` in `(0.0, 1.0]`
  - `max_tokens` > 0
  - `history_window.n` positive when mode is `LastN`
- Cross-reference:
  - Unique node IDs and edge IDs
  - Edges `from`/`to` refer to existing node IDs
  - Agent `tools.attached` IDs exist and refer to `tool` nodes
  - Warnings for negative `max_calls_per_turn` or `parallelism < 1`

**Example (camelCase)**
```
{
  "meta": {"id": "g1", "name": "Sample", "version": "1", "metadata": {"showToolNodesOnCanvas": true}},
  "nodes": [
    {"id": "tool1", "kind": "tool", "label": "Web Search", "data": {"name": "duckduckgo", "toolId": "duckduckgo-search", "version": "1.0"}},
    {"id": "agent1", "kind": "agent.codeless", "label": "Main Agent", "data": {
      "systemInstructions": "You are helpful.",
      "model": {"provider": "openai", "modelId": "gpt-4o-mini", "topP": 1.0, "jsonModeEnabled": false},
      "context": {"historyWindow": {"mode": "LastN", "n": 5}},
      "tools": {"attached": ["tool1"]},
      "safety": {"policyRef": "policy:v1"},
      "telemetry": {"labels": {}}
    }}
  ],
  "edges": [{"id": "e1", "from": "agent1", "to": "tool1"}]
}
```

**Normalized (snake_case)**
```
{
  "meta": {"id": "g1", "name": "Sample", "version": "1", "metadata": {"show_tool_nodes_on_canvas": true}},
  "nodes": [
    {"id": "tool1", "kind": "tool", "label": "Web Search", "data": {"name": "duckduckgo", "tool_id": "duckduckgo-search", "version": "1.0"}},
    {"id": "agent1", "kind": "agent.codeless", "label": "Main Agent", "data": {
      "system_instructions": "You are helpful.",
      "model": {"provider": "openai", "model_id": "gpt-4o-mini", "top_p": 1.0, "json_mode_enabled": false},
      "context": {"history_window": {"mode": "LastN", "n": 5}},
      "tools": {"attached": ["tool1"]},
      "safety": {"policy_ref": "policy:v1"},
      "telemetry": {"labels": {}}
    }}
  ],
  "edges": [{"id": "e1", "from": "agent1", "to": "tool1"}]
}
```

**Usage**
- `from codeless_orchestrator.config.loader import load_graph`
- `cfg, report = load_graph(data_or_path)`
- `report.has_errors()` to gate compilation; warnings are non-fatal.

See also: `docs/context.md` for prompt assembly, history windows, variable substitution, and model parameter pass-through.
See also: `docs/structured-output.md` for JSON mode, schema validation, and repair behavior.
