**Context & Prompt Assembly**

- History windows: `LastN` keeps the last N non-system messages by role in {`user`,`assistant`,`tool`}, preserving order. System messages are not counted in N and are not persisted in the transcript by the engine; instead, a single system message is assembled per agent turn.
- Organization preamble: If enabled via `AgentContext.inject_org_preamble`, a preamble is injected ahead of the agent’s `system_instructions` and `style_guide`. The server provides it via `ORG_PREAMBLE_FILE` (UTF‑8) or `ORG_PREAMBLE` at startup.
- Variables: Simple templates based on Python’s `str.format_map`. Missing keys are preserved as `{key}` (no exceptions). Merge precedence: engine defaults (reserved for future) < `context.vars` from config < request `vars` on the Execute API.
- Assembly order per agent turn: `[system?] + windowed history`. The system message is a concatenation of `[org_preamble?, system_instructions?, style_guide?]` with two newlines between non-empty parts, after variable substitution.

**Model Parameters Normalization**

- Normalized fields in `ModelParams` are forwarded to providers each turn: `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `json_mode_enabled`.
- Tool policy mapping for provider requests: `Auto` → `tool_choice="auto"`, `None` → `tool_choice="none"` (a future PR may change `Required`).

**Examples**

- Config snippet:
  - `context.historyWindow: { mode: "LastN", n: 2 }`
  - `context.injectOrgPreamble: true`
  - `context.vars: { tone: "formal" }`
  - `systemInstructions: "Hello {dept}"`
  - `styleGuide: "Use {tone} tone."`
- Execute input (request-level vars): `{ "vars": { "dept": "Legal" } }`
- Effective system message: `"<ORG_PREAMBLE>\n\nHello Legal\n\nUse formal tone."`

