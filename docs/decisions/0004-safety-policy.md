Decision 0004: Safety Policy

Context
- Add light-weight safety with PII redaction and prompt-injection defense.

Decision
- Start with heuristic detectors and a default `enterprise-v3` policy with an adjustable threshold and default redaction enabled.
- Non-destructive defaults: redact logs/transcripts/SSE; do not mutate tool execution arguments by default.
- Allow `onBlock` to control behavior: `Refuse` vs `Warn`.

Rationale
- Keeps runtime deterministic and simple while enabling governance hooks.
- Leaves room for external DLP or LLM guardrails later without breaking API.

Consequences
- False positives may occur on heuristics; thresholds and `onBlock` trade off safety vs. friction.
- Provider models still need robust prompts; this layer complements, not replaces, model-side safeguards.

