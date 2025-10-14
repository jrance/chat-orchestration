**Safety & Governance**

- Config fields (per agent):
  - `safety.policyRef`: selects policy defaults (e.g., `enterprise-v3`).
  - `safety.onBlock`: `Refuse` (default) or `Warn` for prompt‑injection actions.
  - `safety.piiRedaction`: enable PII redaction in provider prompts, transcripts, and SSE.
  - `safety.promptInjectionDefense`: enable basic prompt-injection detection.

- Policies (`runtime/policies.py`):
  - `enterprise-v3`: `injection_threshold=0.7`, `redact_pii_default=true`.

- PII Redaction (`runtime/safety.py`):
  - Email, phone (US/E.164), SSN, IPv4, and credit cards (Luhn) are replaced by tokens like `[REDACTED:EMAIL]`.
  - Redaction applies to provider-bound messages, assistant messages, and SSE payloads.

- Prompt‑Injection Defense:
  - Heuristic scoring based on phrases such as “ignore previous instructions”, “developer mode”, “print system prompt”, etc.
  - If score ≥ threshold:
    - `onBlock=Refuse`: a refusal message is emitted; no provider call occurs for that turn.
    - `onBlock=Warn`: proceed but annotate metadata and continue.

- Tools:
  - Arguments/results are redacted for logs/transcripts/SSE by default; actual tool execution receives unmodified arguments.

- SSE Behavior:
  - Redacted `text_delta`/`arguments_delta` when enabled.
  - On refusal, emits role → content.delta (refusal) → message.end with `finish_reason="content_filter"`.

