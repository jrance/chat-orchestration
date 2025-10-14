FAQ

- API keys / Providers
  - Tests and examples use stub providers; no external network calls happen during tests. For live runs, configure provider auth per `docs/providers.md`.

- SSE behind proxies
  - Some proxies buffer responses. Use `curl -N` or a true SSE client. If using reverse proxies, ensure `Connection: keep-alive` and disable buffering.

- Concurrency throttling
  - Concurrent fan-out uses a bounded thread pool. Cross-branch concurrency is the min of branch `tools.parallelism`. Tool execution within a branch is also bounded by `parallelism`.

- JSON Schema failures
  - With `structuredOutput.enabled`, the engine retries with a repair instruction up to `maxRepairAttempts`. On persistent failure, behavior depends on `onViolation`: `RetryAndRepair` returns best-effort text, `Refuse` raises.

- Safety refusal reasons
  - `finish_reason=content_filter` indicates a refusal due to safety policy. PII is redacted in outputs when enabled. See `docs/safety.md`.

