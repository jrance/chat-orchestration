**Decision**

Adopt an OpenAI Messages API–shaped internal provider contract and implement an `OpenAIProvider` as the first concrete provider. This keeps adapters straightforward, enables direct mapping to Gemini in a future PR, and avoids hidden transformations present in generic wrappers.

**Context**

- The orchestration engine needs a stable, minimal, and portable message and tools interface.
- OpenAI v2 Chat Completions provide a widely used baseline with function tools and streaming support.
- We anticipate a Gemini adapter that benefits from the same interface and streaming event taxonomy.

**Details**

- Interface mirrors OpenAI messages and tools to minimize impedance mismatch and support tool/function-calling and JSON mode.
- Streaming produces structured deltas for both text and function tool_calls, which maps cleanly to SSE.
- Direct SDK use avoids stale or opaque transformations from wrapper libraries, improving debuggability and forward-compatibility.

**Tradeoffs**

- Ties the internal contract to OpenAI’s shape; however, it is intentionally narrow and pragmatic, and compatible with other providers.
- JSON Schema–based response shaping deferred to a follow-up PR to keep scope manageable.

**Consequences**

- Simplifies provider adapters and test doubles.
- Enables consistent server-side streaming with predictable event types.

