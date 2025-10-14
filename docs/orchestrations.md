Orchestrations

Overview
- Sequential: Linear chain of agents (A → B → C), each with its own tool loop.
- Handoff: A branches to one of multiple next agents (A → {B|C|…}) selected by an LLM router.
- Concurrency: A single stage fans out to multiple agents in parallel and fans in to a join step with a merge strategy.
- Group Chat: A strongly connected set of agents take turns speaking using either round‑robin or a moderator policy.

Detection
- Build an agent‑only graph: nodes are `agent.codeless`; edges only where both endpoints are agents (ignore tool nodes/edges).
- Sequential when: one root, graph is a single path covering all agents, and every agent has outdegree ≤ 1.
- Handoff when: one root and at least one agent has outdegree > 1, and the pattern does not meet concurrency rules below.
- Concurrency when: a node has outdegree > 1 and either (a) all direct successors are leaves, or (b) all direct successors share a single common next agent (an explicit join).
- Group chat when: the agent‑only graph contains a strongly connected component (SCC) with size ≥ 2, or any agent has a self‑loop. The compiler delegates to the group chat builder.
- Moderator vs round‑robin selection: moderator mode if exactly one agent in the SCC is a hub (indegree ≥ 2 and outdegree ≥ 2) or any label contains “moderator” (case‑insensitive). Otherwise, round‑robin.

State & Flow
- Global state: `messages`, `metadata`, `tool_calls`, `steps`, `agent_cursor`.
- Concurrency bucket: `concurrent` transient bucket for fan‑out/fan‑in stages.
- Group chat bucket: `group = { participants, speaker, turn, max_turns, mode, end }`.
- Each agent node:
  - Calls its provider with its model params and attached tool specs.
  - Appends assistant message; captures `tool_calls`; increments `steps`.
- Tool nodes (per agent):
  - Execute pending tool calls deterministically with overrides, timeouts, and limits.
  - Append `tool` messages with `tool_call_id`; clear `tool_calls` and loop back to the same agent.

Group Chat
- State machine: `select_next → speak:{agent_id} → tools:{agent_id} → select_next` until `turn >= max_turns` or moderator ends.
- Round‑robin: picks the next participant in order, wrapping.
- Moderator: a low‑temperature LLM decides `{ "next": "agent_id" }` or `{ "end": true }`; parsing is robust (code‑fence tolerant, JSON first) with deterministic fallback to the first eligible participant.
- Tool loop per speaker: enforces per‑agent overrides, timeouts, and bounded parallelism; preserves original tool call order.
- Speaker attribution: assistant messages include `[Agent: <label>]` prefix for UIs to attribute speakers without changing the message schema.

Concurrency
- Execution model: a synthetic `fanout:{group}` node runs multiple branch agents in parallel (each performs a full assistant turn, including its per‑turn tool loop). A `join:{group}` node merges the branch outputs.
- Throttling across branches: bounded by the minimum `tools.parallelism` across the branch agents (or the number of branches, whichever is smaller).
- Throttling within a turn: if the assistant returns multiple `tool_calls` in one turn, tool executions run concurrently up to `agent.tools.parallelism` and preserve the original order in the transcript.
- Merge strategies:
  - `aggregate` (default): concatenate branch outputs in deterministic order (by agent id).
  - `first_finish`: pick the earliest completed branch.
  - `vote(length)`: pick the longest assistant text; ties broken by agent id.
  - Strategy selection: defaults to `aggregate`. When an explicit common downstream agent is present, the engine prefers `first_finish` to feed a single next agent.

Handoff Routing
- After an agent finishes without tool calls:
  - 0 outgoing: END.
  - 1 outgoing: go to that agent.
  - >1 outgoing: use the LLM router to choose the next agent.
- Router: `choose_next_agent(messages, current_agent, candidates, provider)`.
  - Builds a low‑temperature classification prompt and calls `provider.chat` with JSON mode where possible.
  - Parses strictly, then heuristically; falls back deterministically to the first candidate and logs a warning.

Provider & Tools
- Each agent gets its own provider resolved from its `model` params.
- Tool specs are taken from the registry for the agent’s `tools.attached` IDs.
- Execution honors per‑agent `tools.timeout_ms`, `max_calls_per_turn`, parameter overrides (from tool nodes), and `parallelism`.

Streaming
- Basic streaming remains single‑turn without tool calls. If a turn yields tool calls, streaming ends and the engine continues non‑streaming. Rich interleaving SSE is planned in a later PR.

Limitations
- Nested/complex DAG concurrency patterns beyond a single stage are not yet supported.
- Advanced memory, org preamble, and graph variables will arrive later.
