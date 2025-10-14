from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class ModelParams(BaseModel):
    """Normalized model parameters for an LLM provider/model.

    Accepts both camelCase and snake_case via validation aliases.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    provider: Literal["openai", "gemini"]
    model_id: str = Field(validation_alias=AliasChoices("model_id", "modelId"))
    temperature: float | None = 0.7
    top_p: float | None = Field(
        default=None, validation_alias=AliasChoices("top_p", "topP")
    )
    max_tokens: int | None = Field(
        default=None, validation_alias=AliasChoices("max_tokens", "maxTokens")
    )
    seed: int | None = None
    stop: list[str] = Field(default_factory=list)
    json_mode_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("json_mode_enabled", "jsonModeEnabled"),
    )

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not (0.0 <= v <= 2.0):
            raise ValueError("temperature must be within [0.0, 2.0]")
        return v

    @field_validator("top_p")
    @classmethod
    def _validate_top_p(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not (0.0 < v <= 1.0):
            raise ValueError("top_p must be within (0.0, 1.0]")
        return v

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v <= 0:
            raise ValueError("max_tokens must be > 0")
        return v


class HistoryWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    mode: Literal["LastN"]
    n: int | None = None

    @model_validator(mode="after")
    def _validate_lastn(self) -> HistoryWindow:
        if self.mode == "LastN":
            if self.n is None or self.n <= 0:
                raise ValueError("history_window.n must be positive when mode is LastN")
        return self


class AgentContext(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    history_window: HistoryWindow = Field(
        validation_alias=AliasChoices("history_window", "historyWindow")
    )
    inject_org_preamble: bool = Field(
        default=True,
        validation_alias=AliasChoices("inject_org_preamble", "injectOrgPreamble"),
    )
    vars: dict[str, Any] = Field(default_factory=dict)


class ToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    policy: Literal["Auto", "None", "Required"] = "Auto"
    timeout_ms: int = Field(
        default=10000, validation_alias=AliasChoices("timeout_ms", "timeoutMs")
    )
    max_calls_per_turn: int = Field(
        default=0,
        validation_alias=AliasChoices("max_calls_per_turn", "maxCallsPerTurn"),
    )
    parallelism: int = 10
    redact_pii: bool = Field(
        default=True, validation_alias=AliasChoices("redact_pii", "redactPII")
    )
    attached: list[str] = Field(default_factory=list)


class StructuredOutputPostProcess(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    normalize_whitespace: bool = Field(
        default=True,
        validation_alias=AliasChoices("normalize_whitespace", "normalizeWhitespace"),
    )
    ensure_markdown: bool = Field(
        default=True, validation_alias=AliasChoices("ensure_markdown", "ensureMarkdown")
    )


class StructuredOutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    enabled: bool = False
    # Use a trailing underscore to avoid clashing with BaseModel.schema method.
    schema_: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("schema"),
        serialization_alias="schema",
    )
    on_violation: Literal["RetryAndRepair", "Refuse", "Ignore"] = Field(
        default="RetryAndRepair",
        validation_alias=AliasChoices("on_violation", "onViolation"),
    )
    max_repair_attempts: int = Field(
        default=2,
        validation_alias=AliasChoices("max_repair_attempts", "maxRepairAttempts"),
    )
    post_process: StructuredOutputPostProcess = Field(
        default_factory=StructuredOutputPostProcess,
        validation_alias=AliasChoices("post_process", "postProcess"),
    )


class SafetyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    policy_ref: str = Field(
        validation_alias=AliasChoices("policy_ref", "policyRef")
    )
    on_block: Literal["Refuse", "Warn"] = Field(
        default="Refuse", validation_alias=AliasChoices("on_block", "onBlock")
    )
    pii_redaction: bool = Field(
        default=True, validation_alias=AliasChoices("pii_redaction", "piiRedaction")
    )
    prompt_injection_defense: bool = Field(
        default=True,
        validation_alias=AliasChoices("prompt_injection_defense", "promptInjectionDefense"),
    )


class TelemetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    labels: dict[str, str] = Field(default_factory=dict)
    emit_usage: bool = Field(
        default=True, validation_alias=AliasChoices("emit_usage", "emitUsage")
    )


class AgentNodeData(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    system_instructions: str = Field(
        validation_alias=AliasChoices("system_instructions", "systemInstructions")
    )
    style_guide: str = Field(
        default="", validation_alias=AliasChoices("style_guide", "styleGuide")
    )
    model: ModelParams
    context: AgentContext
    tools: ToolsConfig
    structured_output: StructuredOutputConfig = Field(
        default_factory=StructuredOutputConfig,
        validation_alias=AliasChoices("structured_output", "structuredOutput"),
    )
    safety: SafetyConfig
    telemetry: TelemetryConfig


class AgentNode(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    id: str
    kind: Literal["agent.codeless"] = "agent.codeless"
    label: str
    data: AgentNodeData


class ToolNodeData(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    name: str
    tool_id: str = Field(validation_alias=AliasChoices("tool_id", "toolId"))
    version: str
    parameter_overrides: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("parameter_overrides", "parameterOverrides"),
    )


class ToolNode(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    id: str
    kind: Literal["tool"] = "tool"
    label: str
    data: ToolNodeData


Node = Annotated[AgentNode | ToolNode, Field(discriminator="kind")]


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    id: str
    from_: str = Field(validation_alias=AliasChoices("from_", "from"))
    to: str


class MetaMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    show_tool_nodes_on_canvas: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "show_tool_nodes_on_canvas", "showToolNodesOnCanvas"
        ),
    )


class Meta(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    id: str
    name: str
    version: str
    metadata: MetaMetadata


class GraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    meta: Meta
    nodes: list[Node]
    edges: list[Edge]


__all__ = [
    "ModelParams",
    "HistoryWindow",
    "AgentContext",
    "ToolsConfig",
    "StructuredOutputPostProcess",
    "StructuredOutputConfig",
    "SafetyConfig",
    "TelemetryConfig",
    "AgentNodeData",
    "AgentNode",
    "ToolNodeData",
    "ToolNode",
    "Node",
    "Edge",
    "MetaMetadata",
    "Meta",
    "GraphConfig",
    "ValidationError",
]
