"""Core type definitions for the pi_ai package.

This module defines the unified types used across all LLM provider adapters,
mirroring the TypeScript ``packages/ai/src/types.ts``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

# --- API and provider identifiers ---

KnownApi: TypeAlias = Literal[
    "openai-completions",
    "mistral-conversations",
    "openai-responses",
    "azure-openai-responses",
    "openai-codex-responses",
    "anthropic-messages",
    "bedrock-converse-stream",
    "google-generative-ai",
    "google-vertex",
    "pi-messages",
]
Api: TypeAlias = str
KnownImagesApi: TypeAlias = Literal["openrouter-images"]
ImagesApi: TypeAlias = str

KnownProvider: TypeAlias = Literal[
    "amazon-bedrock",
    "ant-ling",
    "anthropic",
    "google",
    "google-vertex",
    "openai",
    "azure-openai-responses",
    "openai-codex",
    "radius",
    "nvidia",
    "deepseek",
    "github-copilot",
    "xai",
    "groq",
    "cerebras",
    "openrouter",
    "vercel-ai-gateway",
    "zai",
    "zai-coding-cn",
    "mistral",
    "minimax",
    "minimax-cn",
    "moonshotai",
    "moonshotai-cn",
    "huggingface",
    "fireworks",
    "together",
    "opencode",
    "opencode-go",
    "kimi-coding",
    "cloudflare-workers-ai",
    "cloudflare-ai-gateway",
    "qwen-token-plan",
    "qwen-token-plan-cn",
    "xiaomi",
    "xiaomi-token-plan-cn",
    "xiaomi-token-plan-ams",
    "xiaomi-token-plan-sgp",
]
ProviderId: TypeAlias = str
KnownImagesProvider: TypeAlias = Literal["openrouter"]
ImagesProviderId: TypeAlias = str

ThinkingLevel: TypeAlias = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
ModelThinkingLevel: TypeAlias = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]
ThinkingLevelMap: TypeAlias = dict[ModelThinkingLevel, str | None]

ChatTemplateKwargValue: TypeAlias = str | int | float | bool | None | dict[str, Any]

CacheRetention: TypeAlias = Literal["none", "short", "long"]
Transport: TypeAlias = Literal["sse", "websocket", "websocket-cached", "auto"]
ProviderEnv: TypeAlias = dict[str, str]
ProviderHeaders: TypeAlias = dict[str, str | None]
SessionAffinityFormat: TypeAlias = Literal["openai", "openai-nosession", "openrouter"]

GrammarFormat: TypeAlias = Literal["openai_lark", "openai_regex"]
GrammarVariants: TypeAlias = dict[GrammarFormat, str]


class StopReason(str, Enum):
    PENDING = "pending"
    STOP = "stop"
    LENGTH = "length"
    TOOL_USE = "toolUse"
    ERROR = "error"
    ABORTED = "aborted"


ImagesStopReason: TypeAlias = Literal["stop", "error", "aborted"]


# --- Content types ---


@dataclass
class TextContent:
    type: str = field(default="text", init=False)
    text: str = ""
    text_signature: str | None = None


@dataclass
class ThinkingContent:
    type: str = field(default="thinking", init=False)
    thinking: str = ""
    thinking_signature: str | None = None
    redacted: bool | None = None


@dataclass
class ImageContent:
    type: str = field(default="image", init=False)
    data: str = ""
    mime_type: str = ""


@dataclass
class ToolCall:
    type: str = field(default="toolCall", init=False)
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    thought_signature: str | None = None


AssistantContent: TypeAlias = TextContent | ThinkingContent | ToolCall
UserContent: TypeAlias = TextContent | ImageContent
ToolResultContent: TypeAlias = TextContent | ImageContent


# --- Usage ---


@dataclass
class UsageCost:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: int | None = None
    reasoning: int | None = None
    total_tokens: int = 0
    cost: UsageCost = field(default_factory=UsageCost)


# --- Messages ---


@dataclass
class UserMessage:
    role: str = field(default="user", init=False)
    content: str | list[UserContent] = ""
    timestamp: int = 0


@dataclass
class AssistantMessage:
    role: str = field(default="assistant", init=False)
    content: list[AssistantContent] = field(default_factory=list)
    api: str = ""
    provider: str = ""
    model: str = ""
    response_model: str | None = None
    response_id: str | None = None
    diagnostics: list[dict[str, Any]] | None = None
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = StopReason.PENDING
    error_message: str | None = None
    raw_stop_reason: str | None = None
    timestamp: int = 0


@dataclass
class ToolResultMessage:
    role: str = field(default="toolResult", init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    content: list[ToolResultContent] = field(default_factory=list)
    details: Any = None
    usage: Usage | None = None
    added_tool_names: list[str] | None = None
    is_error: bool = False
    timestamp: int = 0


Message: TypeAlias = UserMessage | AssistantMessage | ToolResultMessage


# --- Model cost ---


@dataclass
class ModelCostRates:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclass
class ModelCostTier(ModelCostRates):
    input_tokens_above: int = 0


@dataclass
class ModelCost(ModelCostRates):
    tiers: list[ModelCostTier] | None = None


# --- Constrained sampling ---


@dataclass
class ConstrainedSamplingJsonSchema:
    type: str = field(default="json_schema", init=False)
    strict: Literal["prefer", "require"] = "prefer"


@dataclass
class ConstrainedSamplingGrammar:
    type: str = field(default="grammar", init=False)
    variants: GrammarVariants = field(default_factory=dict)


ConstrainedSamplingConfig: TypeAlias = ConstrainedSamplingJsonSchema | ConstrainedSamplingGrammar


# --- Tool definition ---


@dataclass
class Tool:
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    constrained_sampling: bool | ConstrainedSamplingConfig | None = None


# --- Context ---


@dataclass
class Context:
    system_prompt: str | None = None
    messages: list[Message] = field(default_factory=list)
    tools: Sequence[Tool] | None = None


# --- Stream options ---


@dataclass
class ProviderResponse:
    status: int = 0
    headers: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class FetchFunction(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


PayloadCallback: TypeAlias = Callable[[Any, Any], Any | None]
ResponseCallback: TypeAlias = Callable[[ProviderResponse, Any], None]


@dataclass
class StreamOptions:
    temperature: float | None = None
    max_tokens: int | None = None
    signal: Any = None
    api_key: str | None = None
    fetch: FetchFunction | None = None
    transport: Transport | None = None
    cache_retention: CacheRetention | None = None
    session_id: str | None = None
    on_payload: PayloadCallback | None = None
    on_response: ResponseCallback | None = None
    headers: ProviderHeaders | None = None
    timeout_ms: int | None = None
    websocket_connect_timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None
    metadata: dict[str, Any] | None = None
    env: ProviderEnv | None = None


@dataclass
class SimpleStreamOptions(StreamOptions):
    reasoning: ThinkingLevel | None = None
    thinking_budgets: dict[str, int] | None = None


@dataclass
class ThinkingBudgets:
    minimal: int | None = None
    low: int | None = None
    medium: int | None = None
    high: int | None = None


# --- Image types ---


@dataclass
class ImagesContext:
    input: list[TextContent | ImageContent] = field(default_factory=list)


@dataclass
class AssistantImages:
    api: str = ""
    provider: str = ""
    model: str = ""
    output: list[TextContent | ImageContent] = field(default_factory=list)
    response_id: str | None = None
    usage: Usage | None = None
    stop_reason: ImagesStopReason = "stop"
    error_message: str | None = None
    timestamp: int = 0


@dataclass
class ImagesOptions:
    signal: Any = None
    api_key: str | None = None
    fetch: FetchFunction | None = None
    env: ProviderEnv | None = None
    on_payload: PayloadCallback | None = None
    on_response: ResponseCallback | None = None
    headers: ProviderHeaders | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None
    metadata: dict[str, Any] | None = None


# --- Model ---


@dataclass
class Model:
    id: str = ""
    name: str = ""
    api: str = ""
    provider: str = ""
    base_url: str = ""
    reasoning: bool = False
    thinking_level_map: ThinkingLevelMap | None = None
    input: list[Literal["text", "image"]] = field(default_factory=list)
    cost: ModelCost = field(default_factory=ModelCost)
    context_window: int = 0
    max_tokens: int = 0
    headers: dict[str, str] | None = None
    compat: Any = None


@dataclass
class ImagesModel:
    id: str = ""
    name: str = ""
    api: str = ""
    provider: str = ""
    base_url: str = ""
    input: list[Literal["text", "image"]] = field(default_factory=list)
    cost: ModelCost = field(default_factory=ModelCost)
    headers: dict[str, str] | None = None
    output: list[Literal["text", "image"]] = field(default_factory=list)


# --- Compat types ---


@dataclass
class OpenAICompletionsCompat:
    supports_store: bool | None = None
    supports_developer_role: bool | None = None
    supports_reasoning_effort: bool | None = None
    supports_usage_in_streaming: bool | None = None
    supports_finish_reason: bool | None = None
    max_tokens_field: Literal["max_completion_tokens", "max_tokens"] | None = None
    requires_tool_result_name: bool | None = None
    requires_assistant_after_tool_result: bool | None = None
    requires_thinking_as_text: bool | None = None
    requires_reasoning_content_on_assistant_messages: bool | None = None
    thinking_format: str | None = None
    chat_template_kwargs: dict[str, ChatTemplateKwargValue] | None = None
    open_router_routing: Any = None
    vercel_gateway_routing: Any = None
    zai_tool_stream: bool | None = None
    supports_openai_grammar_tools: bool | None = None
    supports_strict_mode: bool | None = None
    cache_control_format: Literal["anthropic"] | None = None
    send_session_affinity_headers: bool | None = None
    deferred_tools_mode: Literal["kimi"] | None = None
    session_affinity_format: SessionAffinityFormat | None = None
    supports_long_cache_retention: bool | None = None


@dataclass
class OpenAIResponsesCompat:
    supports_developer_role: bool | None = None
    session_affinity_format: SessionAffinityFormat | None = None
    supports_long_cache_retention: bool | None = None
    supports_strict_mode: bool | None = None
    supports_openai_grammar_tools: bool | None = None
    supports_tool_search: bool | None = None
    supports_explicit_prompt_cache_mode: bool | None = None


@dataclass
class AnthropicMessagesCompat:
    supports_eager_tool_input_streaming: bool | None = None
    supports_long_cache_retention: bool | None = None
    send_session_affinity_headers: bool | None = None
    supports_cache_control_on_tools: bool | None = None
    supports_temperature: bool | None = None
    force_adaptive_thinking: bool | None = None
    allow_empty_signature: bool | None = None
    supports_strict_tools: bool | None = None
    supports_tool_references: bool | None = None


@dataclass
class BedrockCompat:
    supports_strict_mode: bool | None = None


# --- Assistant message events ---


@dataclass
class AssistantMessageEvent:
    """Base class for stream events."""

    type: str = ""
    partial: AssistantMessage | None = None


@dataclass
class TextStartEvent(AssistantMessageEvent):
    type: str = field(default="text_start", init=False)
    content_index: int = 0


@dataclass
class TextDeltaEvent(AssistantMessageEvent):
    type: str = field(default="text_delta", init=False)
    content_index: int = 0
    delta: str = ""


@dataclass
class TextEndEvent(AssistantMessageEvent):
    type: str = field(default="text_end", init=False)
    content_index: int = 0
    content: str = ""


@dataclass
class ThinkingStartEvent(AssistantMessageEvent):
    type: str = field(default="thinking_start", init=False)
    content_index: int = 0


@dataclass
class ThinkingDeltaEvent(AssistantMessageEvent):
    type: str = field(default="thinking_delta", init=False)
    content_index: int = 0
    delta: str = ""


@dataclass
class ThinkingEndEvent(AssistantMessageEvent):
    type: str = field(default="thinking_end", init=False)
    content_index: int = 0
    content: str = ""


@dataclass
class ToolCallStartEvent(AssistantMessageEvent):
    type: str = field(default="toolcall_start", init=False)
    content_index: int = 0


@dataclass
class ToolCallDeltaEvent(AssistantMessageEvent):
    type: str = field(default="toolcall_delta", init=False)
    content_index: int = 0
    delta: str = ""


@dataclass
class ToolCallEndEvent(AssistantMessageEvent):
    type: str = field(default="toolcall_end", init=False)
    content_index: int = 0
    tool_call: ToolCall | None = None


@dataclass
class StartEvent(AssistantMessageEvent):
    type: str = field(default="start", init=False)


@dataclass
class DoneEvent:
    type: str = field(default="done", init=False)
    reason: Literal["stop", "length", "toolUse"] = "stop"
    message: AssistantMessage | None = None


@dataclass
class ErrorEvent:
    type: str = field(default="error", init=False)
    reason: Literal["aborted", "error"] = "error"
    error: AssistantMessage | None = None


AssistantMessageEventType: TypeAlias = (
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | DoneEvent
    | ErrorEvent
)


# --- Text signature ---


@dataclass
class TextSignatureV1:
    v: int = field(default=1, init=False)
    id: str = ""
    phase: Literal["commentary", "final_answer"] | None = None
