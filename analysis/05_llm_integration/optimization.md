# LLM Optimization

## TL;DR
CrewAI optimize LLM usage qua **streaming** (realtime output), **caching** (tool results), **rate limiting** (RPM control), và **token tracking**. Native providers support prompt caching. Context window management prevents token overflow.

---

## 1. Streaming

### 1.1 Streaming Implementation

```python
# File: lib/crewai/src/crewai/llm.py

def _handle_streaming_response(
    self,
    params: dict,
    **kwargs
) -> str:
    """Handle streaming LLM response."""

    import litellm

    params["stream"] = True
    params["stream_options"] = {"include_usage": True}

    full_response = ""

    for chunk in litellm.completion(**params):
        # Extract content from chunk
        content = self._extract_chunk_content(chunk)

        if content:
            full_response += content

            # Emit stream event
            crewai_event_bus.emit(
                self,
                LLMStreamChunkEvent(
                    chunk=content,
                    response_id=chunk.id,
                ),
            )

    return full_response

def _extract_chunk_content(self, chunk) -> str | None:
    """Extract content from stream chunk."""

    if hasattr(chunk, "choices") and chunk.choices:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content"):
            return delta.content

    return None
```

### 1.2 Crew-Level Streaming

```python
# Enable streaming for crew
crew = Crew(
    agents=[agent],
    tasks=[task],
    stream=True,  # Enable streaming output
)

# Iterate over stream
output = crew.kickoff()
for chunk in output:
    print(chunk, end="")

# Access final result
result = output.result
```

---

## 2. Caching

### 2.1 Tool Result Caching

```python
# File: lib/crewai/src/crewai/agents/cache/cache_handler.py

class CacheHandler:
    """Cache tool execution results."""

    def __init__(self):
        self._cache: dict[str, Any] = {}

    def _build_key(self, tool: str, input: str) -> str:
        """Build cache key from tool + input hash."""
        return f"{tool}:{hashlib.md5(input.encode()).hexdigest()}"

    def read(self, tool: str, input: str) -> Any | None:
        """Read from cache."""
        key = self._build_key(tool, input)
        return self._cache.get(key)

    def add(self, tool: str, input: str, output: Any) -> None:
        """Add to cache."""
        key = self._build_key(tool, input)
        self._cache[key] = output
```

### 2.2 Custom Cache Function

```python
# Per-tool cache control
class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something"

    # Custom cache function
    cache_function: Callable = lambda args, result: (
        len(str(result)) < 10000  # Only cache small results
    )

    def _run(self, query: str) -> str:
        return expensive_operation(query)
```

### 2.3 Crew Cache Configuration

```python
crew = Crew(
    agents=[agent],
    tasks=[task],
    cache=True,  # Enable caching (default True)
)
```

---

## 3. Rate Limiting

### 3.1 RPM Controller

```python
# File: lib/crewai/src/crewai/utilities/rpm_controller.py

class RPMController:
    """Controls requests per minute rate limiting."""

    def __init__(
        self,
        max_rpm: int | None = None,
        logger: Logger | None = None,
    ):
        self.max_rpm = max_rpm
        self.logger = logger
        self._request_times: list[float] = []

    def check_or_wait(self) -> bool:
        """Check rate limit, wait if needed."""

        if not self.max_rpm:
            return True

        current_time = time.time()

        # Remove requests older than 1 minute
        self._request_times = [
            t for t in self._request_times
            if current_time - t < 60
        ]

        if len(self._request_times) >= self.max_rpm:
            # Calculate wait time
            oldest = self._request_times[0]
            wait_time = 60 - (current_time - oldest)

            if wait_time > 0:
                if self.logger:
                    self.logger.log(
                        "info",
                        f"Rate limit reached. Waiting {wait_time:.1f}s"
                    )
                time.sleep(wait_time)

        # Record this request
        self._request_times.append(time.time())
        return True
```

### 3.2 Usage in Agent

```python
# Crew-level configuration
crew = Crew(
    agents=[agent],
    tasks=[task],
    max_rpm=60,  # Max 60 requests per minute
)

# In executor loop
enforce_rpm_limit(self.request_within_rpm_limit)
answer = get_llm_response(...)
```

---

## 4. Token Tracking

### 4.1 Token Usage Tracking

```python
# File: lib/crewai/src/crewai/llms/base_llm.py

class BaseLLM:
    _token_usage: dict = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "successful_requests": 0,
        "cached_prompt_tokens": 0,
    }

    def _track_token_usage_internal(self, usage_data: dict) -> None:
        """Track token usage from response."""

        # Provider-agnostic extraction
        prompt_tokens = (
            usage_data.get("prompt_tokens") or
            usage_data.get("prompt_token_count") or
            usage_data.get("input_tokens") or
            0
        )

        completion_tokens = (
            usage_data.get("completion_tokens") or
            usage_data.get("candidates_token_count") or
            usage_data.get("output_tokens") or
            0
        )

        cached_tokens = (
            usage_data.get("cached_tokens") or
            usage_data.get("cached_prompt_tokens") or
            0
        )

        self._token_usage["prompt_tokens"] += prompt_tokens
        self._token_usage["completion_tokens"] += completion_tokens
        self._token_usage["total_tokens"] += prompt_tokens + completion_tokens
        self._token_usage["cached_prompt_tokens"] += cached_tokens
        self._token_usage["successful_requests"] += 1

    def get_token_usage_summary(self) -> UsageMetrics:
        """Get token usage summary."""
        return UsageMetrics(**self._token_usage)
```

### 4.2 Crew Usage Metrics

```python
# Access after execution
result = crew.kickoff()
print(crew.usage_metrics)
# UsageMetrics(
#     total_tokens=15000,
#     prompt_tokens=10000,
#     completion_tokens=5000,
#     successful_requests=25,
# )
```

---

## 5. Context Window Management

### 5.1 Window Sizes

```python
# File: lib/crewai/src/crewai/llms/constants.py

LLM_CONTEXT_WINDOW_SIZES = {
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "claude-3-opus": 200000,
    "gemini-1.5-pro": 1000000,
}

CONTEXT_USAGE_RATIO = 0.85  # Use 85% of context max
```

### 5.2 Context Management

```python
# In agent configuration
agent = Agent(
    role="Researcher",
    respect_context_window=True,  # Auto-summarize when exceeded
    llm="gpt-4o",  # 128k context
)
```

---

## 6. Prompt Caching (Native)

### 6.1 Anthropic Prompt Caching

```python
# File: lib/crewai/src/crewai/llms/providers/anthropic/completion.py

def call(self, messages, **kwargs):
    # Add cache control for long system prompts
    if len(messages[0]["content"]) > 1000:
        messages[0]["cache_control"] = {"type": "ephemeral"}

    response = self.client.messages.create(
        model=self.model,
        messages=messages,
    )

    # Track cached tokens
    if hasattr(response.usage, "cache_read_input_tokens"):
        self._track_cached_tokens(response.usage.cache_read_input_tokens)
```

### 6.2 OpenAI Caching (Responses API)

```python
# Automatic caching in Responses API
# No additional configuration needed
response = self.client.responses.create(
    model=self.model,
    input=messages,  # Automatically cached
)
```

---

## 7. Optimization Comparison

| Technique | Impact | Configuration |
|-----------|--------|---------------|
| Streaming | UX, latency | `stream=True` |
| Tool Caching | Cost, speed | `cache=True` (default) |
| Rate Limiting | API compliance | `max_rpm=60` |
| Token Tracking | Monitoring | Automatic |
| Context Management | Reliability | `respect_context_window=True` |
| Prompt Caching | Cost | Native provider support |

---

## 8. Best Practices

```python
# Optimized crew configuration
crew = Crew(
    agents=[agent],
    tasks=[task],

    # Streaming for user feedback
    stream=True,

    # Caching for repeated tool calls
    cache=True,

    # Rate limiting for API compliance
    max_rpm=100,

    # Memory for context reuse
    memory=True,

    # Verbose for debugging
    verbose=True,
)

# Optimized agent configuration
agent = Agent(
    role="Researcher",
    llm="gpt-4o",  # Good balance of capability/cost

    # Context management
    respect_context_window=True,

    # Limit iterations to control cost
    max_iter=15,

    # Use function calling for efficiency
    # (auto-detected from LLM)
)
```

---

## 9. Key Takeaways

1. **Streaming**: Real-time output improves user experience.

2. **Tool Caching**: Automatic with custom override support.

3. **RPM Control**: Prevents API rate limit errors.

4. **Token Tracking**: Built-in for cost monitoring.

5. **Context Management**: Auto-summarize to prevent overflow.

6. **Native Caching**: Anthropic/OpenAI prompt caching supported.

7. **Cost Optimization**: Combine caching + rate limiting + context management.

---

## File References

| Component | Path |
|-----------|------|
| LLM Streaming | `lib/crewai/src/crewai/llm.py` |
| Cache Handler | `lib/crewai/src/crewai/agents/cache/cache_handler.py` |
| RPM Controller | `lib/crewai/src/crewai/utilities/rpm_controller.py` |
| Token Tracking | `lib/crewai/src/crewai/llms/base_llm.py` |
| Context Sizes | `lib/crewai/src/crewai/llms/constants.py` |
