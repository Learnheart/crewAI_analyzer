# Anti-Patterns & Improvements

## TL;DR
CrewAI codebase có một số **areas for improvement**: large file sizes (crew.py ~2000 LOC), tight coupling trong một số areas, hardcoded values, và limited error granularity. Tuy nhiên, overall architecture solid với clear patterns.

---

## 1. Large File Sizes

### 1.1 Issue

```
crew.py          ~2,039 lines
agent/core.py    ~2,162 lines
llm.py           ~2,362 lines
flow/flow.py     ~2,641 lines
```

### 1.2 Impact
- Difficult to navigate
- Higher cognitive load
- Merge conflicts more likely
- Testing complexity

### 1.3 Potential Improvement

```python
# Split crew.py into:
# crew/base.py - Core Crew class
# crew/sequential.py - Sequential process
# crew/hierarchical.py - Hierarchical process
# crew/memory.py - Memory integration
# crew/streaming.py - Streaming support
```

---

## 2. Hardcoded Context Window Sizes

### 2.1 Issue

```python
# File: lib/crewai/src/crewai/llms/constants.py

LLM_CONTEXT_WINDOW_SIZES = {
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    # ... hardcoded values
}
```

### 2.2 Impact
- Outdated when models update
- Manual maintenance required
- New models need code changes

### 2.3 Potential Improvement

```python
# Option 1: API-based discovery
def get_context_window_size(model: str) -> int:
    # Query model capabilities from API
    capabilities = get_model_capabilities(model)
    return capabilities.context_window

# Option 2: Configuration file
# config/model_limits.yaml
# gpt-4:
#   context_window: 8192
#   supports_function_calling: true
```

---

## 3. Mixed Abstraction Levels

### 3.1 Issue

```python
# crew.py mixes high-level and low-level concerns
class Crew:
    def kickoff(self):
        # High-level orchestration
        inputs = prepare_kickoff(...)

        # Low-level process execution
        if self.process == Process.sequential:
            result = self._run_sequential_process()

        # Low-level usage metrics
        self.usage_metrics = self.calculate_usage_metrics()

        return result
```

### 3.2 Potential Improvement

```python
# Separate into layers
class CrewOrchestrator:
    """High-level orchestration."""
    def kickoff(self, inputs):
        prepared = self.preparer.prepare(inputs)
        result = self.executor.execute(prepared)
        return self.finalizer.finalize(result)

class ProcessExecutor:
    """Process execution logic."""
    def execute(self, prepared):
        strategy = self.get_strategy(self.process)
        return strategy.run(prepared)
```

---

## 4. Limited Error Granularity

### 4.1 Issue

```python
# Generic exceptions
except Exception as e:
    if self._run_attempts > self._max_parsing_attempts:
        return ToolUsageError(f"Error: {e}").message
```

### 4.2 Impact
- Hard to handle specific errors
- Generic error messages
- Difficult debugging

### 4.3 Potential Improvement

```python
# Define specific exceptions
class ToolParsingError(CrewAIError):
    """Tool input could not be parsed."""

class ToolExecutionError(CrewAIError):
    """Tool execution failed."""

class ToolUsageLimitError(CrewAIError):
    """Tool usage limit exceeded."""

# Handle specifically
try:
    result = tool.execute(args)
except ToolParsingError:
    return self._handle_parsing_error()
except ToolExecutionError:
    return self._handle_execution_error()
```

---

## 5. Tight Coupling in Places

### 5.1 Issue

```python
# Agent directly accesses crew internals
if self._is_any_available_memory():
    contextual_memory = ContextualMemory(
        self.crew._short_term_memory,  # Direct access
        self.crew._long_term_memory,
        self.crew._entity_memory,
        self.crew._external_memory,
    )
```

### 5.2 Potential Improvement

```python
# Use interface/abstraction
class Crew:
    def get_memory_provider(self) -> MemoryProvider:
        """Return memory abstraction."""
        return CrewMemoryProvider(
            stm=self._short_term_memory,
            ltm=self._long_term_memory,
            em=self._entity_memory,
            xm=self._external_memory,
        )

# Agent uses abstraction
memory_provider = self.crew.get_memory_provider()
context = memory_provider.get_context_for_task(task)
```

---

## 6. String-Based Type Checking

### 6.1 Issue

```python
# Type checking via string comparison
if self._memory_provider == "mem0":
    # Mem0 specific logic
else:
    # Default logic
```

### 6.2 Potential Improvement

```python
# Use polymorphism
class MemoryProvider(ABC):
    @abstractmethod
    def format_for_save(self, data: str) -> str:
        pass

class Mem0Provider(MemoryProvider):
    def format_for_save(self, data: str) -> str:
        return f"Remember the following: {data}"

class RAGProvider(MemoryProvider):
    def format_for_save(self, data: str) -> str:
        return data

# Usage
formatted = self.provider.format_for_save(data)
```

---

## 7. Global State (Event Bus)

### 7.1 Issue

```python
# Global event bus instance
crewai_event_bus = EventBus()

# Used throughout codebase
crewai_event_bus.emit(self, event)
```

### 7.2 Impact
- Difficult to test in isolation
- Global state complications
- Threading concerns

### 7.3 Potential Improvement

```python
# Dependency injection
class Agent:
    def __init__(self, event_bus: EventBus = None):
        self.event_bus = event_bus or default_event_bus

    def execute(self):
        self.event_bus.emit(self, StartedEvent())

# Testing
mock_bus = MockEventBus()
agent = Agent(event_bus=mock_bus)
```

---

## 8. Inconsistent Async Support

### 8.1 Issue

```python
# Some operations have async, some don't
class Memory:
    def save(self, ...):  # Sync
        ...

    async def asave(self, ...):  # Async
        ...

    # But some methods only sync
    def reset(self):  # No async version
        ...
```

### 8.2 Potential Improvement

```python
# Consistent async support for all I/O operations
class Memory:
    def reset(self):
        ...

    async def areset(self):  # Add async version
        ...
```

---

## 9. Magic Strings

### 9.1 Issue

```python
# Scattered string literals
if msg["role"] == "system":
    ...
if msg["role"] == "user":
    ...
if msg["role"] == "assistant":
    ...
```

### 9.2 Potential Improvement

```python
# Use constants/enums
class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

if msg["role"] == MessageRole.SYSTEM:
    ...
```

---

## 10. Summary Table

| Issue | Severity | Fix Effort | Priority |
|-------|----------|------------|----------|
| Large files | Medium | High | Medium |
| Hardcoded values | Low | Low | Low |
| Mixed abstraction | Medium | High | Medium |
| Error granularity | Medium | Medium | High |
| Tight coupling | Medium | Medium | Medium |
| String type checks | Low | Low | Low |
| Global state | Low | High | Low |
| Inconsistent async | Low | Medium | Low |
| Magic strings | Low | Low | Low |

---

## 11. What's Done Well

Despite these areas for improvement, CrewAI does many things well:

1. **Pydantic Usage**: Consistent data validation
2. **Event System**: Good observability
3. **Provider Abstraction**: Clean LLM integration
4. **Type Hints**: Good type coverage
5. **Documentation**: Docstrings present
6. **Async Support**: Where implemented, done correctly
7. **Plugin Architecture**: Tools, providers extensible
8. **Clear Patterns**: Consistent coding style

---

## File References

| Area | Files |
|------|-------|
| Large files | `crew.py`, `agent/core.py`, `llm.py` |
| Constants | `llms/constants.py` |
| Memory coupling | `agent/core.py`, `memory/*.py` |
| Events | `events/event_bus.py` |
