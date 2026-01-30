# Code Patterns

## TL;DR
CrewAI codebase sử dụng các **coding patterns** nhất quán: Pydantic models cho data validation, PrivateAttr cho internal state, TYPE_CHECKING cho circular imports, event emission pattern, và async/sync dual implementation. Patterns này tạo codebase clean và maintainable.

---

## 1. Pydantic Model Pattern

### 1.1 BaseModel Usage

```python
# Most classes extend Pydantic BaseModel
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict

class Agent(BaseAgent):
    model_config = ConfigDict()

    # Public fields with validation
    role: str = Field(description="Agent's role")
    goal: str = Field(description="Agent's objective")
    llm: str | InstanceOf[BaseLLM] = Field(default=None)

    # Default values
    max_iter: int = Field(default=25)
    verbose: bool = Field(default=False)

    # Private attributes (not serialized)
    _times_executed: int = PrivateAttr(default=0)
    _mcp_clients: list = PrivateAttr(default_factory=list)
```

### 1.2 Model Validators

```python
from pydantic import model_validator

class Agent(BaseAgent):
    @model_validator(mode="after")
    def post_init_setup(self) -> Self:
        """Run after all fields are set."""
        self.llm = create_llm(self.llm)
        return self

    @model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: dict) -> dict:
        """Run before field validation."""
        if "llm" not in data:
            data["llm"] = "gpt-4o"
        return data
```

### 1.3 Field Validators

```python
from pydantic import field_validator

class Task(BaseModel):
    @field_validator("output_pydantic", mode="before")
    @classmethod
    def validate_output_model(cls, v):
        """Validate output model is Pydantic BaseModel."""
        if v is not None and not issubclass(v, BaseModel):
            raise ValueError("output_pydantic must be Pydantic model")
        return v
```

---

## 2. Private Attribute Pattern

### 2.1 For Internal State

```python
class Agent(BaseAgent):
    # Public attributes (validated, serialized)
    role: str
    goal: str

    # Private attributes (internal state only)
    _times_executed: int = PrivateAttr(default=0)
    _mcp_clients: list[Any] = PrivateAttr(default_factory=list)
    _last_messages: list[LLMMessage] = PrivateAttr(default_factory=list)

    # Access in methods
    def execute_task(self, task):
        self._times_executed += 1
        # ...
```

### 2.2 For Computed Values

```python
class Crew(BaseModel):
    # User-provided
    agents: list[Agent]
    tasks: list[Task]

    # Computed/managed internally
    _cache_handler: CacheHandler = PrivateAttr()
    _rpm_controller: RPMController = PrivateAttr()
    _logger: Logger = PrivateAttr()

    @model_validator(mode="after")
    def set_private_attrs(self) -> Self:
        self._cache_handler = CacheHandler()
        self._rpm_controller = RPMController(max_rpm=self.max_rpm)
        return self
```

---

## 3. TYPE_CHECKING Pattern

### 3.1 Avoid Circular Imports

```python
# File: lib/crewai/src/crewai/agent/core.py

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool

class Agent(BaseAgent):
    # Use string annotation for forward reference
    def execute_task(self, task: "Task") -> Any:
        ...
```

### 3.2 Runtime vs Type Check Imports

```python
# Runtime import (always loaded)
from crewai.llm import LLM

# Type-only import (not loaded at runtime)
if TYPE_CHECKING:
    from crewai.crew import Crew

class Agent:
    crew: "Crew"  # String annotation
```

---

## 4. Event Emission Pattern

### 4.1 Consistent Event Pattern

```python
def execute_task(self, task):
    # Emit started event
    crewai_event_bus.emit(
        self,
        AgentExecutionStartedEvent(
            agent=self,
            task=task,
            task_prompt=task_prompt,
        ),
    )

    try:
        result = self._execute(task)

        # Emit completed event
        crewai_event_bus.emit(
            self,
            AgentExecutionCompletedEvent(
                agent=self,
                task=task,
                output=result,
            ),
        )
        return result

    except Exception as e:
        # Emit error event
        crewai_event_bus.emit(
            self,
            AgentExecutionErrorEvent(
                agent=self,
                task=task,
                error=str(e),
            ),
        )
        raise
```

### 4.2 Event Class Pattern

```python
# File: lib/crewai/src/crewai/events/types/agent_events.py

from pydantic import BaseModel

class AgentExecutionStartedEvent(BaseModel):
    """Event emitted when agent starts execution."""
    agent: Any
    task: Any
    task_prompt: str
    tools: list[Any] = []

class AgentExecutionCompletedEvent(BaseModel):
    """Event emitted when agent completes."""
    agent: Any
    task: Any
    output: Any
```

---

## 5. Async/Sync Dual Implementation

### 5.1 Dual Methods

```python
class Memory:
    def save(self, value: Any, metadata: dict = None):
        """Sync save."""
        self.storage.save(value, metadata)

    async def asave(self, value: Any, metadata: dict = None):
        """Async save."""
        await self.storage.asave(value, metadata)

    def search(self, query: str, limit: int = 5):
        """Sync search."""
        return self.storage.search(query, limit)

    async def asearch(self, query: str, limit: int = 5):
        """Async search."""
        return await self.storage.asearch(query, limit)
```

### 5.2 Async Wrapper Pattern

```python
class Crew:
    def kickoff(self, inputs=None):
        """Sync execution."""
        return self._run_process()

    async def kickoff_async(self, inputs=None):
        """Async wrapper around sync."""
        return await asyncio.to_thread(self.kickoff, inputs)

    async def akickoff(self, inputs=None):
        """Native async execution."""
        return await self._arun_process()
```

---

## 6. Builder/Fluent Pattern

### 6.1 Chained Configuration

```python
# Memory builder pattern
memory = (
    Memory()
    .set_crew(crew)
    .configure_storage(RAGStorage(...))
)

# Agent with chained setters
agent = Agent(role="Researcher", goal="Find info", backstory="Expert")
agent.set_cache_handler(cache_handler)
agent.set_rpm_controller(rpm_controller)
```

### 6.2 Return Self Pattern

```python
class Memory:
    def set_crew(self, crew: Any) -> "Memory":
        """Set crew and return self for chaining."""
        self.crew = crew
        return self
```

---

## 7. Lazy Initialization Pattern

### 7.1 Lazy Provider Loading

```python
# File: lib/crewai/src/crewai/llm.py

def _get_native_provider(provider: str, model: str, **kwargs):
    """Lazy load provider class."""
    providers = {
        "openai": ("crewai.llms.providers.openai", "OpenAICompletion"),
        "anthropic": ("crewai.llms.providers.anthropic", "AnthropicCompletion"),
    }

    module_path, class_name = providers[provider]

    # Import only when needed
    module = importlib.import_module(module_path)
    provider_class = getattr(module, class_name)

    return provider_class(model=model, **kwargs)
```

### 7.2 Lazy Executor Creation

```python
class Agent:
    def execute_task(self, task):
        # Create executor only when needed
        if self.agent_executor is None:
            self.create_agent_executor(task=task)
        else:
            self._update_executor_parameters(task=task)
```

---

## 8. Guard Clause Pattern

### 8.1 Early Returns

```python
def search(self, query: str, limit: int = 5):
    # Guard clauses
    if not query:
        return []

    if not self.storage:
        return []

    if limit <= 0:
        return []

    # Main logic
    return self.storage.search(query, limit)
```

### 8.2 Validation Guards

```python
def _run(self, file_name: str) -> str:
    # Guard: No files available
    if not self._files:
        return "No input files available."

    # Guard: File not found
    if file_name not in self._files:
        available = ", ".join(self._files.keys())
        return f"File '{file_name}' not found. Available: {available}"

    # Main logic
    return self._files[file_name].read()
```

---

## 9. Context Manager Pattern

### 9.1 Database Connections

```python
def save(self, task_description: str, metadata: dict):
    with sqlite3.connect(self.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO memories VALUES (?, ?)",
            (task_description, json.dumps(metadata)),
        )
        conn.commit()
```

### 9.2 OpenTelemetry Context

```python
from opentelemetry.context import attach, detach

def kickoff(self, inputs=None):
    baggage_ctx = baggage.set_baggage("crew_context", ctx)
    token = attach(baggage_ctx)

    try:
        result = self._run_process()
        return result
    finally:
        detach(token)
```

---

## 10. Key Takeaways

| Pattern | Usage | Benefit |
|---------|-------|---------|
| **Pydantic Models** | Data classes | Validation, serialization |
| **PrivateAttr** | Internal state | Clean API |
| **TYPE_CHECKING** | Imports | Avoid circular |
| **Event Emission** | Observability | Loose coupling |
| **Async/Sync Dual** | All I/O ops | Flexibility |
| **Lazy Init** | Providers | Performance |
| **Guard Clauses** | Methods | Readability |
| **Context Manager** | Resources | Safety |

---

## File References

| Pattern | Example Files |
|---------|---------------|
| Pydantic | `agent/core.py`, `crew.py`, `task.py` |
| PrivateAttr | `agent/core.py`, `crew.py` |
| TYPE_CHECKING | `agent/core.py`, `llm.py` |
| Events | `events/types/*.py` |
| Async/Sync | `memory/*.py`, `crew.py` |
