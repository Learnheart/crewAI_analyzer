# Implementation Guide

## TL;DR
Hướng dẫn áp dụng patterns từ CrewAI vào project của bạn. Bao gồm: **Agent architecture**, **Memory system**, **Tool framework**, **LLM abstraction**, và **Orchestration patterns**. Code examples và best practices cho mỗi component.

---

## 1. Building Agent System

### 1.1 Agent Base Class

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any

class BaseAgent(BaseModel, ABC):
    """Base class for all agents."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    role: str = Field(description="Agent's role")
    goal: str = Field(description="Agent's objective")
    backstory: str = Field(description="Agent's background")

    llm: Any = Field(default=None)
    tools: list = Field(default_factory=list)
    memory: Any = Field(default=None)

    max_iterations: int = Field(default=25)
    verbose: bool = Field(default=False)

    @abstractmethod
    def execute(self, task: str, context: str = "") -> str:
        """Execute a task."""
        pass
```

### 1.2 ReAct Implementation

```python
class ReactAgent(BaseAgent):
    """Agent using ReAct pattern."""

    def execute(self, task: str, context: str = "") -> str:
        messages = self._build_initial_messages(task, context)
        iterations = 0

        while iterations < self.max_iterations:
            # Get LLM response
            response = self.llm.call(messages)

            # Parse response
            parsed = self._parse_response(response)

            if parsed.is_final_answer:
                return parsed.output

            # Execute tool
            if parsed.tool_call:
                result = self._execute_tool(
                    parsed.tool_call.name,
                    parsed.tool_call.args
                )
                messages.append({
                    "role": "user",
                    "content": f"Observation: {result}"
                })

            iterations += 1

        return "Max iterations reached"

    def _build_initial_messages(self, task: str, context: str) -> list:
        system_prompt = f"""You are {self.role}.
Your goal is: {self.goal}
Background: {self.backstory}

Available tools: {self._format_tools()}

Use this format:
Thought: [your reasoning]
Action: [tool name]
Action Input: [tool arguments as JSON]

Or when done:
Thought: [final reasoning]
Final Answer: [your answer]
"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}\nContext: {context}"}
        ]
```

---

## 2. Building Memory System

### 2.1 Memory Interface

```python
from abc import ABC, abstractmethod

class MemoryStorage(ABC):
    """Abstract storage interface."""

    @abstractmethod
    def save(self, value: Any, metadata: dict) -> None:
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> list:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
```

### 2.2 RAG Storage Implementation

```python
import chromadb

class RAGStorage(MemoryStorage):
    """ChromaDB-based semantic storage."""

    def __init__(self, collection_name: str, embedder=None):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedder
        )

    def save(self, value: str, metadata: dict = None) -> None:
        self.collection.add(
            documents=[value],
            metadatas=[metadata or {}],
            ids=[str(uuid4())]
        )

    def search(self, query: str, limit: int = 5) -> list:
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        return [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(
                results["documents"][0],
                results["metadatas"][0]
            )
        ]

    def reset(self) -> None:
        self.client.delete_collection(self.collection.name)
```

### 2.3 Multi-Tier Memory

```python
class MemoryManager:
    """Manages multiple memory types."""

    def __init__(self):
        self.short_term = RAGStorage("short_term")
        self.long_term = SQLiteStorage("long_term.db")
        self.entities = RAGStorage("entities")

    def get_context(self, query: str) -> str:
        """Aggregate context from all sources."""
        context_parts = []

        # Short-term (recent)
        stm_results = self.short_term.search(query, limit=3)
        if stm_results:
            context_parts.append("Recent:\n" + self._format(stm_results))

        # Long-term (historical)
        ltm_results = self.long_term.search(query, limit=3)
        if ltm_results:
            context_parts.append("Historical:\n" + self._format(ltm_results))

        # Entities
        entity_results = self.entities.search(query, limit=3)
        if entity_results:
            context_parts.append("Entities:\n" + self._format(entity_results))

        return "\n\n".join(context_parts)
```

---

## 3. Building Tool System

### 3.1 Tool Base Class

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

class BaseTool(BaseModel, ABC):
    """Base class for tools."""

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the tool."""
        pass

    def get_schema(self) -> dict:
        """Get tool schema for LLM."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._get_parameters_schema()
            }
        }

    @abstractmethod
    def _get_parameters_schema(self) -> dict:
        """Get parameters JSON schema."""
        pass
```

### 3.2 Tool Decorator

```python
from functools import wraps
from inspect import signature

def tool(func=None, *, name=None, description=None):
    """Decorator to create tool from function."""

    def decorator(fn):
        tool_name = name or fn.__name__
        tool_description = description or fn.__doc__ or ""

        # Build schema from signature
        sig = signature(fn)
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            prop = {"type": "string"}  # Default type

            if param.annotation != param.empty:
                prop["type"] = _python_type_to_json(param.annotation)

            properties[param_name] = prop

            if param.default == param.empty:
                required.append(param_name)

        @wraps(fn)
        def wrapper(**kwargs):
            return fn(**kwargs)

        wrapper.name = tool_name
        wrapper.description = tool_description
        wrapper.schema = {
            "type": "object",
            "properties": properties,
            "required": required
        }
        wrapper.run = fn

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
```

### 3.3 Tool Registry

```python
class ToolRegistry:
    """Manages available tools."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Get tool by name."""
        return self._tools.get(name)

    def get_all(self) -> list[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict]:
        """Get schemas for all tools."""
        return [tool.get_schema() for tool in self._tools.values()]
```

---

## 4. Building LLM Abstraction

### 4.1 LLM Interface

```python
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """Abstract LLM interface."""

    @abstractmethod
    def call(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs
    ) -> str:
        """Make LLM call."""
        pass

    @abstractmethod
    def stream(
        self,
        messages: list[dict],
        **kwargs
    ) -> Iterator[str]:
        """Stream LLM response."""
        pass
```

### 4.2 Provider Factory

```python
class LLMFactory:
    """Factory for creating LLM instances."""

    _providers = {
        "openai": OpenAILLM,
        "anthropic": AnthropicLLM,
        "gemini": GeminiLLM,
    }

    @classmethod
    def create(cls, model: str, **kwargs) -> BaseLLM:
        """Create LLM instance based on model name."""
        provider = cls._detect_provider(model)
        llm_class = cls._providers.get(provider)

        if llm_class is None:
            raise ValueError(f"Unknown provider: {provider}")

        return llm_class(model=model, **kwargs)

    @classmethod
    def _detect_provider(cls, model: str) -> str:
        """Detect provider from model name."""
        if "gpt" in model.lower():
            return "openai"
        if "claude" in model.lower():
            return "anthropic"
        if "gemini" in model.lower():
            return "gemini"
        raise ValueError(f"Cannot detect provider for: {model}")
```

### 4.3 OpenAI Implementation

```python
from openai import OpenAI

class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation."""

    def __init__(self, model: str = "gpt-4o", **kwargs):
        self.model = model
        self.client = OpenAI(**kwargs)

    def call(self, messages, tools=None, **kwargs):
        params = {
            "model": self.model,
            "messages": messages,
        }

        if tools:
            params["tools"] = tools

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content

    def stream(self, messages, **kwargs):
        params = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        for chunk in self.client.chat.completions.create(**params):
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

---

## 5. Building Orchestration

### 5.1 Crew Implementation

```python
class Crew:
    """Orchestrates multiple agents."""

    def __init__(
        self,
        agents: list[BaseAgent],
        tasks: list[Task],
        process: str = "sequential",
        memory: MemoryManager | None = None,
    ):
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.memory = memory

    def kickoff(self, inputs: dict = None) -> str:
        """Execute the crew."""
        inputs = inputs or {}

        if self.process == "sequential":
            return self._run_sequential(inputs)
        elif self.process == "hierarchical":
            return self._run_hierarchical(inputs)

        raise ValueError(f"Unknown process: {self.process}")

    def _run_sequential(self, inputs: dict) -> str:
        """Execute tasks sequentially."""
        context = ""

        for task in self.tasks:
            # Build context
            task_context = context
            if self.memory:
                task_context += "\n" + self.memory.get_context(task.description)

            # Execute
            result = task.agent.execute(task.description, task_context)

            # Update context
            context += f"\n\nPrevious result:\n{result}"

            # Save to memory
            if self.memory:
                self.memory.short_term.save(result, {"task": task.description})

        return result
```

### 5.2 Event System

```python
from typing import Callable

class EventBus:
    """Simple event bus for observability."""

    def __init__(self):
        self._handlers: dict[str, list[Callable]] = {}

    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def emit(self, event_type: str, data: dict):
        """Emit event to handlers."""
        for handler in self._handlers.get(event_type, []):
            handler(data)

# Usage
event_bus = EventBus()

event_bus.subscribe("agent.started", lambda d: print(f"Agent started: {d}"))
event_bus.emit("agent.started", {"agent": "researcher"})
```

---

## 6. Complete Example

```python
# Define tools
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Calculate mathematical expression."""
    return str(eval(expression))

# Define agents
researcher = ReactAgent(
    role="Researcher",
    goal="Find accurate information",
    backstory="Expert researcher with 10 years experience",
    llm=LLMFactory.create("gpt-4o"),
    tools=[search_web],
)

analyst = ReactAgent(
    role="Data Analyst",
    goal="Analyze data and provide insights",
    backstory="Expert in data analysis",
    llm=LLMFactory.create("gpt-4o"),
    tools=[calculate],
)

# Define tasks
research_task = Task(
    description="Research AI trends in healthcare",
    agent=researcher,
)

analysis_task = Task(
    description="Analyze the research findings",
    agent=analyst,
)

# Create and run crew
crew = Crew(
    agents=[researcher, analyst],
    tasks=[research_task, analysis_task],
    process="sequential",
    memory=MemoryManager(),
)

result = crew.kickoff({"topic": "AI in healthcare"})
print(result)
```

---

## 7. Key Takeaways

1. **Start Simple**: Begin with basic agent, add complexity gradually
2. **Interface First**: Define abstractions before implementations
3. **Pydantic for Data**: Use Pydantic for validation and serialization
4. **Events for Observability**: Add event emission from the start
5. **Test Each Layer**: Unit test components independently
6. **Memory is Important**: Context retention improves quality significantly

---

## File References

| Component | CrewAI Reference |
|-----------|------------------|
| Agent | `agent/core.py` |
| Memory | `memory/` |
| Tools | `tools/base_tool.py` |
| LLM | `llm.py`, `llms/` |
| Orchestration | `crew.py` |
| Events | `events/` |
