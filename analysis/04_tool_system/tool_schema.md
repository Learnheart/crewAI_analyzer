# Tool Schema Definition

## TL;DR
CrewAI tools được định nghĩa qua **BaseTool class** hoặc **@tool decorator**. Schema được auto-generate từ Pydantic model hoặc function signature. Tools cần `name`, `description`, và `_run()` method. Schema convert sang OpenAI function calling format khi gửi đến LLM.

---

## 1. Tool Definition Options

```mermaid
graph TB
    subgraph "Definition Methods"
        BT[BaseTool Class<br/>Full control]
        TD[@tool Decorator<br/>Quick definition]
    end

    subgraph "Schema Generation"
        PS[Pydantic Schema<br/>args_schema]
        FS[Function Signature<br/>Auto-infer]
    end

    subgraph "Output"
        CST[CrewStructuredTool]
        OAI[OpenAI Function Schema]
    end

    BT --> PS
    TD --> FS
    PS --> CST
    FS --> CST
    CST --> OAI
```

---

## 2. BaseTool Class

### 2.1 Core Structure

```python
# File: lib/crewai/src/crewai/tools/base_tool.py:55-125

class BaseTool(BaseModel, ABC):
    """Abstract base class for all tools in crewAI."""

    # Required fields
    name: str = Field(
        description="The unique name of the tool"
    )
    description: str = Field(
        description="Used to tell the model how/when/why to use the tool"
    )

    # Schema definition
    args_schema: type[PydanticBaseModel] = Field(
        default=_ArgsSchemaPlaceholder,
        validate_default=True,
        description="Pydantic model for tool arguments"
    )

    # Optional configurations
    env_vars: list[EnvVar] = Field(
        default_factory=list,
        description="Environment variables used by the tool"
    )
    cache_function: Callable[..., bool] = Field(
        default=lambda _args=None, _result=None: True,
        description="Function to determine caching behavior"
    )
    result_as_answer: bool = Field(
        default=False,
        description="If True, tool result becomes final answer"
    )
    max_usage_count: int | None = Field(
        default=None,
        description="Maximum times this tool can be used"
    )
    current_usage_count: int = Field(
        default=0,
        description="Current usage count"
    )

    # Abstract method - must implement
    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronous tool execution."""
        pass

    # Optional async implementation
    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Asynchronous tool execution."""
        raise NotImplementedError("Async not implemented")
```

### 2.2 Implementation Example

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Define argument schema
class SearchToolSchema(BaseModel):
    """Schema for search tool arguments."""
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=10, description="Max results")

# Implement tool
class SearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for information"
    args_schema: type[BaseModel] = SearchToolSchema

    def _run(self, query: str, max_results: int = 10) -> str:
        """Execute web search."""
        # Implementation
        results = perform_search(query, max_results)
        return format_results(results)

    async def _arun(self, query: str, max_results: int = 10) -> str:
        """Async web search."""
        results = await async_search(query, max_results)
        return format_results(results)
```

---

## 3. @tool Decorator

### 3.1 Decorator Implementation

```python
# File: lib/crewai/src/crewai/tools/base_tool.py:450-555

def tool(
    *args: Callable[P2, R2] | str,
    result_as_answer: bool = False,
    max_usage_count: int | None = None,
) -> Tool[P2, R2] | Callable[[Callable[P2, R2]], Tool[P2, R2]]:
    """Decorator to create a Tool from a function.

    Can be used in three ways:
    1. @tool - decorator without arguments
    2. @tool("name") - decorator with custom name
    3. @tool(result_as_answer=True) - decorator with options
    """

    def _make_with_name(tool_name: str):
        def _make_tool(f: Callable[P2, R2]) -> Tool[P2, R2]:
            if f.__doc__ is None:
                raise ValueError("Function must have a docstring")

            # Extract signature
            func_sig = signature(f)
            fields: dict[str, Any] = {}

            for param_name, param in func_sig.parameters.items():
                if param_name == "return":
                    continue

                annotation = param.annotation if param.annotation != param.empty else Any

                if param.default is param.empty:
                    fields[param_name] = (annotation, ...)  # Required
                else:
                    fields[param_name] = (annotation, param.default)

            # Create Pydantic schema from fields
            class_name = "".join(tool_name.split()).title()
            args_schema = create_model(class_name, **fields)

            return Tool(
                name=tool_name,
                description=f.__doc__,
                func=f,
                args_schema=args_schema,
                result_as_answer=result_as_answer,
                max_usage_count=max_usage_count,
            )

        return _make_tool

    # Handle different usage patterns
    if len(args) == 1 and callable(args[0]):
        return _make_with_name(args[0].__name__)(args[0])

    if len(args) == 1 and isinstance(args[0], str):
        return _make_with_name(args[0])

    if len(args) == 0:
        def decorator(f):
            return _make_with_name(f.__name__)(f)
        return decorator

    raise ValueError("Invalid arguments")
```

### 3.2 Usage Examples

```python
from crewai.tools import tool

# Method 1: Simple decorator (uses function name)
@tool
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Method 2: Custom name
@tool("custom_calculator")
def my_calc(x: float, y: float) -> float:
    """Perform calculation on two numbers."""
    return x * y

# Method 3: With options
@tool(result_as_answer=True, max_usage_count=5)
def final_answer(answer: str) -> str:
    """Provide the final answer to the user."""
    return answer

# All create Tool instances
calculate_sum.run(1, 2)  # Returns 3
```

---

## 4. Auto Schema Generation

### 4.1 From Function Signature

```python
# File: lib/crewai/src/crewai/tools/base_tool.py:120-150

@field_validator("args_schema", mode="before")
@classmethod
def _default_args_schema(cls, v: type[PydanticBaseModel]) -> type[PydanticBaseModel]:
    """Auto-generate schema from _run signature if not provided."""

    if v != cls._ArgsSchemaPlaceholder:
        return v  # Schema explicitly provided

    run_sig = signature(cls._run)
    fields: dict[str, Any] = {}

    for param_name, param in run_sig.parameters.items():
        if param_name in ("self", "return"):
            continue

        annotation = param.annotation if param.annotation != param.empty else Any

        if param.default is param.empty:
            fields[param_name] = (annotation, ...)  # Required field
        else:
            fields[param_name] = (annotation, param.default)  # Optional

    return create_model(f"{cls.__name__}Schema", **fields)
```

### 4.2 Schema Example

```python
# This tool definition...
class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something"

    def _run(self, query: str, count: int = 5) -> str:
        return f"Query: {query}, Count: {count}"

# ...auto-generates this schema:
class MyToolSchema(BaseModel):
    query: str      # Required (no default)
    count: int = 5  # Optional with default
```

---

## 5. OpenAI Function Schema Conversion

### 5.1 Conversion Logic

```python
# File: lib/crewai/src/crewai/llms/providers/utils/common.py

def convert_tools_to_openai_schema(
    tools: list[CrewStructuredTool]
) -> tuple[list[dict], dict]:
    """Convert tools to OpenAI function calling format."""

    openai_tools = []
    available_functions = {}

    for tool in tools:
        # Build function schema
        function_schema = {
            "type": "function",
            "function": {
                "name": sanitize_tool_name(tool.name),
                "description": tool.description,
                "parameters": tool.args_schema.model_json_schema(),
            }
        }
        openai_tools.append(function_schema)

        # Map name to function
        available_functions[sanitize_tool_name(tool.name)] = tool.func

    return openai_tools, available_functions
```

### 5.2 Output Format

```json
{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    }
}
```

---

## 6. CrewStructuredTool

### 6.1 Class Definition

```python
# File: lib/crewai/src/crewai/tools/structured_tool.py:24-125

class CrewStructuredTool:
    """A structured tool that operates on any number of inputs."""

    def __init__(
        self,
        name: str,
        description: str,
        args_schema: type[BaseModel],
        func: Callable[..., Any],
        result_as_answer: bool = False,
        max_usage_count: int | None = None,
        current_usage_count: int = 0,
    ) -> None:
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.func = func
        self.result_as_answer = result_as_answer
        self.max_usage_count = max_usage_count
        self.current_usage_count = current_usage_count

        self._validate_function_signature()
```

### 6.2 BaseTool to CrewStructuredTool

```python
# File: lib/crewai/src/crewai/tools/base_tool.py:200-220

def to_structured_tool(self) -> CrewStructuredTool:
    """Convert BaseTool to CrewStructuredTool."""

    self._set_args_schema()  # Ensure schema is set

    structured_tool = CrewStructuredTool(
        name=self.name,
        description=self.description,
        args_schema=self.args_schema,
        func=self._run,
        result_as_answer=self.result_as_answer,
        max_usage_count=self.max_usage_count,
        current_usage_count=self.current_usage_count,
    )

    # Keep reference to original for cache_function access
    structured_tool._original_tool = self

    return structured_tool
```

---

## 7. Tool Schema Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | ✓ | Unique tool identifier |
| `description` | str | ✓ | Usage instructions for LLM |
| `args_schema` | Pydantic Model | Auto | Input parameters schema |
| `env_vars` | list[EnvVar] | ✗ | Environment variables |
| `result_as_answer` | bool | ✗ | Use result as final answer |
| `max_usage_count` | int | ✗ | Usage limit |
| `cache_function` | Callable | ✗ | Custom caching logic |

---

## 8. Key Takeaways

1. **Two Definition Methods**: BaseTool class cho full control, @tool decorator cho quick definition.

2. **Auto Schema Generation**: Schema được infer từ function signature nếu không provide.

3. **Pydantic-Based**: args_schema là Pydantic model cho validation.

4. **OpenAI Compatible**: Convert to OpenAI function calling format cho LLM.

5. **CrewStructuredTool**: Internal representation used by executor.

6. **Docstring Required**: @tool decorator requires docstring cho description.

7. **Type Annotations Important**: Parameters cần type hints cho schema generation.

---

## File References

| Component | Path |
|-----------|------|
| BaseTool | `lib/crewai/src/crewai/tools/base_tool.py:55-125` |
| @tool Decorator | `lib/crewai/src/crewai/tools/base_tool.py:450-555` |
| CrewStructuredTool | `lib/crewai/src/crewai/tools/structured_tool.py` |
| Schema Conversion | `lib/crewai/src/crewai/llms/providers/utils/common.py` |
