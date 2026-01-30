# Prompt Template System

## TL;DR
CrewAI sử dụng **Prompts class** để build system prompts từ agent attributes (role, goal, backstory) và tools. Templates được load từ i18n files, cho phép customization qua `system_template`, `prompt_template`, `response_template`. Output format được inject vào prompt khi cần structured response.

---

## 1. Prompt Building Flow

```mermaid
graph TB
    subgraph "Inputs"
        Agent[Agent<br/>role, goal, backstory]
        Tools[Tools<br/>descriptions]
        Task[Task<br/>description, output]
        I18N[I18N Templates]
    end

    subgraph "Prompts Class"
        Build[Prompts.task_execution()]
        Format[Format with variables]
    end

    subgraph "Output"
        System[System Message]
        User[User Message]
    end

    Agent --> Build
    Tools --> Build
    I18N --> Build

    Build --> Format
    Format --> System

    Task --> User
```

---

## 2. Prompts Class

### 2.1 Class Structure

```python
# File: lib/crewai/src/crewai/utilities/prompts.py

class Prompts:
    """Builds prompts for agent execution."""

    def __init__(
        self,
        agent: Agent,
        tools: list[BaseTool],
        i18n: I18N,
        system_template: str | None = None,
        prompt_template: str | None = None,
        response_template: str | None = None,
    ):
        self.agent = agent
        self.tools = tools
        self.i18n = i18n
        self.system_template = system_template
        self.prompt_template = prompt_template
        self.response_template = response_template

    def task_execution(self) -> str:
        """Build complete system prompt for task execution."""

        # Get base template
        if self.system_template:
            template = self.system_template
        else:
            template = self.i18n.slice("system")

        # Format with agent attributes
        prompt = template.format(
            role=self.agent.role,
            goal=self.agent.goal,
            backstory=self.agent.backstory,
        )

        # Add tools section
        if self.tools:
            tools_section = self._build_tools_section()
            prompt += "\n\n" + tools_section

        # Add response format
        if self.response_template:
            prompt += "\n\n" + self.response_template
        else:
            prompt += "\n\n" + self.i18n.slice("response_format")

        return prompt
```

### 2.2 Tools Section

```python
def _build_tools_section(self) -> str:
    """Build tools description section."""

    tools_desc = render_text_description_and_args(self.tools)
    tools_names = get_tool_names(self.tools)

    return self.i18n.slice("tools").format(
        tools=tools_desc,
        tool_names=tools_names,
    )
```

---

## 3. I18N Templates

### 3.1 System Template

```python
# File: lib/crewai/src/crewai/utilities/i18n.py

SYSTEM_TEMPLATE = """
You are {role}.

Your personal goal is: {goal}

You MUST follow these instructions:
{backstory}

To achieve your goal, you have access to the following tools:
"""
```

### 3.2 Tools Template

```python
TOOLS_TEMPLATE = """
{tools}

To use a tool, respond with:
Thought: [your reasoning]
Action: [tool name]
Action Input: [tool input as JSON]

Available tools: {tool_names}
"""
```

### 3.3 Response Format Template

```python
RESPONSE_FORMAT_TEMPLATE = """
When you have a final answer, respond with:
Thought: [your final reasoning]
Final Answer: [your complete response]

IMPORTANT: You must ALWAYS use one of the above formats.
"""
```

---

## 4. Custom Templates

### 4.1 Agent-Level Customization

```python
agent = Agent(
    role="Data Analyst",
    goal="Analyze data accurately",
    backstory="Expert in statistical analysis",

    # Custom templates
    system_template="""
    You are an expert {role} with deep knowledge in data science.

    Your mission: {goal}

    Background: {backstory}

    Always provide statistical confidence intervals.
    """,

    prompt_template="""
    Analyze the following data carefully:
    {task}

    Use the tools available to gather information.
    """,

    response_template="""
    Provide your analysis in the following format:
    ## Summary
    [Brief summary]

    ## Key Findings
    [Bullet points]

    ## Recommendations
    [Action items]
    """,
)
```

### 4.2 Template Variables

| Variable | Source | Description |
|----------|--------|-------------|
| `{role}` | Agent.role | Agent's role |
| `{goal}` | Agent.goal | Agent's objective |
| `{backstory}` | Agent.backstory | Agent's background |
| `{tools}` | Generated | Tool descriptions |
| `{tool_names}` | Generated | Tool name list |
| `{task}` | Task.description | Task to execute |
| `{memory}` | ContextualMemory | Retrieved context |

---

## 5. Task Prompt Building

### 5.1 In execute_task()

```python
# File: lib/crewai/src/crewai/agent/core.py:363-365

def execute_task(self, task, context=None):
    # Build base task prompt
    task_prompt = task.prompt()

    # Add schema if structured output
    task_prompt = build_task_prompt_with_schema(task, task_prompt, self.i18n)

    # Add context from previous tasks
    task_prompt = format_task_with_context(task_prompt, context, self.i18n)
```

### 5.2 Task Prompt Method

```python
# File: lib/crewai/src/crewai/task.py

class Task:
    def prompt(self) -> str:
        """Build task prompt string."""

        prompt = f"""
Task Description:
{self.description}

Expected Output:
{self.expected_output}
"""
        return prompt
```

### 5.3 Schema Injection

```python
# File: lib/crewai/src/crewai/utilities/prompts.py

def build_task_prompt_with_schema(
    task: Task,
    task_prompt: str,
    i18n: I18N,
) -> str:
    """Add output schema to task prompt."""

    if task.output_pydantic:
        schema = task.output_pydantic.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        task_prompt += i18n.slice("output_schema").format(
            schema=schema_str
        )

    if task.output_json:
        task_prompt += i18n.slice("output_json_instruction")

    return task_prompt
```

---

## 6. Memory & Knowledge Injection

### 6.1 Memory Context

```python
# File: lib/crewai/src/crewai/agent/core.py:367-415

if self._is_any_available_memory():
    memory = contextual_memory.build_context_for_task(task, context)

    if memory.strip():
        task_prompt += self.i18n.slice("memory").format(memory=memory)
```

### 6.2 Memory Template

```python
MEMORY_TEMPLATE = """
This is the context you're working with:
{memory}
"""
```

### 6.3 Knowledge Injection

```python
# Knowledge results appended similarly
if knowledge_results:
    task_prompt += self.i18n.slice("knowledge").format(
        knowledge=knowledge_results
    )
```

---

## 7. Complete Prompt Structure

```
┌─────────────────────────────────────────────┐
│              SYSTEM MESSAGE                  │
├─────────────────────────────────────────────┤
│ You are {role}.                             │
│                                             │
│ Your personal goal is: {goal}               │
│                                             │
│ You MUST follow these instructions:         │
│ {backstory}                                 │
│                                             │
│ TOOLS:                                      │
│ {tools_descriptions}                        │
│                                             │
│ RESPONSE FORMAT:                            │
│ {response_template}                         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│               USER MESSAGE                   │
├─────────────────────────────────────────────┤
│ Task Description:                           │
│ {task.description}                          │
│                                             │
│ Expected Output:                            │
│ {task.expected_output}                      │
│                                             │
│ Context:                                    │
│ {previous_task_outputs}                     │
│                                             │
│ Memory:                                     │
│ {contextual_memory}                         │
│                                             │
│ Knowledge:                                  │
│ {knowledge_results}                         │
│                                             │
│ Output Schema (if applicable):              │
│ {pydantic_schema}                           │
└─────────────────────────────────────────────┘
```

---

## 8. Key Takeaways

1. **Separation of Concerns**: System prompt (who) vs User prompt (what).

2. **Template System**: I18N-based với fallback defaults.

3. **Customizable**: Agent-level template overrides.

4. **Dynamic Tools**: Tool descriptions auto-generated from schemas.

5. **Memory Integration**: Context injected into user message.

6. **Schema Injection**: Structured output schema added when needed.

7. **Format Instructions**: Clear ReAct format expectations.

---

## File References

| Component | Path |
|-----------|------|
| Prompts Class | `lib/crewai/src/crewai/utilities/prompts.py` |
| I18N | `lib/crewai/src/crewai/utilities/i18n.py` |
| Task Prompt | `lib/crewai/src/crewai/task.py` |
| Agent Execute | `lib/crewai/src/crewai/agent/core.py:363-415` |
