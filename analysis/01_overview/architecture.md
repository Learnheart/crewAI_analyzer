# CrewAI Architecture Overview

## TL;DR
CrewAI là một **multi-agent orchestration framework** cho phép tạo và điều phối nhiều AI agents làm việc cùng nhau. Kiến trúc theo pattern **hierarchical composition**: `Flow` → `Crew` → `Agent` → `Task`, với event-driven communication và pluggable components cho LLM, memory, tools.

---

## 1. High-Level Architecture Diagram

```mermaid
graph TB
    subgraph "Entry Layer"
        CLI[CLI Interface]
        API[Python API]
    end

    subgraph "Orchestration Layer"
        Flow[Flow<br/>Event-driven Workflow]
        Crew[Crew<br/>Multi-agent Coordinator]
    end

    subgraph "Agent Layer"
        Agent[Agent<br/>Autonomous Unit]
        Task[Task<br/>Execution Unit]
    end

    subgraph "Core Services"
        LLM[LLM Abstraction<br/>Multi-provider]
        Memory[Memory System<br/>Multi-tier]
        Tools[Tool System<br/>Extensible]
        Knowledge[Knowledge/RAG<br/>Vector-based]
    end

    subgraph "Infrastructure"
        Events[Event Bus<br/>Pub-Sub]
        Telemetry[Telemetry<br/>OpenTelemetry]
        MCP[MCP Client<br/>External Tools]
    end

    subgraph "External"
        LLMProviders[LLM Providers<br/>OpenAI, Anthropic, etc.]
        VectorDB[Vector DBs<br/>ChromaDB, Qdrant]
        ExternalTools[External Services]
    end

    CLI --> Flow
    API --> Flow
    API --> Crew

    Flow --> Crew
    Crew --> Agent
    Agent --> Task

    Agent --> LLM
    Agent --> Memory
    Agent --> Tools
    Agent --> Knowledge

    LLM --> LLMProviders
    Memory --> VectorDB
    Tools --> MCP
    Knowledge --> VectorDB
    MCP --> ExternalTools

    Agent -.-> Events
    Crew -.-> Events
    Flow -.-> Events
    Events -.-> Telemetry
```

---

## 2. Core Components Architecture

### 2.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────────┐  │
│  │   CLI    │  │ @project │  │     Decorators (@crew,...)   │  │
│  └──────────┘  └──────────┘  └──────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    ORCHESTRATION LAYER                          │
│  ┌─────────────────────────────┐  ┌─────────────────────────┐  │
│  │           Flow              │  │         Process         │  │
│  │  - @start, @listen, @router │  │  - Sequential           │  │
│  │  - State management         │  │  - Hierarchical         │  │
│  │  - Conditional execution    │  └─────────────────────────┘  │
│  └─────────────────────────────┘                                │
├─────────────────────────────────────────────────────────────────┤
│                      EXECUTION LAYER                            │
│  ┌──────────────────────────┐  ┌────────────────────────────┐  │
│  │          Crew            │  │           Task             │  │
│  │  - Agent coordination    │  │  - Task definition         │  │
│  │  - Task assignment       │  │  - Output validation       │  │
│  │  - Memory management     │  │  - Guardrails              │  │
│  └──────────────────────────┘  └────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                        Agent                                ││
│  │  - Role, Goal, Backstory                                    ││
│  │  - LLM integration                                          ││
│  │  - Tool execution                                           ││
│  │  - Memory access                                            ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                      SERVICE LAYER                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │   LLM    │ │  Memory  │ │  Tools   │ │  Knowledge/RAG   │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                   INFRASTRUCTURE LAYER                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │  Events  │ │Telemetry │ │   MCP    │ │     Storage      │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Relationships

```mermaid
classDiagram
    class Flow {
        +state: FlowState
        +kickoff()
        +plot()
    }

    class Crew {
        +agents: List~Agent~
        +tasks: List~Task~
        +process: Process
        +kickoff()
        +train()
        +test()
    }

    class Agent {
        +role: str
        +goal: str
        +backstory: str
        +llm: LLM
        +tools: List~Tool~
        +execute_task()
    }

    class Task {
        +description: str
        +agent: Agent
        +expected_output: str
        +execute()
    }

    class LLM {
        +model: str
        +call()
        +stream()
    }

    class Memory {
        +save()
        +search()
    }

    Flow "1" --> "*" Crew : orchestrates
    Crew "1" --> "*" Agent : manages
    Crew "1" --> "*" Task : executes
    Agent "1" --> "1" LLM : uses
    Agent "1" --> "*" Tool : has
    Agent "1" --> "1" Memory : accesses
    Task "*" --> "1" Agent : assigned to
```

---

## 3. Execution Flow

### 3.1 Standard Execution Path

```mermaid
sequenceDiagram
    participant User
    participant Crew
    participant Agent
    participant LLM
    participant Tools
    participant Memory

    User->>Crew: kickoff(inputs)
    Crew->>Crew: Initialize memory

    loop For each Task
        Crew->>Agent: execute_task(task)
        Agent->>Memory: Retrieve context
        Agent->>LLM: Generate response

        alt Tool needed
            LLM-->>Agent: Tool call request
            Agent->>Tools: Execute tool
            Tools-->>Agent: Tool result
            Agent->>LLM: Continue with result
        end

        LLM-->>Agent: Final response
        Agent->>Memory: Save to memory
        Agent-->>Crew: TaskOutput
    end

    Crew-->>User: CrewOutput
```

### 3.2 Hierarchical Process Flow

```mermaid
flowchart LR
    subgraph Manager["Manager Agent"]
        M[Plan & Delegate]
    end

    subgraph Workers["Worker Agents"]
        A1[Agent 1]
        A2[Agent 2]
        A3[Agent 3]
    end

    M --> A1
    M --> A2
    M --> A3

    A1 --> M
    A2 --> M
    A3 --> M
```

---

## 4. Key Design Decisions

### 4.1 Pydantic-First Architecture
```python
# File: lib/crewai/src/crewai/crew.py:133
class Crew(FlowTrackable, BaseModel):
    """Uses Pydantic BaseModel for:
    - Runtime type validation
    - JSON serialization
    - Configuration via Field()
    """
    tasks: list[Task] = Field(default_factory=list)
    agents: list[BaseAgent] = Field(default_factory=list)
    process: Process = Field(default=Process.sequential)
```

**Rationale**: Pydantic v2 provides:
- Strong type safety at runtime
- Automatic validation
- Easy serialization for persistence
- Clear API contracts

### 4.2 Event-Driven Observability
```python
# File: lib/crewai/src/crewai/events/event_bus.py
crewai_event_bus.emit(
    AgentExecutionStartedEvent(agent=agent, task=task)
)
```

**Rationale**: Decoupled event system enables:
- Non-invasive telemetry
- Custom listeners for logging/monitoring
- Easy integration with external observability tools

### 4.3 Provider Abstraction Pattern
```python
# LLM providers follow common interface
# File: lib/crewai/src/crewai/llms/base_llm.py
class BaseLLM:
    def call(self, messages, tools=None): ...
    def stream(self, messages): ...
```

**Rationale**: Single interface for multiple providers allows:
- Easy switching between OpenAI, Anthropic, etc.
- Consistent behavior across providers
- Simple testing with mock providers

---

## 5. Directory Structure Overview

```
lib/crewai/src/crewai/
├── __init__.py              # Public API exports
├── agent/                   # Agent core implementation
│   └── core.py             # Agent class (2,162 LOC)
├── crew.py                  # Crew orchestrator (2,039 LOC)
├── task.py                  # Task definition (1,261 LOC)
├── flow/                    # Workflow engine
│   └── flow.py             # Flow class (2,641 LOC)
├── llm.py                   # LLM wrapper (2,362 LOC)
├── llms/                    # LLM providers
│   └── providers/          # OpenAI, Anthropic, Bedrock, etc.
├── memory/                  # Memory system
│   ├── short_term/         # Conversation history
│   ├── long_term/          # Persistent storage
│   ├── entity/             # Fact extraction
│   └── external/           # Mem0 integration
├── tools/                   # Tool system
├── rag/                     # Vector embeddings
│   └── embeddings/         # 13+ embedding providers
├── knowledge/               # Knowledge base
├── events/                  # Event system
├── mcp/                     # Model Context Protocol
├── utilities/               # Helper modules (45+)
└── cli/                     # Command-line interface
```

---

## 6. Key Takeaways

1. **Composition over Inheritance**: Framework uses composition (`Flow` contains `Crew`, `Crew` contains `Agent`) rather than deep inheritance hierarchies.

2. **Plugin Architecture**: Tools, LLM providers, embedding providers, and memory backends are all pluggable via interfaces.

3. **Event-Driven Design**: Decoupled event bus enables observability without tight coupling between components.

4. **Type Safety**: Heavy use of Pydantic v2 for runtime validation and clear API contracts.

5. **Multi-Provider Support**: Abstractions allow switching between LLM providers (OpenAI, Anthropic, Bedrock, etc.) without code changes.

6. **Async-First**: Native async support throughout the codebase for better concurrency.

7. **Memory Hierarchy**: Multi-tier memory system (short-term, long-term, entity, external) for context retention.

---

## File References

| Component | File Path | LOC |
|-----------|-----------|-----|
| Agent Core | `lib/crewai/src/crewai/agent/core.py` | 2,162 |
| Crew | `lib/crewai/src/crewai/crew.py` | 2,039 |
| Task | `lib/crewai/src/crewai/task.py` | 1,261 |
| Flow | `lib/crewai/src/crewai/flow/flow.py` | 2,641 |
| LLM | `lib/crewai/src/crewai/llm.py` | 2,362 |
| Events | `lib/crewai/src/crewai/events/` | - |
| Memory | `lib/crewai/src/crewai/memory/` | - |
