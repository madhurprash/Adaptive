# Self-Healing Agent Architecture Diagrams

## System Architecture Overview

```mermaid
graph TB
    subgraph "User Interface"
        CLI[CLI Interface<br/>adaptive run]
        User[User Input]
    end

    subgraph "Configuration Layer"
        Config[config.yaml<br/>Model configs, prompts, middleware]
        EnvVars[Environment Variables<br/>API Keys, AWS Config]
    end

    subgraph "Agent Orchestration Layer"
        Router[Routing Agent<br/>Claude Haiku<br/>Intent Detection]
        StateGraph[LangGraph StateGraph<br/>Workflow Orchestration]
    end

    subgraph "Core Agents"
        InsightsAgent[Insights Agent<br/>Claude Sonnet<br/>Trace Analysis]
        EvolutionAgent[Evolution Agent<br/>Claude Sonnet<br/>Prompt Optimization]
    end

    subgraph "Platform Integration"
        MCPLangsmith[MCP Server<br/>LangSmith Tools]
        MCPLangfuse[MCP Server<br/>Langfuse Tools]
        InsightsFactory[InsightsAgentFactory<br/>Dynamic Agent Creation]
    end

    subgraph "Memory & Storage"
        AgentCore[AgentCore Memory<br/>Semantic Search<br/>Long-term Storage]
        Checkpointer[Memory Checkpointer<br/>Conversation State]
    end

    subgraph "Context Management"
        TokenLimit[Token Limit Middleware<br/>Context Monitoring]
        Summarization[Summarization Middleware<br/>History Compression]
        ToolSummarizer[Tool Response Summarizer<br/>Output Condensing]
        Prune[Pruning Middleware<br/>Artifact Removal]
    end

    subgraph "File Operations"
        FileTools[File System Tools<br/>read_file, write_file<br/>list_directory, search_files]
        HITL[Human-in-the-Loop<br/>GraphInterrupt<br/>Approval Workflow]
    end

    subgraph "External Systems"
        LangSmith[(LangSmith<br/>Observability Platform)]
        Langfuse[(Langfuse<br/>Observability Platform)]
        AgentRepo[Agent Repository<br/>Local or GitHub]
        Bedrock[Amazon Bedrock<br/>Claude Models]
    end

    CLI --> User
    User --> StateGraph
    Config --> Router
    Config --> InsightsFactory
    Config --> EvolutionAgent
    EnvVars --> MCPLangsmith
    EnvVars --> MCPLangfuse

    StateGraph --> Router
    Router -->|Route to Insights| InsightsAgent
    Router -->|Route to Evolution| EvolutionAgent

    InsightsFactory -->|Creates| InsightsAgent
    InsightsFactory -->|Connects| MCPLangsmith
    InsightsFactory -->|Connects| MCPLangfuse

    InsightsAgent --> AgentCore
    InsightsAgent --> Checkpointer

    MCPLangsmith --> LangSmith
    MCPLangfuse --> Langfuse

    InsightsAgent --> TokenLimit
    InsightsAgent --> Summarization
    InsightsAgent --> ToolSummarizer
    InsightsAgent --> Prune

    EvolutionAgent --> FileTools
    EvolutionAgent --> HITL
    FileTools --> AgentRepo

    InsightsAgent --> Bedrock
    EvolutionAgent --> Bedrock
    Router --> Bedrock

    AgentCore --> Bedrock

    style CLI fill:#e1f5ff
    style StateGraph fill:#fff4e1
    style InsightsAgent fill:#e8f5e9
    style EvolutionAgent fill:#f3e5f5
    style AgentCore fill:#fce4ec
    style Bedrock fill:#fff3e0
```

## Workflow State Machine

```mermaid
stateDiagram-v2
    [*] --> START
    START --> SelectPlatform: Initialize Session

    SelectPlatform --> GetInsights: Platform Selected<br/>(LangSmith/Langfuse)

    state GetInsights {
        [*] --> DetectIntent: User Question
        DetectIntent --> SkipInsights: Evolution Intent Detected
        DetectIntent --> GenerateInsights: Analysis Intent Detected

        GenerateInsights --> MemorySearch: Search AgentCore Memory
        MemorySearch --> CreateAgent: Create Platform-Specific Agent
        CreateAgent --> InvokeAgent: Invoke with Context
        InvokeAgent --> StreamResponse: Stream Results
        StreamResponse --> StoreMemory: Store in AgentCore Memory
        StoreMemory --> [*]

        SkipInsights --> [*]: Skip to Evolution
    }

    GetInsights --> RouteToEvolution: Insights Generated

    state RouteToEvolution {
        [*] --> AnalyzeIntent: Check User Question
        AnalyzeIntent --> CheckMarker: Check Special Marker
        CheckMarker --> ToEvolution: Evolution Needed
        CheckMarker --> ToEnd: No Evolution Needed
    }

    RouteToEvolution --> SelectRepo: Route to Evolution
    RouteToEvolution --> END: End Workflow

    SelectRepo --> EvolutionEngine: Repository Selected

    state EvolutionEngine {
        [*] --> LoadInsights: Load Analysis Results
        LoadInsights --> CreateOptAgent: Create Optimization Agent
        CreateOptAgent --> GenerateOptimizations: Analyze & Propose Changes
        GenerateOptimizations --> GraphInterrupt: Trigger HITL Review

        state GraphInterrupt {
            [*] --> ExtractPayload: Extract Change Request
            ExtractPayload --> GenerateDiff: Create Unified Diff
            GenerateDiff --> DisplayPatch: Show Changes to User
            DisplayPatch --> AwaitApproval: Request Decision
            AwaitApproval --> Approved: User Approves
            AwaitApproval --> Rejected: User Rejects

            Approved --> ApplyChanges: Write Files
            Rejected --> DiscardChanges: No Changes Made
        }

        GraphInterrupt --> [*]: Changes Applied/Rejected
    }

    EvolutionEngine --> END
    END --> [*]

    note right of SelectPlatform
        Platform selection persists
        for entire session
    end note

    note right of GetInsights
        Uses MCP tools specific
        to selected platform
    end note

    note right of EvolutionEngine
        HITL ensures human
        oversight for all changes
    end note
```

## Multi-Agent Interaction Flow

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as CLI Interface
    participant SG as StateGraph
    participant PS as Platform Selector
    participant RS as Repo Selector
    participant RT as Router Agent
    participant IA as Insights Agent
    participant MCP as MCP Server
    participant OP as Observability Platform
    participant MEM as AgentCore Memory
    participant EA as Evolution Agent
    participant FS as File System
    participant HITL as HITL Middleware

    U->>CLI: adaptive run --session-id xyz
    CLI->>SG: Initialize with thread_id

    SG->>PS: select_platform()
    PS->>U: Prompt: Select Platform
    U->>PS: Choice: 1 (LangSmith)
    PS->>SG: Update state[platform] = "langsmith"

    SG->>IA: get_insights(state)

    Note over IA: Load router prompt template
    IA->>RT: Detect user intent
    RT-->>IA: Intent: CONTINUE_WITH_INSIGHTS

    Note over IA: Create platform-specific agent
    IA->>MCP: Initialize LangSmith MCP Server
    MCP->>OP: Connect to LangSmith API
    MCP-->>IA: Return LangSmith tools

    Note over IA: Memory search (pre-model hook)
    IA->>MEM: Search relevant context
    MEM-->>IA: Return top-K memories

    Note over IA: Invoke agent with enriched context
    IA->>OP: Query traces via MCP tools
    OP-->>IA: Return trace data

    IA->>U: Stream insights in real-time

    Note over IA: Memory storage (post-model hook)
    IA->>MEM: Store conversation

    IA-->>SG: Update state with insights

    SG->>RT: route_to_evolution(state)
    RT->>RT: Analyze user question
    RT-->>SG: Decision: "adapt_prompts"

    SG->>RS: select_agent_repository()
    RS->>U: Prompt: Enter repo path/URL
    U->>RS: Path: /path/to/agent/repo
    RS-->>SG: Update state[agent_repo]

    SG->>EA: evolution_engine(state)
    EA->>EA: Create optimization agent

    Note over EA: Analyze insights and generate optimizations
    EA->>FS: read_file(prompt_template.txt)
    FS-->>EA: Current prompt content

    EA->>EA: Generate optimized prompt

    Note over EA: Trigger HITL for approval
    EA->>HITL: write_file() triggers GraphInterrupt
    HITL->>U: Display unified diff
    HITL->>U: Show statistics
    HITL->>U: Prompt: Approve/Reject?
    U->>HITL: Decision: Approve

    HITL->>FS: Apply changes
    FS-->>HITL: Files written
    HITL-->>EA: Resume execution

    EA-->>SG: Update state[evolution_status]
    SG->>U: Display completion message
```

## Memory System Architecture

```mermaid
graph TB
    subgraph "Memory Lifecycle"
        UserQ[User Question]
        PreHook[Pre-Model Hook<br/>Memory Search]
        Agent[Agent Processing]
        PostHook[Post-Model Hook<br/>Memory Storage]
    end

    subgraph "AgentCore Memory Components"
        MemStore[Memory Store<br/>BaseStore Interface]
        Search[Semantic Search<br/>Vector Embeddings]
        Storage[Conversation Storage<br/>Structured Data]
        Namespace[Namespace Management<br/>user_id, insights_namespace]
    end

    subgraph "Memory Operations"
        SearchOp[memory_store.search<br/>query, limit, namespace]
        PutOp[memory_store.put<br/>namespace, key, value]
        Embedding[Bedrock Embeddings<br/>Semantic Vectorization]
    end

    subgraph "Context Enrichment"
        Recent[Recent Messages<br/>Last N messages]
        Relevant[Relevant Context<br/>Top-K from memory]
        Combined[Enriched Context<br/>System message + recent]
    end

    UserQ --> PreHook
    PreHook --> SearchOp
    SearchOp --> Search
    Search --> Embedding
    Search --> Relevant

    Recent --> Combined
    Relevant --> Combined
    Combined --> Agent

    Agent --> PostHook
    PostHook --> PutOp
    PutOp --> Storage
    Storage --> Embedding
    Storage --> Namespace

    MemStore --> SearchOp
    MemStore --> PutOp
    Namespace --> Storage

    style PreHook fill:#e8f5e9
    style PostHook fill:#f3e5f5
    style Search fill:#fff4e1
    style Storage fill:#e1f5ff
    style Combined fill:#fce4ec
```

## Middleware Pipeline

```mermaid
graph LR
    subgraph "Message Flow"
        Input[Input Messages]
        Output[Agent Response]
    end

    subgraph "Pre-Processing Middleware"
        TL[Token Limit Check<br/>Monitor context size]
        MS[Memory Search<br/>Retrieve context]
    end

    subgraph "Agent Execution"
        Model[LLM Invocation<br/>Claude Sonnet/Haiku]
        Tools[Tool Execution<br/>MCP/File Tools]
    end

    subgraph "Post-Processing Middleware"
        TS[Tool Summarizer<br/>Condense outputs]
        CS[Conversation Summarizer<br/>Compress history]
        PM[Pruning Middleware<br/>Remove artifacts]
        Todo[TodoList Middleware<br/>Task tracking]
    end

    subgraph "HITL Gateway"
        HITL[Human-in-the-Loop<br/>Approval required]
        Interrupt[GraphInterrupt<br/>Halt execution]
    end

    Input --> MS
    MS --> TL
    TL -->|Under threshold| Model
    TL -->|Over threshold| CS
    CS --> Model

    Model --> Tools
    Tools --> TS
    TS --> PM
    PM --> Todo

    Todo -->|write_file called| HITL
    HITL --> Interrupt
    Interrupt -->|Approved| Output
    Interrupt -->|Rejected| Output

    Todo -->|Other operations| Output

    style TL fill:#fff3e0
    style MS fill:#e8f5e9
    style TS fill:#e1f5ff
    style CS fill:#f3e5f5
    style HITL fill:#ffebee
    style Interrupt fill:#ffcdd2
```

## Factory Pattern for Agent Creation

```mermaid
classDiagram
    class InsightsAgentFactory {
        -config_data: Dict
        -context_engineering_info: Dict
        -_mcp_client: MultiServerMCPClient
        +__init__(config_file_path)
        +create_insights_agent(platform) Agent
        -_initialize_middleware()
        -_create_insights_llm() ChatBedrockConverse
        -_load_platform_prompt(platform) str
        -_create_mcp_tools(platform) List~Tool~
        +cleanup()
    }

    class PromptOptimizationAgentFactory {
        -config_data: Dict
        -optimization_agent_config: Dict
        -hitl_middleware: HumanInTheLoopMiddleware
        +__init__(config_file_path)
        +create_optimization_agent(type) Agent
        -_initialize_middleware()
        -_create_optimization_llm() ChatBedrockConverse
        -_load_optimization_prompt(type) str
        -_create_file_tools() List~Tool~
        -_load_hitl_middleware(config_file) Middleware
    }

    class ObservabilityPlatform {
        <<enumeration>>
        LANGSMITH
        LANGFUSE
    }

    class OfflineOptimizationType {
        <<enumeration>>
        SYSTEM_PROMPT
    }

    class Agent {
        +model: ChatBedrockConverse
        +tools: List~Tool~
        +system_prompt: str
        +middleware: List~Middleware~
        +astream() AsyncGenerator
        +ainvoke() Dict
    }

    class MultiServerMCPClient {
        +get_tools() List~Tool~
        +close()
    }

    class HumanInTheLoopMiddleware {
        +interrupt_on: Dict
        +__call__()
    }

    InsightsAgentFactory --> ObservabilityPlatform
    InsightsAgentFactory --> Agent : creates
    InsightsAgentFactory --> MultiServerMCPClient : uses

    PromptOptimizationAgentFactory --> OfflineOptimizationType
    PromptOptimizationAgentFactory --> Agent : creates
    PromptOptimizationAgentFactory --> HumanInTheLoopMiddleware : uses

    Agent --> "1..*" Middleware : applies
```

## Configuration-Driven Architecture

```mermaid
graph TD
    subgraph "Configuration Files"
        MainConfig[config.yaml<br/>Main Configuration]
        HITLConfig[hitl_config.yaml<br/>HITL Rules]
        EnvFile[.env<br/>Secrets & Keys]
    end

    subgraph "Configuration Sections"
        InsightsConfig[insights_agent_model_information<br/>Model ID, Prompts, Parameters]
        EvolutionConfig[prompt_optimization_agent_model_information<br/>Optimization Settings]
        RouterConfig[routing_configuration<br/>Router Model, Prompts]
        MemoryConfig[agentcore_memory<br/>Memory Settings]
        ContextConfig[context_engineering_info<br/>Middleware Settings]
    end

    subgraph "Runtime Components"
        InsightsAgent[Insights Agent]
        EvolutionAgent[Evolution Agent]
        RouterAgent[Router Agent]
        MemorySystem[AgentCore Memory]
        Middleware[Middleware Stack]
    end

    MainConfig --> InsightsConfig
    MainConfig --> EvolutionConfig
    MainConfig --> RouterConfig
    MainConfig --> MemoryConfig
    MainConfig --> ContextConfig

    InsightsConfig --> InsightsAgent
    EvolutionConfig --> EvolutionAgent
    RouterConfig --> RouterAgent
    MemoryConfig --> MemorySystem
    ContextConfig --> Middleware

    HITLConfig --> EvolutionAgent
    EnvFile --> InsightsAgent
    EnvFile --> MemorySystem

    style MainConfig fill:#fff4e1
    style InsightsAgent fill:#e8f5e9
    style EvolutionAgent fill:#f3e5f5
    style MemorySystem fill:#fce4ec
```
