# Self-Healing Agent Architecture Design

## Overview

This document describes the architecture for a self-improving LangGraph agent that uses observability traces from LangSmith to evaluate and optimize its own performance through Agent Context Engineering (ACE) and Dynamic Cheatsheet (DC) techniques.

## Problem Statement

Current agent development workflow has a gap:
1. **Agent Operates** → CloudWatch monitoring, Lambda auto-tuning, etc.
2. **Traces Collected** → LangSmith for observability
3. **Gap**: No automated mechanism to use traces for self-improvement
4. **Desired**: Agent should self-improve by:
   - Analyzing its own traces
   - Responding to user feedback
   - Updating prompts, memory, and strategies
   - Operating in both offline (batch) and online (real-time) modes

## Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Self-Healing Agent System                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  Operational     │         │   LangSmith      │          │
│  │  Agent           │────────>│   Traces         │          │
│  │  (LangGraph)     │         │   & Metrics      │          │
│  └──────────────────┘         └──────────────────┘          │
│         │                              │                     │
│         │                              ▼                     │
│         │                    ┌──────────────────┐           │
│         │                    │   Evaluation     │           │
│         │                    │   Module         │           │
│         │                    └──────────────────┘           │
│         │                              │                     │
│         │                              ▼                     │
│         │                    ┌──────────────────┐           │
│         │                    │  Self-Improvement│           │
│         │                    │  Engine          │           │
│         │                    │  ┌────────────┐ │           │
│         │                    │  │ ACE: Context│ │           │
│         │                    │  │ Engineering │ │           │
│         │                    │  └────────────┘ │           │
│         │                    │  ┌────────────┐ │           │
│         │                    │  │ DC: Dynamic │ │           │
│         │                    │  │ Cheatsheet  │ │           │
│         │                    │  └────────────┘ │           │
│         │                    └──────────────────┘           │
│         │                              │                     │
│         └──────────────────────────────┘                     │
│                  (Updates prompts, memory, strategies)       │
│                                                               │
│  ┌─────────────────────────────────────────────┐            │
│  │  User Interaction Layer                      │            │
│  │  - Query traces                              │            │
│  │  - Request improvements                      │            │
│  │  - Provide feedback                          │            │
│  └─────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Operational Agent (LangGraph)

The main agent performing domain-specific tasks (e.g., CloudWatch monitoring, Lambda tuning).

**Key Features:**
- Built using LangGraph for stateful, graph-based execution
- All executions traced to LangSmith
- Consults Dynamic Cheatsheet during inference
- Uses optimized context from ACE

**State Schema:**
```python
class AgentState(TypedDict):
    messages: List[BaseMessage]
    cheatsheet: Dict[str, Any]  # Dynamic memory
    context: Dict[str, Any]      # ACE-optimized context
    task: str
    execution_history: List[Dict]
    improvement_suggestions: List[str]
```

### 2. LangSmith Integration

**Trace Collection:**
- Every agent execution traced with metadata
- Custom metadata tags: task_type, success, error_type, latency
- Trace grouping by session/task

**Metrics Tracked:**
- Task success rate
- Error patterns
- Tool usage patterns
- Execution latency
- User feedback scores

### 3. Evaluation Module

Analyzes traces to assess agent performance and identify improvement opportunities.

**Evaluation Dimensions:**
1. **Task Success Rate**: % of successful completions
2. **Error Analysis**: Common failure patterns
3. **Efficiency**: Tool calls, latency, token usage
4. **User Feedback**: Explicit ratings and corrections
5. **Pattern Recognition**: Recurring scenarios

**Evaluation Modes:**
- **Offline**: Batch analysis of historical traces
- **Online**: Real-time evaluation after each execution

### 4. Self-Improvement Engine

#### 4.1 Agent Context Engineering (ACE)

Iteratively refines the agent's contextual elements based on performance feedback.

**Components to Optimize:**
- **System Prompt**: Core instructions and persona
- **Task Instructions**: Specific task guidance
- **Few-Shot Examples**: In-context learning examples
- **Tool Descriptions**: Clarified tool usage patterns

**Optimization Process:**
1. Collect traces for a task category
2. Analyze successful vs. failed executions
3. Identify context patterns in successful cases
4. Generate prompt variations
5. Test variations on held-out traces
6. Deploy best-performing context

**Update Triggers:**
- Success rate drops below threshold
- New error pattern emerges
- User requests optimization
- Periodic scheduled optimization

#### 4.2 Dynamic Cheatsheet (DC)

Maintains an evolving memory of successful strategies, code snippets, and problem-solving insights.

**Cheatsheet Structure:**
```python
{
    "strategies": [
        {
            "id": "str_001",
            "task_type": "lambda_memory_optimization",
            "pattern": "When memory usage > 80% for 5 consecutive invocations",
            "action": "Increase memory by 128MB increments",
            "success_rate": 0.92,
            "usage_count": 15,
            "last_used": "2025-01-07T10:30:00Z",
            "created": "2025-01-05T14:20:00Z"
        }
    ],
    "code_snippets": [
        {
            "id": "code_001",
            "purpose": "cloudwatch_query_optimization",
            "snippet": "...",
            "language": "python",
            "success_rate": 0.88,
            "usage_count": 23
        }
    ],
    "heuristics": [
        {
            "id": "heur_001",
            "condition": "High latency in Lambda",
            "rule": "Check VPC configuration before memory adjustment",
            "confidence": 0.85,
            "usage_count": 8
        }
    ],
    "failure_patterns": [
        {
            "id": "fail_001",
            "pattern": "Timeout when querying CloudWatch for >7 day period",
            "lesson": "Split queries into 24-hour chunks",
            "last_seen": "2025-01-06T16:45:00Z"
        }
    ]
}
```

**Cheatsheet Operations:**

1. **Curation** (After each execution):
   ```python
   def curate_cheatsheet(
       execution_result: Dict,
       cheatsheet: Dict
   ) -> Dict:
       """
       Decides what to add, update, or remove from cheatsheet.
       """
       if execution_result["success"]:
           # Extract successful pattern
           pattern = extract_pattern(execution_result)
           if is_generalizable(pattern):
               add_or_update_entry(cheatsheet, pattern)
       else:
           # Record failure pattern
           failure = extract_failure_pattern(execution_result)
           add_failure_lesson(cheatsheet, failure)

       # Prune low-performing entries
       prune_entries(cheatsheet, min_success_rate=0.6)

       return cheatsheet
   ```

2. **Retrieval** (Before execution):
   ```python
   def retrieve_relevant_entries(
       task: str,
       cheatsheet: Dict,
       top_k: int = 3
   ) -> List[Dict]:
       """
       Retrieve top-k most relevant cheatsheet entries.
       Uses embedding-based similarity.
       """
       task_embedding = embed(task)
       entries = []

       for category in ["strategies", "code_snippets", "heuristics"]:
           for entry in cheatsheet[category]:
               similarity = cosine_similarity(
                   task_embedding,
                   entry["embedding"]
               )
               entries.append((similarity, entry))

       # Sort by similarity and success_rate
       entries.sort(key=lambda x: (x[0], x[1]["success_rate"]), reverse=True)

       return [entry for _, entry in entries[:top_k]]
   ```

3. **Update Mechanism**:
   - **Online**: After each execution, curator evaluates and updates
   - **Offline**: Periodic batch curation to refine entries
   - **Decay**: Reduce confidence of unused entries over time
   - **Promotion**: Promote frequently successful patterns

### 5. User Interaction Layer

Enables users to query, understand, and guide improvements.

**Capabilities:**
1. **Query Traces**: "Show me all failed Lambda optimizations this week"
2. **Explain Decisions**: "Why did you increase memory instead of timeout?"
3. **Request Changes**: "Always check VPC config before memory changes"
4. **Review Cheatsheet**: "What patterns have you learned about CloudWatch queries?"
5. **Approve Improvements**: Review and approve context changes

## LangGraph Implementation

### Graph Structure

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    messages: List[BaseMessage]
    cheatsheet: Dict[str, Any]
    context: Dict[str, Any]
    task: str
    execution_history: List[Dict]
    improvement_suggestions: List[str]
    needs_improvement: bool

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve_from_cheatsheet", retrieve_from_cheatsheet)
workflow.add_node("execute_task", execute_task)
workflow.add_node("evaluate", evaluate_execution)
workflow.add_node("curate_cheatsheet", curate_cheatsheet)
workflow.add_node("optimize_context", optimize_context)
workflow.add_node("user_interaction", handle_user_interaction)

# Define edges
workflow.set_entry_point("retrieve_from_cheatsheet")

workflow.add_edge("retrieve_from_cheatsheet", "execute_task")
workflow.add_edge("execute_task", "evaluate")

workflow.add_conditional_edges(
    "evaluate",
    should_improve,
    {
        "curate": "curate_cheatsheet",
        "optimize": "optimize_context",
        "user": "user_interaction",
        "end": END
    }
)

workflow.add_edge("curate_cheatsheet", END)
workflow.add_edge("optimize_context", END)
workflow.add_edge("user_interaction", END)

# Compile
app = workflow.compile()
```

### Key Node Functions

#### 1. Retrieve from Cheatsheet
```python
def retrieve_from_cheatsheet(state: AgentState) -> AgentState:
    """
    Retrieve relevant patterns from cheatsheet before execution.
    """
    task = state["task"]
    cheatsheet = state["cheatsheet"]

    relevant_entries = retrieve_relevant_entries(
        task=task,
        cheatsheet=cheatsheet,
        top_k=3
    )

    # Add to context
    state["context"]["cheatsheet_entries"] = relevant_entries

    return state
```

#### 2. Execute Task
```python
def execute_task(state: AgentState) -> AgentState:
    """
    Execute the main task using current context and cheatsheet.
    """
    # Build prompt with context and cheatsheet entries
    prompt = build_prompt(
        task=state["task"],
        context=state["context"],
        cheatsheet_entries=state["context"]["cheatsheet_entries"]
    )

    # Execute with tracing
    with tracing_context(run_name="operational_execution"):
        result = agent.invoke(prompt)

    # Record execution
    state["execution_history"].append({
        "timestamp": datetime.now(),
        "task": state["task"],
        "result": result,
        "cheatsheet_used": state["context"]["cheatsheet_entries"]
    })

    return state
```

#### 3. Evaluate Execution
```python
def evaluate_execution(state: AgentState) -> AgentState:
    """
    Evaluate the execution and determine if improvement is needed.
    """
    last_execution = state["execution_history"][-1]

    evaluation = {
        "success": last_execution["result"]["success"],
        "efficiency_score": calculate_efficiency(last_execution),
        "error_type": last_execution["result"].get("error"),
        "improvement_opportunities": []
    }

    # Check for improvement triggers
    if not evaluation["success"]:
        evaluation["improvement_opportunities"].append(
            "failure_pattern_detected"
        )

    if evaluation["efficiency_score"] < 0.7:
        evaluation["improvement_opportunities"].append(
            "low_efficiency"
        )

    # Store evaluation
    state["execution_history"][-1]["evaluation"] = evaluation
    state["needs_improvement"] = len(evaluation["improvement_opportunities"]) > 0

    return state
```

#### 4. Curate Cheatsheet
```python
def curate_cheatsheet(state: AgentState) -> AgentState:
    """
    Update cheatsheet based on execution results.
    """
    last_execution = state["execution_history"][-1]
    cheatsheet = state["cheatsheet"]

    # Curator LLM call
    curator_prompt = f"""
    Analyze this execution and update the cheatsheet:

    Task: {last_execution["task"]}
    Result: {last_execution["result"]}
    Success: {last_execution["evaluation"]["success"]}

    Current Cheatsheet Entries Used:
    {json.dumps(last_execution["cheatsheet_used"], indent=2)}

    Decide:
    1. Should we add a new strategy/pattern?
    2. Should we update existing entries?
    3. Should we add a failure lesson?
    4. Should we prune any low-performing entries?

    Focus on concise, generalizable insights.
    """

    updates = curator_llm.invoke(curator_prompt)

    # Apply updates
    state["cheatsheet"] = apply_cheatsheet_updates(
        cheatsheet,
        updates
    )

    # Persist to storage
    save_cheatsheet(state["cheatsheet"])

    return state
```

#### 5. Optimize Context (ACE)
```python
def optimize_context(state: AgentState) -> AgentState:
    """
    Optimize agent context using ACE techniques.
    """
    # Fetch recent traces from LangSmith
    traces = fetch_langsmith_traces(
        task_type=state["task"],
        lookback_days=7
    )

    # Analyze successful vs failed executions
    analysis = analyze_traces(traces)

    # Generate context variations
    variations = generate_context_variations(
        current_context=state["context"],
        analysis=analysis
    )

    # Test variations (offline or on held-out data)
    best_context = evaluate_context_variations(variations)

    if best_context["performance"] > state["context"]["performance"]:
        state["context"] = best_context
        state["improvement_suggestions"].append(
            f"Context optimized: {best_context['improvement_description']}"
        )

        # Persist updated context
        save_context(best_context)

    return state
```

#### 6. User Interaction
```python
def handle_user_interaction(state: AgentState) -> AgentState:
    """
    Handle user queries and feedback.
    """
    # This would be triggered by user input
    # For now, just log improvement suggestions

    if state["improvement_suggestions"]:
        logger.info(
            f"Improvement suggestions:\n" +
            "\n".join(f"- {s}" for s in state["improvement_suggestions"])
        )

    return state
```

### Conditional Logic
```python
def should_improve(state: AgentState) -> str:
    """
    Determine next step after evaluation.
    """
    if not state["needs_improvement"]:
        return "end"

    last_execution = state["execution_history"][-1]
    opportunities = last_execution["evaluation"]["improvement_opportunities"]

    # Priority: failure pattern > low efficiency > user feedback
    if "failure_pattern_detected" in opportunities:
        return "curate"

    if "low_efficiency" in opportunities:
        # Check if we should optimize context
        recent_success_rate = calculate_recent_success_rate(state)
        if recent_success_rate < 0.7:
            return "optimize"
        else:
            return "curate"

    # Default: user interaction for feedback
    return "user"
```

## Improvement Modes

### Online Mode (Real-time)

Improvements happen during or immediately after each execution.

**Characteristics:**
- Low latency requirement
- Lightweight curation
- Immediate cheatsheet updates
- No context optimization (too slow)

**Use Cases:**
- Production agents needing continuous operation
- Quick pattern learning
- Failure recovery

### Offline Mode (Batch)

Periodic batch processing of traces for deep optimization.

**Characteristics:**
- Comprehensive analysis
- Context optimization via ACE
- Cheatsheet refinement and pruning
- A/B testing of variations

**Use Cases:**
- Weekly/daily optimization cycles
- Major context updates
- Performance tuning

## Integration with LangSmith

### Trace Collection
```python
from langsmith import Client
from langsmith.run_helpers import traceable

langsmith_client = Client()

@traceable(
    run_type="chain",
    name="operational_agent",
    project_name="self-healing-agent"
)
def run_agent_with_tracing(task: str, state: AgentState):
    """
    Execute agent with full tracing to LangSmith.
    """
    # Add custom metadata
    metadata = {
        "task_type": classify_task(task),
        "cheatsheet_version": state["cheatsheet"]["version"],
        "context_version": state["context"]["version"]
    }

    return app.invoke(state, config={"metadata": metadata})
```

### Trace Analysis
```python
def fetch_and_analyze_traces(
    project_name: str = "self-healing-agent",
    lookback_days: int = 7
) -> Dict[str, Any]:
    """
    Fetch traces from LangSmith and analyze patterns.
    """
    # Fetch traces
    runs = langsmith_client.list_runs(
        project_name=project_name,
        start_time=datetime.now() - timedelta(days=lookback_days)
    )

    # Group by task type
    traces_by_task = defaultdict(list)
    for run in runs:
        task_type = run.extra.get("metadata", {}).get("task_type")
        traces_by_task[task_type].append(run)

    # Analyze each task type
    analysis = {}
    for task_type, task_runs in traces_by_task.items():
        analysis[task_type] = {
            "total_runs": len(task_runs),
            "success_rate": calculate_success_rate(task_runs),
            "avg_latency": calculate_avg_latency(task_runs),
            "common_errors": extract_common_errors(task_runs),
            "tool_usage_patterns": analyze_tool_usage(task_runs)
        }

    return analysis
```

### User Query Interface
```python
def query_traces(user_query: str) -> str:
    """
    Allow users to query traces using natural language.
    """
    # Parse user query
    query_params = parse_user_query(user_query)

    # Fetch relevant traces
    traces = langsmith_client.list_runs(
        project_name="self-healing-agent",
        filter=query_params["filter"],
        start_time=query_params.get("start_time"),
        end_time=query_params.get("end_time")
    )

    # Summarize for user
    summary = summarize_traces(traces)

    return summary
```

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Set up LangGraph basic structure
- [ ] Integrate LangSmith tracing
- [ ] Implement basic operational agent
- [ ] Design cheatsheet schema
- [ ] Set up storage (JSON file or database)

### Phase 2: Dynamic Cheatsheet (Week 3-4)
- [ ] Implement cheatsheet retrieval
- [ ] Build curation module
- [ ] Add embedding-based similarity search
- [ ] Implement online update mechanism
- [ ] Test cheatsheet evolution

### Phase 3: ACE Integration (Week 5-6)
- [ ] Build trace analysis module
- [ ] Implement context variation generation
- [ ] Create evaluation framework
- [ ] Deploy offline optimization pipeline
- [ ] A/B test context improvements

### Phase 4: User Interaction (Week 7-8)
- [ ] Build query interface for traces
- [ ] Implement feedback collection
- [ ] Add approval workflow for improvements
- [ ] Create dashboard for monitoring
- [ ] Documentation and examples

## File Structure

```
self-healing-agent/
├── agent.py                          # Main agent entry point
├── agent_tools/
│   ├── __init__.py
│   ├── operational_tools.py          # Domain-specific tools (CloudWatch, Lambda)
│   └── langsmith_tools.py            # LangSmith integration tools
├── graph/
│   ├── __init__.py
│   ├── state.py                      # AgentState definition
│   ├── nodes.py                      # Graph node functions
│   ├── edges.py                      # Conditional logic
│   └── workflow.py                   # Graph construction
├── cheatsheet/
│   ├── __init__.py
│   ├── schema.py                     # Cheatsheet data models
│   ├── curator.py                    # Curation logic
│   ├── retriever.py                  # Retrieval with embeddings
│   └── storage.py                    # Persistence layer
├── ace/
│   ├── __init__.py
│   ├── trace_analyzer.py             # LangSmith trace analysis
│   ├── context_optimizer.py          # ACE implementation
│   └── evaluator.py                  # Context evaluation
├── user_interface/
│   ├── __init__.py
│   ├── query_handler.py              # Natural language queries
│   ├── feedback_collector.py         # User feedback
│   └── dashboard.py                  # Monitoring dashboard (optional)
├── config.yaml                       # Configuration
├── constants.py                      # Constants
├── utils.py                          # Utility functions
├── pyproject.toml                    # Dependencies
└── README.md                         # Documentation
```

## Key Design Decisions

### 1. Why LangGraph?
- **Stateful**: Maintains agent state across executions
- **Flexible**: Easy to add/modify nodes
- **Observable**: Native LangSmith integration
- **Production-ready**: Streaming, checkpointing, error handling

### 2. Why Separate Cheatsheet and Context?
- **Cheatsheet (DC)**: Task-specific, fast retrieval, high churn
- **Context (ACE)**: Agent-wide, slower optimization, stable
- Different update frequencies and mechanisms

### 3. Online vs Offline
- **Online**: Fast feedback loop, immediate learning
- **Offline**: Deep optimization, requires compute
- Hybrid approach balances responsiveness and quality

### 4. Storage Strategy
- **Cheatsheet**: Fast key-value store (Redis) or vector DB (Pinecone, Weaviate)
- **Context**: Version-controlled (Git) or config management
- **Traces**: LangSmith (managed)

## Evaluation Metrics

### Agent Performance
- **Task Success Rate**: Overall and per-task-type
- **Error Rate**: Failures per task type
- **Efficiency**: Tokens used, tool calls, latency
- **User Satisfaction**: Explicit feedback scores

### Self-Improvement Effectiveness
- **Improvement Velocity**: Rate of context/cheatsheet updates
- **Improvement Impact**: Delta in success rate after updates
- **Cheatsheet Utility**: Usage rate of entries
- **Context Stability**: How often context changes

### System Health
- **Update Latency**: Time from execution to improvement
- **Storage Growth**: Cheatsheet size over time
- **Retrieval Speed**: Cheatsheet query latency
- **Trace Analysis Cost**: LangSmith API calls

## Security and Privacy

### Considerations
1. **PII in Traces**: Sanitize before storage
2. **Cheatsheet Access**: Role-based access control
3. **Context Updates**: Approval workflow for production
4. **API Keys**: Secure storage (AWS Secrets Manager, etc.)

### Best Practices
- Encrypt cheatsheet storage
- Audit log for all improvements
- Rollback mechanism for bad updates
- Rate limiting on improvement frequency

## Next Steps

1. **Review this design** with team
2. **Set up repository** with file structure
3. **Implement Phase 1** foundation
4. **Define specific use case** (e.g., CloudWatch + Lambda)
5. **Iterate** based on learnings

## References

1. **Agent Context Engineering**: [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)
2. **Dynamic Cheatsheet**: [arXiv:2504.07952](https://arxiv.org/abs/2504.07952)
3. **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
4. **LangSmith Documentation**: https://docs.smith.langchain.com/

---

**Document Version**: 1.0
**Last Updated**: 2025-01-07
**Author**: AI Assistant
**Status**: Draft for Review
