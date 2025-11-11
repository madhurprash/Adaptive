"""
Self-Healing Agent - Log Curator

This is the agent that is responsible to do the following high value tasks:

1. Collect agent traces. A requirement is that any agent that is running or any
multi-agent system's traces are being logged in LangSmith or another observability
platform.
2. Curate logs based on user questions and needs.
3. Answer user questions about agent execution traces.
"""
import os
import sys
import json
import yaml
import uuid
import logging
import argparse
from utils import *
from constants import *
from typing import Annotated
from dotenv import load_dotenv
from langsmith import traceable
from typing_extensions import TypedDict
from langchain.agents import create_agent
from typing import Any, Dict, List, Optional
from langchain_aws import ChatBedrockConverse
from langgraph.graph.message import add_messages
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents.middleware import (
    SummarizationMiddleware,
    TodoListMiddleware,
)
from deepagents import create_deep_agent
from deepagents.backends import StateBackend, FilesystemBackend
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage

# Import custom middleware
from agent_middleware.context_middleware import (
    TokenLimitCheckMiddleware,
    tool_response_summarizer,
)

from agent_tools.langsmith_tools import (
    LangSmithConfig,
    get_session_info,
    get_runs_from_session,
    get_session_metadata,
    get_session_insights,
    list_sessions,
    list_session_runs_summary,
    get_run_details,
    get_latest_run,
    get_latest_error_run,
    get_run_error_only,
)

from agent_tools.file_tools import (
    write_file,
    read_file,
)

# Load environment variables from .env file
load_dotenv()

# Import Tavily for internet search
from tavily import TavilyClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# load the config data that contains information about the agents and the models
config_data: Dict = load_config(CONFIG_FILE_FPATH)
logger.info(f"Loaded the configuration file: {json.dumps(config_data, indent=4)}")

# initialize the context engineering information
context_engineering_info: Dict = config_data['context_engineering_info']

# ---------------- ROUTING CONFIGURATION ----------------
"""
This section initializes the routing LLM that decides whether to invoke
the deep research agent based on the user's question and available logs.
"""
routing_config: Dict = config_data.get('routing_configuration', {})
router_model_id = routing_config.get('router_model_id', 'us.anthropic.claude-3-5-haiku-20241022-v1:0')
router_inference_params = routing_config.get('inference_parameters', {})

# Initialize routing LLM (small, fast model for routing decisions)
router_llm = ChatBedrockConverse(
    model=router_model_id,
    temperature=router_inference_params.get('temperature', 0.1),
    max_tokens=router_inference_params.get('max_tokens', 500),
    top_p=router_inference_params.get('top_p', 0.92),
)
logger.info(f"Initialized routing LLM: {router_llm}")

# Load routing prompt
router_prompt_path = routing_config.get('router_prompt_path', 'prompt_templates/routing_decision_prompt.txt')
router_prompt_template = ""
try:
    router_prompt_template = load_system_prompt(router_prompt_path)
    logger.info(f"Loaded routing prompt from: {router_prompt_path}")
except Exception as e:
    logger.warning(f"Could not load routing prompt: {e}")
    router_prompt_template = "Determine if deep research is needed. Respond with 'ROUTE_TO_RESEARCH' or 'END'."

# ---------------- AGENT 1: INITIALIZE THE INSIGHTS AGENT ----------------
"""
This agent is responsible for fetching insights about agents from langSmith
"""

# initialize the insights agent model information
insights_agent_model_config: Dict = config_data["insights_agent_model_information"]
# Initialize LLM
insights_llm = ChatBedrockConverse(
    model=insights_agent_model_config["model_id"],
    temperature=insights_agent_model_config["inference_parameters"]["temperature"],
    max_tokens=insights_agent_model_config["inference_parameters"]["max_tokens"],
    top_p=insights_agent_model_config["inference_parameters"]["top_p"],
)
logger.info(f"Initialized the insights agent LLM: {insights_llm}")

# initialize the insights agent base system prompt
insights_agent_base_prompt: str = load_system_prompt(insights_agent_model_config.get('insights_agent_prompt'))
logger.info(f"Loaded the insights agent base system prompt")

# ---------------- INITIALIZE SUMMARIZATION MIDDLEWARE ----------------
"""
Create a lightweight LLM for conversation summarization.
Uses configuration from context_engineering_info for optimal summarization.
"""
logger.info("Initializing summarization LLM for middleware...")

# Get summarization middleware configuration
summarization_config: Dict = context_engineering_info.get('summarization_middleware')
logger.info(f"Loaded summarization middleware config:\n{json.dumps(summarization_config, indent=4)}")

# Initialize the summarization LLM with config values
summarization_llm = ChatBedrockConverse(
    model=summarization_config.get("model", insights_agent_model_config["model_id"]),
    temperature=summarization_config.get('temperature', 0.1),  # Lower temperature for consistent summaries
    max_tokens=summarization_config.get('max_tokens', 2000),  # Sufficient for summaries
)
logger.info(f"Initialized summarization LLM with model: {summarization_config.get('model')}")

# Load the custom summarization prompt
summary_prompt_path: str = summarization_config.get('summary_prompt')
summary_prompt_text: Optional[str] = None

if summary_prompt_path:
    try:
        summary_prompt_text = load_system_prompt(summary_prompt_path)
        logger.info(f"Loaded custom summarization prompt from: {summary_prompt_path}")
    except Exception as e:
        logger.warning(f"Could not load summary prompt from {summary_prompt_path}: {e}. Using default.")
        summary_prompt_text = None
else:
    logger.info("No custom summary prompt configured, using LangChain default")

# Create the summarization middleware with config values
logger.info("Creating SummarizationMiddleware...")
middleware_params = {
    "model": summarization_llm,
    "max_tokens_before_summary": summarization_config.get("max_tokens_before_summary", 4000),
    "messages_to_keep": summarization_config.get("messages_to_keep", 20),
}

# Add custom summary prompt if available
if summary_prompt_text:
    middleware_params["summary_prompt"] = summary_prompt_text

conversation_summarization_middleware = SummarizationMiddleware(**middleware_params)
logger.info(f"SummarizationMiddleware created successfully with max_tokens_before_summary={middleware_params['max_tokens_before_summary']}, messages_to_keep={middleware_params['messages_to_keep']}")


# ---------------- INITIALIZE CUSTOM TOKEN LIMIT MIDDLEWARE ----------------
"""
Custom middleware to check if input tokens exceed 100k before model calls.
If exceeded, triggers automatic summarization.
"""
logger.info("Creating TokenLimitCheckMiddleware...")
token_limit_middleware = TokenLimitCheckMiddleware(
    model=insights_llm,
    summarization_llm=summarization_llm,
    token_threshold=config_data['context_engineering_info'].get('token_threshold', 100000),
    summary_prompt=summary_prompt_text,
)
logger.info("TokenLimitCheckMiddleware created successfully")


# ---------------- INITIALIZE TOOL RESPONSE SUMMARIZER MIDDLEWARE ----------------
"""
Middleware to automatically summarize large tool responses to prevent context overflow.
Uses the SAME token threshold, summarization LLM, and summarization prompt as TokenLimitCheckMiddleware
to ensure consistent context management across all components.
"""
logger.info("Creating tool_response_summarizer middleware...")

# Use the same token threshold as TokenLimitCheckMiddleware (100k tokens)
tool_response_token_threshold = config_data['context_engineering_info'].get('token_threshold', 100000)

# Create the tool response summarizer middleware with same config as TokenLimitCheckMiddleware
tool_summarizer_middleware = tool_response_summarizer(
    summarization_llm=summarization_llm,
    token_threshold=tool_response_token_threshold,  # Same 100k threshold
    store_full_responses=True,  # Store full responses for retrieval if needed
    summary_prompt=summary_prompt_text,  # Same summarization prompt
)
logger.info(
    f"Tool response summarizer created with:\n"
    f"  - token_threshold={tool_response_token_threshold} (same as TokenLimitCheckMiddleware)\n"
    f"  - Using same summarization prompt and LLM"
)


# ---------------- INITIALIZE TODO LIST MIDDLEWARE ----------------
"""
Todo list middleware to equip the agent with task planning and tracking capabilities.
"""
logger.info("Creating TodoListMiddleware...")
todo_list_middleware = TodoListMiddleware()
logger.info("TodoListMiddleware created successfully")


# Create tools list
langsmith_tools = [
    get_session_info,
    get_runs_from_session,
    get_session_metadata,
    get_session_insights,
    list_sessions,
    list_session_runs_summary,
    get_run_details,
    get_latest_run,
    get_latest_error_run,
    get_run_error_only,
]

# Create the insights agent using create_react_agent with middleware
# Middleware order matters: they execute in the order listed
# 1. TokenLimitCheckMiddleware - checks 100k token limit first
# 2. ToolResponseSummarizer - summarizes large tool outputs
# 3. TodoListMiddleware - provides task planning and tracking
# 4. SummarizationMiddleware - handles regular conversation summarization
logger.info("Creating insights agent with LangSmith tools and middleware stack...")
insights_agent = create_agent(
    model=insights_llm,
    tools=langsmith_tools,
    system_prompt=insights_agent_base_prompt,
    middleware=[
        # this middleware is used to check if the messages or the context window exceeds 100k tokens
        token_limit_middleware,
        # this middleware summarizes large tool responses to prevent context overflow
        tool_summarizer_middleware,
        # this is a to do list middleware
        todo_list_middleware,
        # this is the summarization middleware that will be used to summarize the current conversation with the user
        conversation_summarization_middleware,
    ],
)
logger.info(
    "Created the insights agent with middleware stack:\n"
    "  1. TokenLimitCheckMiddleware (100k token threshold)\n"
    "  2. ToolResponseSummarizer (summarizes large tool outputs)\n"
    "  3. TodoListMiddleware (task planning)\n"
    "  4. SummarizationMiddleware (conversation summarization)"
)

# Initialize in-memory checkpointer for conversation memory
logger.info("Initializing MemorySaver checkpointer for conversation persistence...")
memory = MemorySaver()
logger.info("Memory checkpointer initialized successfully")

# ---------------- AGENT 2: INITIALIZE THE DEEP RESEARCH AGENT ----------------
"""
The deep research agent performs comprehensive error analysis and internet research.

This agent:
1. Analyzes error patterns from insights
2. Performs internet research to find solutions
3. Generates comprehensive markdown reports with actionable recommendations
4. Writes results to files for review

It combines error analysis with deep research capabilities in a single unified agent.
"""

# Initialize Tavily client for internet search
logger.info("Initializing Tavily client for internet search...")
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    logger.warning("TAVILY_API_KEY environment variable not set. Internet search will not be available.")
    tavily_client = None
else:
    tavily_client = TavilyClient(api_key=tavily_api_key)
    logger.info("Tavily client initialized successfully")


def _create_internet_search_tool(
    tavily_client_instance: Optional[TavilyClient],
    search_config: Dict[str, Any]
):
    """
    Create internet search tool function configured from config file.

    Args:
        tavily_client_instance: Tavily client instance
        search_config: Configuration dictionary with search parameters

    Returns:
        Configured search function
    """
    if tavily_client_instance is None:
        logger.warning("Tavily client not available, creating mock search function")

        def internet_search(query: str) -> str:
            """Mock internet search when Tavily is not configured."""
            return "Internet search not available. Please configure TAVILY_API_KEY environment variable."

        return internet_search

    # Get search config parameters
    max_results = search_config.get("max_results", 5)
    topic = search_config.get("topic", "general")
    include_raw_content = search_config.get("include_raw_content", False)

    logger.info(f"Configuring internet search with max_results={max_results}, topic={topic}")

    def internet_search(
        query: str,
    ) -> Dict[str, Any]:
        """
        Run a web search using Tavily.

        Args:
            query: Search query string

        Returns:
            Search results dictionary
        """
        logger.info(f"Performing internet search for query: {query}")
        try:
            results = tavily_client_instance.search(
                query,
                max_results=max_results,
                include_raw_content=include_raw_content,
                topic=topic,
            )
            logger.info(f"Search completed, found {len(results.get('results', []))} results")
            return results
        except Exception as e:
            logger.error(f"Error during internet search: {e}", exc_info=True)
            return {"error": str(e), "results": []}

    return internet_search


# Initialize the deep research agent model configuration
deep_research_agent_model_config: Dict = config_data.get("deep_research_agent_model_information", {})
logger.info(f"Loaded deep research agent configuration")

# Get search config
search_config = deep_research_agent_model_config.get("internet_search", {})

# Create the internet search tool
internet_search = _create_internet_search_tool(tavily_client, search_config)

# Initialize LLM for deep research agent
deep_research_llm = ChatBedrockConverse(
    model=deep_research_agent_model_config.get("model_id"),
    temperature=deep_research_agent_model_config.get("inference_parameters", {}).get("temperature", 0.1),
    max_tokens=deep_research_agent_model_config.get("inference_parameters", {}).get("max_tokens", 8192),
    top_p=deep_research_agent_model_config.get("inference_parameters", {}).get("top_p", 0.92),
)
logger.info(f"Initialized deep research agent LLM: {deep_research_llm}")

# Load deep research agent prompt
deep_research_agent_prompt_path: str = deep_research_agent_model_config.get("deep_research_agent_prompt")
deep_research_agent_base_prompt: str = ""
if deep_research_agent_prompt_path:
    try:
        deep_research_agent_base_prompt = load_system_prompt(deep_research_agent_prompt_path)
        logger.info(f"Loaded deep research agent prompt from: {deep_research_agent_prompt_path}")
    except Exception as e:
        logger.warning(f"Could not load deep research agent prompt: {e}")
        deep_research_agent_base_prompt = "You are a deep research agent specializing in analyzing agent errors and finding solutions."

# Create the deep research agent with tools
logger.info("Creating deep research agent with internet search and file tools...")
root_dir: str = None
if root_dir is None:
    root_dir = os.getcwd()
    logger.info(f"Set the root directory: {root_dir}")
    
deep_research_agent = create_deep_agent(
    model=deep_research_llm,
    tools=[internet_search, write_file, read_file],
    system_prompt=deep_research_agent_base_prompt,
    backend=FilesystemBackend(root_dir=root_dir),
)
logger.info("Deep research agent created successfully")

# ---------------- SHARED AGENT STATE ----------------

class UnifiedAgentState(TypedDict):
    """
    Unified state shared by both insights and research agents.

    This state allows:
    1. Insights agent to populate raw_logs and insights
    2. Research agent to access both raw_logs and insights for analysis
    3. Conversation memory through messages field
    """
    user_question: str
    session_id: Optional[str]
    raw_logs: Dict[str, Any]  # Raw logs from LangSmith
    insights: str  # Insights generated by insights agent
    research_results: str  # Research results from research agent
    output_file_path: str  # Path to output report file
    messages: Annotated[List[BaseMessage], add_messages]

@traceable
def get_insights(
    state: UnifiedAgentState
) -> UnifiedAgentState:
    """
    Answer user question with conversation memory.

    This node uses conversation history (existing messages) as context.
    If no history exists and no logs provided, agent can use LangSmith tools.
    """
    print("\n🔍 [INSIGHTS AGENT NODE] Starting analysis...")
    logger.info("Generating insights based on conversation history and context...")

    user_question = state["user_question"]
    raw_logs = state.get("raw_logs", {})
    existing_messages = state.get("messages", [])

    print(f"📊 [INSIGHTS AGENT] Found {len(existing_messages)} messages in conversation history")
    logger.info(f"Found {len(existing_messages)} existing messages in conversation history")

    try:
        # Prepare messages for the agent: use existing conversation history
        messages_to_send = list(existing_messages)  # Copy existing messages (contains all previous insights)

        # Only add raw logs if they exist and are not empty
        if raw_logs and raw_logs != {} and "error" not in raw_logs:
            print("📋 [INSIGHTS AGENT] Adding raw logs to context...")
            analysis_prompt = f"""
## User Question:
{user_question}

## Raw Logs:
{json.dumps(raw_logs, indent=2, default=str)}
"""
        else:
            # No logs provided, just use conversation history and let agent use tools if needed
            print("💬 [INSIGHTS AGENT] Using conversation history only (no new logs provided)...")
            analysis_prompt = user_question

        messages_to_send.append(HumanMessage(content=analysis_prompt))
        logger.info(f"Constructed the insights agent prompt with {len(analysis_prompt)} characters")

        # Count tokens before sending
        try:
            input_token_count = insights_llm.get_num_tokens_from_messages(messages_to_send)
            logger.info(f"Input token count: {input_token_count}")
        except Exception as e:
            logger.warning(f"Could not count input tokens: {e}")
            input_token_count = None

        logger.info(f"Sending {len(messages_to_send)} messages to insights agent (including history)")

        print("💭 [INSIGHTS AGENT] Thinking and generating response...")
        # Invoke the insights agent with full conversation context
        response = insights_agent.invoke({"messages": messages_to_send})
        print("✅ [INSIGHTS AGENT] Response generated successfully")

        # Log token usage from response
        if "messages" in response and len(response["messages"]) > 0:
            last_message = response["messages"][-1]
            if hasattr(last_message, 'usage_metadata') and last_message.usage_metadata:
                usage = last_message.usage_metadata
                logger.info(f"Token usage - Input: {usage.get('input_tokens', 'N/A')}, "
                          f"Output: {usage.get('output_tokens', 'N/A')}, "
                          f"Total: {usage.get('total_tokens', 'N/A')}")
            elif hasattr(last_message, 'response_metadata') and last_message.response_metadata:
                # Alternative location for token usage
                metadata = last_message.response_metadata
                if 'usage' in metadata:
                    usage = metadata['usage']
                    logger.info(f"Token usage from metadata - Input: {usage.get('input_tokens', 'N/A')}, "
                              f"Output: {usage.get('output_tokens', 'N/A')}, "
                              f"Total: {usage.get('total_tokens', 'N/A')}")

        # Extract the insights from the response messages
        if "messages" in response and len(response["messages"]) > 0:
            # Get the last message (agent's response)
            last_message = response["messages"][-1]
            insights_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
            state["insights"] = insights_text

            # Add both the user question and agent insights to state messages
            # The add_messages reducer will append these to existing messages
            new_messages = [
                HumanMessage(content=user_question),
                AIMessage(content=insights_text)
            ]
            state["messages"] = new_messages
        else:
            insights_text = str(response)
            state["insights"] = insights_text
            new_messages = [
                HumanMessage(content=user_question),
                AIMessage(content=insights_text)
            ]
            state["messages"] = new_messages
        logger.info("Successfully generated insights and updated conversation history")
    except Exception as e:
        logger.error(f"Error generating insights: {e}", exc_info=True)
        error_msg = f"Error generating insights: {str(e)}"
        state["insights"] = error_msg
        # Add error to conversation history
        new_messages = [
            HumanMessage(content=user_question),
            AIMessage(content=error_msg)
        ]
        state["messages"] = new_messages
    return state

@traceable
def analyze_error_patterns(
    state: UnifiedAgentState
) -> UnifiedAgentState:
    """
    Analyze error patterns from insights and raw logs, then perform deep research.

    This function takes:
    - Raw logs from LangSmith traces
    - Insights generated by the insights agent

    Then analyzes error patterns, performs internet research to find solutions,
    and generates a comprehensive markdown report.

    Args:
        state: UnifiedAgentState containing raw_logs, insights, and user_question

    Returns:
        Updated state with research_results and output_file_path populated
    """
    print(f"\n🔬 [DEEP RESEARCH NODE] Starting deep error analysis...")
    logger.info("In the DEEP ERROR ANALYSIS NODE - This is the second node...")

    raw_logs = state.get("raw_logs", {})
    insights = state.get("insights", "")
    user_question = state.get("user_question", "")

    print(f"📋 [DEEP RESEARCH] Question: {user_question}")
    print(f"💡 [DEEP RESEARCH] Insights available: {len(insights)} characters")
    print(f"📊 [DEEP RESEARCH] Raw logs available: {'Yes' if raw_logs and raw_logs != {} else 'No'}")

    logger.info(f"Received insights of length: {len(insights)}")
    logger.info(f"Received raw logs: {bool(raw_logs and raw_logs != {})}")

    if not insights:
        logger.warning("No insights provided for analysis")
        print("⚠️ [DEEP RESEARCH] No insights available, skipping analysis")
        state["research_results"] = ""
        return state

    try:
        print("🔍 [DEEP RESEARCH] Preparing analysis prompt...")
        # Get output configuration from config
        output_config = deep_research_agent_model_config.get("output", {})
        output_dir = output_config.get("default_output_dir", "reports")
        file_format = output_config.get("default_file_format", "md")

        # Generate output file path with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = f"{output_dir}/error_analysis_{timestamp}.{file_format}"

        # Create comprehensive analysis prompt with both raw logs and insights
        analysis_prompt = f"""
## User Question:
{user_question}

## Insights from Insights Agent:
{insights}

## Raw Logs from Agent Traces:
{json.dumps(raw_logs, indent=2, default=str)}
"""

        logger.info("Invoking deep research agent for analysis...")
        print("🤖 [DEEP RESEARCH] Invoking deep research agent (this may take a while)...")

        # Invoke the deep research agent
        response = deep_research_agent.invoke({"messages": [HumanMessage(content=analysis_prompt)]})
        print("✅ [DEEP RESEARCH] Analysis complete!")

        # Extract results
        if "messages" in response and len(response["messages"]) > 0:
            last_message = response["messages"][-1]
            research_results = last_message.content if hasattr(last_message, 'content') else str(last_message)
        else:
            research_results = str(response)
        state["research_results"] = research_results
        state["output_file_path"] = output_file_path
        state["messages"] = [
            HumanMessage(content=analysis_prompt),
            AIMessage(content=research_results)
        ]

        logger.info(f"Deep research analysis completed. Report saved to: {output_file_path}")
    except Exception as e:
        logger.error(f"Error during deep research analysis: {e}", exc_info=True)
        error_msg = f"Error during analysis: {str(e)}"
        state["research_results"] = error_msg
        state["output_file_path"] = ""
        # Note: Not updating messages here as the error will be communicated through research_results
    return state


@traceable
def route_to_deep_research(
    state: UnifiedAgentState
) -> str:
    """
    Routing function that determines whether to invoke the deep research agent.

    Uses a small LLM to analyze:
    1. Whether there are logs/traces available
    2. Whether the insights indicate issues to investigate
    3. Whether the user is asking for analysis/debugging help

    Args:
        state: UnifiedAgentState containing user_question, raw_logs, and insights

    Returns:
        "analyze_errors" to route to deep research agent, or END to skip it
    """
    print("\n🚦 [ROUTING] Deciding whether to invoke deep research...")
    user_question = state.get("user_question", "")
    logger.info(f"Going to check if the user question requires analysis: {user_question}")

    # Prepare the routing prompt
    routing_prompt = router_prompt_template.format(
        user_question=user_question
    )

    try:
        # Invoke the routing LLM
        logger.info("Invoking routing LLM to determine if deep research is needed...")
        print("🤔 [ROUTING] Analyzing user question...")
        response = router_llm.invoke([HumanMessage(content=routing_prompt)])
        routing_decision = response.content.strip().upper()
        logger.info(f"Routing LLM decision: {routing_decision}")

        # Parse the decision
        if "ROUTE_TO_RESEARCH" in routing_decision:
            print("➡️  [ROUTING] Decision: Routing to deep research agent")
            logger.info("Routing to deep research agent")
            return "analyze_errors"
        else:
            print("✋ [ROUTING] Decision: Skipping deep research, ending workflow")
            logger.info("Skipping deep research agent, ending workflow")
            return END
    except Exception as e:
        logger.error(f"Error during routing decision: {e}", exc_info=True)
        print(f"❌ [ROUTING] Error in routing: {e}, defaulting to END")
        # Default to ending if there's an error
        logger.info("Error in routing, defaulting to END")
        return END


def _build_graph() -> StateGraph:
    """Build the unified multi-agent graph with memory checkpointer."""
    logger.info("Building unified multi-agent graph...")

    # Create graph with unified state
    workflow = StateGraph(UnifiedAgentState)

    # Add nodes
    workflow.add_node("get_insights", get_insights)
    workflow.add_node("analyze_errors", analyze_error_patterns)

    # Define edges
    workflow.add_edge(START, "get_insights")

    # Conditional routing: Use LLM to decide if deep research is needed
    workflow.add_conditional_edges(
        "get_insights",
        route_to_deep_research,
        {
            "analyze_errors": "analyze_errors",
            END: END
        }
    )

    workflow.add_edge('analyze_errors', END)
    logger.info("Graph built successfully with conditional routing and memory checkpointer")
    return workflow.compile(checkpointer=memory)


# Build the graph
app = _build_graph()

def run_agent(
    user_question: str,
    session_id: Optional[str] = None,
    thread_id: str = "default"
) -> Dict[str, Any]:
    """
    Run the log curator agent with memory support.

    Args:
        user_question: The user's question about agent traces
        session_id: LangSmith session/project ID to analyze
        thread_id: Thread ID for conversation memory (default: "default")

    Returns:
        Dictionary containing the answer and curated logs
    """
    logger.info(f"Running log curator for question: {user_question}")
    logger.info(f"Using thread_id: {thread_id} for conversation memory")

    initial_state: UnifiedAgentState = {
        "user_question": user_question,
        "session_id": session_id,
        "raw_logs": {},
        "insights": "",
        "research_results": "",
        "output_file_path": "",
        "messages": [],
    }

    # Configure thread for memory persistence
    config = {"configurable": {"thread_id": thread_id},
              "recursion_limit": 50}

    result = app.invoke(initial_state, config=config)

    return {
        "question": result["user_question"],
        "insights": result["insights"],
        "research_results": result["research_results"],
        "output_file_path": result["output_file_path"],
    }


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Self-Healing Agent - Log Curator: Analyze agent execution traces from LangSmith",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Interactive mode (default)
    python agent.py --session-id "my-session-123"

    # Interactive mode with environment variable
    export LANGSMITH_SESSION_ID="my-session-123"
    python agent.py

    # Enable debug logging
    python agent.py --session-id "abc123" --debug
"""
    )

    parser.add_argument(
        "--session-id",
        type=str,
        help="LangSmith session/project ID to analyze. Can also be set via LANGSMITH_SESSION_ID env var"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    return parser.parse_args()


def _get_session_id(
    cli_value: Optional[str] = None
) -> Optional[str]:
    """
    Get session ID from CLI argument or environment variable.

    Args:
        cli_value: Session ID from command line argument

    Returns:
        Session ID or None if not provided
    """
    if cli_value:
        return cli_value

    env_value = os.getenv("LANGSMITH_SESSION_ID")
    if env_value:
        logger.info("Using session ID from LANGSMITH_SESSION_ID environment variable")
        return env_value

    return None


def _print_welcome_message(
    session_id: Optional[str],
    thread_id: str
) -> None:
    """Print welcome message for interactive mode."""
    print("\n" + "="*80)
    print("Self-Healing Agent - Unified Multi-Agent Workflow (Interactive Mode)")
    print("="*80)
    print("Workflow: Insights Agent -> Research Agent")
    print("  1. Insights Agent: Analyzes LangSmith traces and generates insights")
    print("  2. Research Agent: Performs error analysis and internet research")
    if session_id:
        print(f"\nSession ID: {session_id}")
    else:
        print("\nSession ID: Not provided (analysis may be limited)")
    print(f"Thread ID: {thread_id} (conversation memory enabled)")
    print("\nType your questions about agent execution traces.")
    print("Commands: 'quit', 'exit', or 'done' to exit")
    print("="*80 + "\n")


def _run_interactive_session(
    session_id: Optional[str]
) -> None:
    """
    Run interactive chatbot session with memory.

    Args:
        session_id: LangSmith session/project ID to analyze
    """
    # Generate unique thread_id for this conversation session
    thread_id = str(uuid.uuid4())
    logger.info(f"Generated thread_id: {thread_id} for interactive session")

    _print_welcome_message(session_id, thread_id)

    conversation_history: List[Dict[str, str]] = []

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'done', 'q']:
                print("\nThank you for using the Log Curator. Goodbye!")
                break

            # Skip empty input
            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Type any question about agent traces")
                print("  - 'history' - Show conversation history")
                print("  - 'clear' - Clear conversation history")
                print("  - 'quit', 'exit', 'done' - Exit the program")
                continue

            if user_input.lower() == 'history':
                if not conversation_history:
                    print("\nNo conversation history yet.")
                else:
                    print("\n" + "="*80)
                    print("CONVERSATION HISTORY:")
                    print("="*80)
                    for idx, entry in enumerate(conversation_history, 1):
                        print(f"\n[{idx}] Q: {entry['question']}")
                        insights_preview = entry.get('insights', '')[:200]
                        print(f"    Insights: {insights_preview}..." if len(entry.get('insights', '')) > 200 else f"    Insights: {insights_preview}")
                        if entry.get('output_file_path'):
                            print(f"    Report: {entry['output_file_path']}")
                    print("="*80)
                continue

            if user_input.lower() == 'clear':
                conversation_history.clear()
                print("\nConversation history cleared.")
                continue

            # Process the question
            logger.info(f"Processing question: {user_input}")
            print("\nAgent: Analyzing...")

            try:
                result = run_agent(
                    user_question=user_input,
                    session_id=session_id,
                    thread_id=thread_id
                )

                # Store in conversation history
                conversation_history.append({
                    "question": result["question"],
                    "insights": result["insights"],
                    "research_results": result["research_results"],
                    "output_file_path": result["output_file_path"]
                })

                # Display results
                print("\n" + "="*80)
                # Show insights if available
                if result.get('insights'):
                    print(result['insights'])

                # Show research results if available
                if result.get('research_results'):
                    if result.get('insights'):  # Add separator if both exist
                        print("\n" + "-"*80 + "\n")
                    print(result['research_results'])

                # Show file path if report was generated
                if result.get('output_file_path'):
                    print("\n" + "-"*80)
                    print(f"📄 Report saved to: {result['output_file_path']}")
                print("="*80)

            except Exception as e:
                logger.error(f"Error processing question: {e}", exc_info=True)
                print(f"\nAgent: Sorry, I encountered an error: {str(e)}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit or continue with another question.")
            continue
        except EOFError:
            print("\n\nGoodbye!")
            break


def main() -> None:
    """Main function to run the unified multi-agent workflow."""
    args = _parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Get session ID from CLI or environment
    session_id = _get_session_id(args.session_id)

    if not session_id:
        logger.warning("No session ID provided. Analysis may be limited without a specific session.")

    try:
        _run_interactive_session(session_id)
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 
