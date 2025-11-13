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
import asyncio
import argparse
from utils import *
from constants import *
from typing import Annotated
from dotenv import load_dotenv
from langsmith import traceable
# Import Tavily for internet search
from tavily import TavilyClient
from typing_extensions import TypedDict
from deepagents import create_deep_agent
from typing import Any, Dict, List, Optional
from langchain_aws import ChatBedrockConverse
from langgraph.graph.message import add_messages
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from deepagents.backends import StateBackend, FilesystemBackend
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
    SystemMessage,
)

# Import tools
from agent_tools.file_tools import *

# Import insights agent factory for dynamic platform-based agent creation
from insights.insights_agents import InsightsAgentFactory

# Load environment variables from .env file
load_dotenv()

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

# ---------------- INSIGHTS AGENT FACTORY INITIALIZATION ----------------
"""
Initialize the InsightsAgentFactory that will create platform-specific agents
dynamically based on the user's question.

The factory handles:
1. LLM initialization
2. Middleware stack creation
3. MCP tool connection
4. Platform-specific prompt loading
"""
logger.info("Initializing InsightsAgentFactory for dynamic agent creation...")
insights_agent_factory = InsightsAgentFactory(config_file_path=CONFIG_FILE_FPATH)
logger.info("✅ InsightsAgentFactory initialized successfully")

# ---------------- AGENTCORE MEMORY INITIALIZATION ----------------
"""
Initialize AgentCore Memory for long-term conversation storage and retrieval.
This enables semantic search and reduces context bleeding by storing full history
and retrieving only relevant context.
"""

# Load AgentCore Memory configuration
agentcore_memory_config = config_data.get('agentcore_memory', {})
agentcore_memory_enabled = agentcore_memory_config.get('enabled', False)

memory = None
memory_store = None

if agentcore_memory_enabled:
    logger.info("AgentCore Memory is enabled. Initializing...")

    try:
        from agent_memory import initialize_agentcore_memory, get_memory_store

        # Get configuration
        memory_name = agentcore_memory_config.get('memory_name', 'SelfHealingAgentMemory')
        region_name = agentcore_memory_config.get('region_name', 'us-west-2')
        memory_model_id = agentcore_memory_config.get('memory_model_id', 'us.anthropic.claude-3-5-sonnet-20240620-v1:0')
        description = agentcore_memory_config.get('description', 'Self-healing agent conversation memory')

        # Get IAM role from environment or config
        memory_execution_role_arn = agentcore_memory_config.get('memory_execution_role_arn')

        if not memory_execution_role_arn:
            logger.error(
                "AgentCore Memory role ARN not configured. "
                "Set AGENTCORE_MEMORY_ROLE_ARN environment variable or agentcore_memory.memory_execution_role_arn in config.yaml"
            )
            raise ValueError("Missing AgentCore Memory role ARN")

        # Initialize or get existing memory
        logger.info(f"Initializing AgentCore Memory: {memory_name}")
        memory_info = initialize_agentcore_memory(
            memory_name=memory_name,
            region_name=region_name,
            memory_execution_role_arn=memory_execution_role_arn,
            model_id=memory_model_id,
            description=description)
        memory_id = memory_info['id']
        logger.info(f"AgentCore Memory initialized with ID: {memory_id}")
        # Get memory store for LangGraph integration
        memory_store = get_memory_store(
            memory_id=memory_id,
            region_name=region_name,
        )
        logger.info("AgentCore Memory Store created successfully")

        # Use AgentCore Memory checkpointer
        from langgraph_checkpoint_aws import AgentCoreMemorySaver
        memory = AgentCoreMemorySaver(
            memory_id=memory_id,
            region_name=region_name,
        )
        logger.info("Using AgentCore Memory Saver as checkpointer")

    except Exception as e:
        logger.error(f"Error initializing AgentCore Memory: {e}", exc_info=True)
        logger.warning("Falling back to in-memory checkpointer")
        agentcore_memory_enabled = False
        memory = MemorySaver()
else:
    logger.info("AgentCore Memory is disabled. Using in-memory checkpointer...")
    memory = MemorySaver()

logger.info("Memory checkpointer initialized successfully")

# ---------------- MEMORY HOOKS ----------------
"""
Memory hooks for search (pre-model) and storage (post-model).
These are direct functions called in the get_insights node for minimal code.
"""

def _memory_search_hook(
    messages: List[BaseMessage],
    user_id: str,
) -> List[BaseMessage]:
    """
    Pre-model hook: Search AgentCore Memory for relevant context.

    Args:
        messages: Current conversation messages
        user_id: User/thread identifier

    Returns:
        Enriched messages with relevant context from memory
    """
    if not agentcore_memory_enabled or not memory_store:
        logger.debug("Memory search disabled")
        return messages
    if not messages:
        logger.debug("No messages to enrich")
        return messages
    try:
        # Get context retrieval config
        context_config = agentcore_memory_config.get('context_retrieval', {})
        top_k_relevant = context_config.get('top_k_relevant', 3)
        keep_recent = context_config.get('keep_recent_messages', 3)
        # Extract the latest user question for semantic search
        user_question = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_question = msg.content
                break

        if not user_question:
            logger.debug("No user question found for memory search")
            return messages

        logger.info(f"🔍 Searching memory for user: {user_id}")
        logger.debug(f"Memory search query: {user_question}")
        logger.debug(f"Namespace: ({user_id}/{ERRORS_AND_INSIGHTS_NAMESPACE})")
        logger.debug(f"Top K: {top_k_relevant}")

        # Search memory using the correct namespace format
        # AgentCore Memory expects: (actor_id, namespace_suffix)
        # BaseStore.search() expects namespace_prefix as a positional argument
        search_results = memory_store.search(
            (user_id, ERRORS_AND_INSIGHTS_NAMESPACE),  # Positional argument
            query=user_question,
            limit=top_k_relevant,
        )
        print(f"SEARCH RESULTS RETURNED BASED ON THE USER AND QUESTION: {search_results}")
        # Extract relevant context
        relevant_context = []
        if search_results:
            logger.info(f"Found {len(search_results)} relevant memories")
            for result in search_results:
                if hasattr(result, 'value') and result.value:
                    relevant_context.append(result.value)
        else:
            logger.info("No relevant memories found")

        # Keep only recent messages
        recent_messages = (
            messages[-keep_recent:]
            if len(messages) > keep_recent
            else messages
        )
        # Build enriched messages
        enriched_messages = []
        # Add relevant context as system message
        if relevant_context:
            context_summary = "## Relevant Context from Previous Conversations:\n\n"
            for i, ctx in enumerate(relevant_context, 1):
                ctx_str = (
                    json.dumps(ctx, indent=2, default=str)
                    if isinstance(ctx, dict)
                    else str(ctx)
                )
                context_summary += f"### Context {i}:\n{ctx_str}\n\n"
            enriched_messages.append(SystemMessage(content=context_summary))
            print(f"🧠 [MEMORY SEARCH] Retrieved {len(relevant_context)} relevant items")
        # Add recent messages
        enriched_messages.extend(recent_messages)
        logger.info(f"Enriched: {len(messages)} → {len(enriched_messages)} messages")
        return enriched_messages
    except Exception as e:
        logger.error(f"Memory search error: {e}", exc_info=True)
        print(f"⚠️  [MEMORY SEARCH] Error: {e}")
        return messages

def _memory_store_hook(
    user_id: str,
    session_id: Optional[str],
    user_question: str,
    ai_response: str,
    raw_logs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Post-model hook: Store conversation in AgentCore Memory.

    Args:
        user_id: User/thread identifier
        session_id: Session identifier
        user_question: User's question
        ai_response: AI's response
        raw_logs: Optional raw logs to store
    """
    if not agentcore_memory_enabled or not memory_store:
        logger.debug("Memory storage disabled")
        return

    if not user_question or not ai_response:
        logger.debug("Missing question or response, skipping storage")
        return

    try:
        from datetime import datetime, timezone

        logger.info(f"💾 Storing conversation for user {user_id}...")
        logger.debug(f"Namespace: ({user_id}, {ERRORS_AND_INSIGHTS_NAMESPACE})")
        logger.debug(f"Question length: {len(user_question)} chars")
        logger.debug(f"Response length: {len(ai_response)} chars")

        # Prepare namespace: (actor_id, namespace_suffix)
        namespace = (user_id, ERRORS_AND_INSIGHTS_NAMESPACE)

        # Add metadata
        timestamp = datetime.now(timezone.utc).isoformat()
        session_info = f"\n\n[Session: {session_id}, Timestamp: {timestamp}]"

        # Store user question
        question_key = f"q_{uuid.uuid4()}"
        logger.debug(f"Storing question with key: {question_key}")
        memory_store.put(
            namespace,  # Positional argument
            question_key,
            {"message": HumanMessage(content=f"{user_question}{session_info}")},
        )
        logger.debug(f"Question stored successfully")

        # Store AI response
        response_key = f"a_{uuid.uuid4()}"
        logger.debug(f"Storing response with key: {response_key}")
        memory_store.put(
            namespace,  # Positional argument
            response_key,
            {"message": AIMessage(content=f"{ai_response}{session_info}")},
        )
        logger.debug(f"Response stored successfully")

        # Store raw logs if provided
        if raw_logs and raw_logs != {} and "error" not in raw_logs:
            logs_summary = (
                f"Raw Logs Context:\n"
                f"Question: {user_question}\n"
                f"Insights: {ai_response[:500]}...\n"
                f"Logs: {json.dumps(raw_logs, indent=2, default=str)[:2000]}...\n"
                f"{session_info}"
            )
            logs_key = f"log_{uuid.uuid4()}"
            logger.debug(f"Storing logs with key: {logs_key}")
            logger.debug(f"Logs summary length: {len(logs_summary)} chars")
            memory_store.put(
                namespace,  # Positional argument
                logs_key,
                {"message": SystemMessage(content=logs_summary)},
            )
            logger.info("Stored raw logs in memory")
            logger.debug(f"Logs stored successfully")

        print("✅ [MEMORY STORAGE] Conversation stored successfully")
        logger.info("Successfully stored conversation in memory")

    except Exception as e:
        logger.error(f"Memory storage error: {e}", exc_info=True)
        print(f"⚠️  [MEMORY STORAGE] Error: {e}")


# Note: Insights agent is now created dynamically in the get_insights node
# based on the platform determined from the user's question.
# The InsightsAgentFactory handles all middleware, LLM, and tool initialization.

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
    4. User identification for AgentCore Memory
    5. Platform persistence across conversation turns
    """
    user_question: str
    session_id: Optional[str]
    user_id: str  # User/thread identifier for AgentCore Memory
    platform: Optional[str]  # Selected observability platform (persisted across session)
    raw_logs: Dict[str, Any]  # Raw logs from LangSmith
    insights: str  # Insights generated by insights agent
    research_results: str  # Research results from research agent
    output_file_path: str  # Path to output report file
    messages: Annotated[List[BaseMessage], add_messages]


def select_platform(
    state: UnifiedAgentState
) -> UnifiedAgentState:
    """
    Prompt user to select observability platform at session start.

    This node:
    1. Checks if platform is already selected in state
    2. If not, prompts user to choose between LangSmith and Langfuse
    3. Stores the selection in state for the entire session
    """
    existing_platform = state.get("platform")

    # If platform already selected, skip prompt
    if existing_platform:
        print(f"✅ [PLATFORM] Using previously selected platform: {existing_platform}")
        logger.info(f"Platform already selected: {existing_platform}")
        return state
    # Prompt user for platform selection
    print("\n🔍 [PLATFORM SELECTION] Please select your observability platform:")
    print("   1. LangSmith")
    print("   2. Langfuse")
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()

        if choice == "1":
            selected_platform = "langsmith"
            print("✅ Selected: LangSmith")
            logger.info("User selected LangSmith platform")
            break
        elif choice == "2":
            selected_platform = "langfuse"
            print("✅ Selected: Langfuse")
            logger.info("User selected Langfuse platform")
            break
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")

    # Store platform in state that will be used across the agent execution process
    state["platform"] = selected_platform
    return state


@traceable
async def get_insights(
    state: UnifiedAgentState
) -> UnifiedAgentState:
    """
    Answer user question with conversation memory and platform-specific agent.
    This node:
    1. Determines which observability platform to use (LangSmith or Langfuse)
    2. Dynamically creates the appropriate insights agent with platform-specific tools
    3. Uses conversation history (existing messages) as context
    4. Memory search (pre-model) and storage (post-model) hooks are called directly
    """
    print("\n🔍 [INSIGHTS AGENT NODE] Starting analysis...")
    logger.info("Generating insights based on conversation history and context...")

    user_question = state["user_question"]
    raw_logs = state.get("raw_logs", {})
    existing_messages = state.get("messages", [])
    user_id = state.get("user_id", "default_user")
    session_id = state.get("session_id")
    stored_platform = state.get("platform")  # Get platform from state

    print(f"📊 [INSIGHTS AGENT] Found {len(existing_messages)} messages in conversation history")
    logger.info(f"Found {len(existing_messages)} existing messages in conversation history")

    try:
        # STEP 1: Get platform from state (selected at session start)
        platform = stored_platform
        if not platform:
            # This shouldn't happen with the new flow, but fallback to default if it does
            platform = DEFAULT_PLATFORM
            logger.warning(f"No platform found in state, using default: {platform}")
            print(f"⚠️  [PLATFORM] No platform selected, using default: {platform}")
        else:
            logger.info(f"Using platform from session: {platform}")
            print(f"✅ [PLATFORM] Using session platform: {platform}")

        # STEP 2: Create platform-specific insights agent dynamically
        print(f"🏗️  [AGENT FACTORY] Creating {platform} insights agent...")
        logger.info(f"Creating insights agent for platform: {platform}")

        from insights.insights_agents import ObservabilityPlatform

        # Convert string to enum
        if platform == PLATFORM_LANGSMITH:
            platform_enum = ObservabilityPlatform.LANGSMITH
        elif platform == PLATFORM_LANGFUSE:
            platform_enum = ObservabilityPlatform.LANGFUSE
        else:
            logger.warning(f"Unknown platform {platform}, defaulting to {DEFAULT_PLATFORM}")
            platform_enum = ObservabilityPlatform.LANGSMITH

        # Create the agent using the factory (async operation)
        insights_agent = await insights_agent_factory.create_insights_agent(platform=platform_enum)
        logger.info(f"✅ Created {platform} insights agent successfully")
        print(f"✅ [AGENT FACTORY] {platform.capitalize()} insights agent ready")
        # PRE-MODEL HOOK: Search memory for relevant context
        messages_to_send = _memory_search_hook(
            messages=list(existing_messages),
            user_id=user_id,
        )
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
            print("💬 [INSIGHTS AGENT] Using conversation history only (no new logs provided)...")
            analysis_prompt = user_question

        messages_to_send.append(HumanMessage(content=analysis_prompt))
        logger.info(f"Constructed the insights agent prompt with {len(analysis_prompt)} characters")
        logger.info(f"Sending {len(messages_to_send)} messages to insights agent (including history)")

        print("💭 [INSIGHTS AGENT] Thinking and generating response...")
        print("\n" + "="*80)
        print("AGENT RESPONSE (streaming):")
        print("="*80 + "\n")

        # Invoke the insights agent with full conversation context (async)
        response = insights_agent.astream(
            {"messages": messages_to_send},
            stream_mode=["updates", "messages"]
        )
        
        # Collect all messages from the agent execution
        all_response_messages = []
        final_ai_message = None
        current_message_content = ""

        # Iterate through the async stream
        async for stream_mode, chunk in response:
            logger.debug(f"Stream mode: {stream_mode}, Chunk type: {type(chunk)}")
            
            # Handle updates mode (agent steps) - this contains the complete state after each step
            if stream_mode == "updates":
                logger.debug(f"Update chunk: {chunk}")
                # Extract messages from the update
                for node_name, node_state in chunk.items():
                    if isinstance(node_state, dict) and 'messages' in node_state:
                        messages_list = node_state['messages']
                        logger.debug(f"Node '{node_name}' has {len(messages_list)} messages")
                        
                        # Store all messages from this node update
                        for msg in messages_list:
                            # Avoid duplicates by checking if message is already in our list
                            if msg not in all_response_messages:
                                all_response_messages.append(msg)
                                
                                # Log the message type for debugging
                                msg_type = type(msg).__name__
                                msg_preview = str(msg.content)[:100] if hasattr(msg, 'content') else str(msg)[:100]
                                logger.debug(f"  Message type: {msg_type}, Preview: {msg_preview}")
                                
                                # Print tool messages as they arrive
                                if not isinstance(msg, (AIMessage, HumanMessage)):
                                    print(f"\n🔧 [{msg_type}]")
                                    if hasattr(msg, 'content'):
                                        content = msg.content if isinstance(msg.content, str) else str(msg.content)
                                        # Truncate long tool outputs for readability
                                        if len(content) > 500:
                                            print(f"{content[:500]}... [truncated, {len(content)} total chars]")
                                        else:
                                            print(content)
                                    print()
                                
                                # Keep track of the final AI message
                                if isinstance(msg, AIMessage):
                                    final_ai_message = msg
            
            # Handle messages mode (LLM tokens and messages) - PRINT TOKENS IN REAL-TIME
            elif stream_mode == "messages":
                # Print tokens as they stream for real-time display
                if hasattr(chunk, 'content') and chunk.content:
                    if isinstance(chunk.content, str):
                        # Simple string content - print directly
                        print(chunk.content, end="", flush=True)
                        current_message_content += chunk.content
                    elif isinstance(chunk.content, list):
                        # Handle structured content (list of dicts with type/text)
                        for item in chunk.content:
                            if isinstance(item, dict):
                                # Extract text from different content types
                                if item.get('type') == 'text':
                                    text = item.get('text', '')
                                    print(text, end="", flush=True)
                                    current_message_content += text
                                elif item.get('type') == 'tool_use':
                                    # Tool use indication - show that a tool is being called
                                    tool_name = item.get('name', 'unknown')
                                    print(f"\n\n🔧 [Calling tool: {tool_name}]\n", flush=True)
                            else:
                                # Fallback for non-dict items
                                text = str(item)
                                print(text, end="", flush=True)
                                current_message_content += text
                
                # Log token usage if available
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    usage = chunk.usage_metadata
                    logger.info(f"Token usage - Input: {usage.get('input_tokens', 'N/A')}, "
                            f"Output: {usage.get('output_tokens', 'N/A')}, "
                            f"Total: {usage.get('total_tokens', 'N/A')}")
                elif hasattr(chunk, 'response_metadata') and chunk.response_metadata:
                    metadata = chunk.response_metadata
                    if 'usage' in metadata:
                        usage = metadata['usage']
                        logger.info(f"Token usage from metadata - Input: {usage.get('input_tokens', 'N/A')}, "
                                f"Output: {usage.get('output_tokens', 'N/A')}, "
                                f"Total: {usage.get('total_tokens', 'N/A')}")
        
        # Print newline after streaming completes
        print("\n")
        print("="*80)

        # Extract insights from FINAL AI message only (skip intermediate responses)
        # Also collect tool messages to include in the output
        insights_text = ""
        tool_messages = []

        # Get the last AI message as the final insights
        for msg in reversed(all_response_messages):
            if isinstance(msg, AIMessage) and not insights_text:
                # This is the final AI response
                if hasattr(msg, 'content') and msg.content:
                    if isinstance(msg.content, str):
                        insights_text = msg.content
                    elif isinstance(msg.content, list):
                        # Extract text from list of dicts
                        text_parts = []
                        for item in msg.content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                text_parts.append(item.get('text', ''))
                        insights_text = ''.join(text_parts)
                break

        # Collect tool messages (already pruned by PruneToolCallMiddleware)
        for msg in all_response_messages:
            if hasattr(msg, '__class__') and msg.__class__.__name__ == 'ToolMessage':
                tool_messages.append(msg)
        
        # If no insights were built from parts, fall back to final AI message
        if not insights_text and final_ai_message:
            if isinstance(final_ai_message.content, str):
                insights_text = final_ai_message.content
            elif isinstance(final_ai_message.content, list):
                # Extract text from list format
                text_parts = []
                for item in final_ai_message.content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                insights_text = ''.join(text_parts)
            logger.info(f"Using final AI message as insights: {len(insights_text)} characters")
        
        # After streaming completes, update state
        if insights_text:
            print(f"\n✅ [INSIGHTS AGENT] Completed - {len(all_response_messages)} messages total")
            print(f"   - Tool messages: {sum(1 for m in all_response_messages if not isinstance(m, (AIMessage, HumanMessage)))}")
            print(f"   - AI messages: {sum(1 for m in all_response_messages if isinstance(m, AIMessage))}")

            state["insights"] = insights_text

            # Add both the user question and agent insights to state messages
            new_messages = [
                HumanMessage(content=user_question),
                AIMessage(content=insights_text)
            ]
            state["messages"] = new_messages
            logger.info("Successfully generated insights and updated conversation history")

            # POST-MODEL HOOK: Store conversation in memory
            _memory_store_hook(
                user_id=user_id,
                session_id=session_id,
                user_question=user_question,
                ai_response=insights_text,
                raw_logs=raw_logs,
            )
        else:
            logger.error("No insights text found in either messages or updates stream")
            logger.error(f"All response messages: {all_response_messages}")
            raise Exception("No insights generated from stream")
            
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
    workflow.add_node("select_platform", select_platform)
    workflow.add_node("get_insights", get_insights)
    workflow.add_node("analyze_errors", analyze_error_patterns)

    # Define edges - platform selection happens first
    workflow.add_edge(START, "select_platform")
    workflow.add_edge("select_platform", "get_insights")

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

async def run_agent(
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
        "user_id": thread_id,  # Use thread_id as user_id for AgentCore Memory
        "raw_logs": {},
        "insights": "",
        "research_results": "",
        "output_file_path": "",
        "messages": [],
    }

    # Configure thread for memory persistence
    # AgentCore Memory requires both thread_id and actor_id
    config = {
        "configurable": {
            "thread_id": thread_id,
            "actor_id": thread_id  # Use thread_id as actor_id for AgentCore Memory
        },
        "recursion_limit": 50
    }

    result = await app.ainvoke(initial_state, config=config)

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
                # Run the async agent using asyncio
                result = asyncio.run(
                    run_agent(
                        user_question=user_input,
                        session_id=session_id,
                        thread_id=thread_id
                    )
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
