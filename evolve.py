"""
Self-Healing Agent - Evolve

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
import difflib
import asyncio
import argparse
from utils import *
from constants import *
from pathlib import Path
from typing import Annotated
from dotenv import load_dotenv
from langsmith import traceable
from typing_extensions import TypedDict
from langgraph.errors import GraphInterrupt
from typing import Any, Dict, List, Optional
from langchain_aws import ChatBedrockConverse
from langgraph.graph.message import add_messages
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
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

# Import the evolution agent factory that is used for offline evaluation and evolution
from evolution.prompt_evolution import PromptOptimizationAgentFactory

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

# ---------------- PROMPT EVOLUTION SYSTEM INITIALIZATION ----------------
"""
The prompt evolution system analyzes insights from observability platforms
and optimizes system prompts to improve agent performance.

This system:
1. Analyzes agent performance patterns from insights
2. Identifies areas for prompt improvement
3. Optimizes system prompts based on observed behaviors
4. Has file system access to read and modify prompts
"""
logger.info("Initializing Prompt Evolution System...")
prompt_evolution_system = PromptOptimizationAgentFactory(config_file_path=EVOLUTION_ENGINE_CONFIG_FILE)
logger.info("✅ Prompt Evolution System initialized successfully")

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
    # this is the agent repository of the agent code to optimize
    agent_repo: str
    # this is the special marker to check whether the step is to skip to evolution if the follow up user
    # question is about evolving the agent code
    spec_insights_marker: str

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

def select_agent_repository(
    state: UnifiedAgentState
) -> UnifiedAgentState:
    """
    This is the node that is to get the agent repository location 
    where the file read and writes will happen.
    """
    import os
    import re
    import subprocess
    import tempfile
    import shutil
    
    agent_repo = state.get("agent_repo")
    # If the agent repo is already selected, skip this step
    if agent_repo:
        print(f"✅ [AGENT REPO] Using previously selected agent repo: {agent_repo}")
        return state
    
    # Prompt user for platform selection
    print("\n🔍 [AGENT REPOSITORY] Please select your agent code repo:")
    print("   1. GitHub URL (e.g., https://github.com/user/repo.git)")
    print("   2. Local path (e.g., /path/to/repo)")
    
    while True:
        agent_path = input("\nEnter your agent destination path: ").strip()
        if not agent_path:
            print("❌ Invalid choice. Provide a valid URL or repo path")
            continue
        # Check if it's a GitHub URL
        github_pattern = r'^https?://github\.com/[\w-]+/[\w.-]+(?:\.git)?$'
        if re.match(github_pattern, agent_path):
            print(f"🔗 Detected GitHub URL: {agent_path}")
            try:
                # Create a temporary directory for cloning
                temp_dir = tempfile.mkdtemp(prefix="agent_repo_")
                print(f"📥 Cloning repository to: {temp_dir}")
                # Clone the repository
                result = subprocess.run(
                    ["git", "clone", agent_path, temp_dir],
                    capture_output=True,
                    text=True,
                    timeout=60  # 60 second timeout
                )
                if result.returncode == 0:
                    print(f"✅ Successfully cloned repository")
                    # Store the local cloned path in state
                    state["agent_repo"] = temp_dir
                    state["agent_path"] = agent_path  # Store original URL for reference
                    state["repo_type"] = "github"
                    return state
                else:
                    print(f"❌ Failed to clone repository: {result.stderr}")
                    # Clean up temp directory if clone failed
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
            except subprocess.TimeoutExpired:
                print("❌ Repository clone timed out. Please check the URL and try again.")
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except FileNotFoundError:
                print("❌ Git is not installed. Please install git and try again.")
            except Exception as e:
                print(f"❌ Error cloning repository: {e}")
                if 'temp_dir' in locals() and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        
        # Check if it's a local path
        else:
            # Expand user home directory if ~ is used
            expanded_path = os.path.expanduser(agent_path)
            absolute_path = os.path.abspath(expanded_path)
            
            if os.path.exists(absolute_path):
                if os.path.isdir(absolute_path):
                    print(f"✅ Valid local directory found: {absolute_path}")
                    # Check if it's a git repository
                    if os.path.exists(os.path.join(absolute_path, '.git')):
                        print("   📁 Detected as a git repository")
                    
                    # Store the validated local path in state
                    state["agent_repo"] = absolute_path
                    state["agent_path"] = agent_path  # Store original path for reference
                    state["repo_type"] = "local"
                    return state
                else:
                    print(f"❌ Path exists but is not a directory: {absolute_path}")
            else:
                print(f"❌ Path does not exist: {absolute_path}")
                # Ask if they want to create it
                create = input("   Would you like to create this directory? (y/n): ").strip().lower()
                if create == 'y':
                    try:
                        os.makedirs(absolute_path, exist_ok=True)
                        print(f"✅ Created directory: {absolute_path}")
                        state["agent_repo"] = absolute_path
                        state["agent_path"] = agent_path
                        state["repo_type"] = "local"
                        return state
                    except Exception as e:
                        print(f"❌ Failed to create directory: {e}")

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
    stored_platform = state.get("platform")

    print(f"📊 [INSIGHTS AGENT] Found {len(existing_messages)} messages in conversation history")
    logger.info(f"Found {len(existing_messages)} existing messages in conversation history")

    try:
        # STEP 1: Get platform from state (selected at session start)
        platform = stored_platform
        if not platform:
            platform = DEFAULT_PLATFORM
            logger.warning(f"No platform found in state, using default: {platform}")
            print(f"⚠️  [PLATFORM] No platform selected, using default: {platform}")
        else:
            logger.info(f"Using platform from session: {platform}")
            print(f"✅ [PLATFORM] Using session platform: {platform}")
        
        # STEP 2: Detect user intent based on their question
        user_intent_router_prompt: str = load_system_prompt(
            config_data['routing_configuration']['router_prompts'].get('user_intent_routing')
        )
        formatted_user_intent_router_prompt = user_intent_router_prompt.format(
            user_question=user_question,
            conversation_history=existing_messages[:3]
        )
        
        logger.info(f"Invoking the user intent router with the following prompt: {formatted_user_intent_router_prompt}")
        user_intent_response = router_llm.invoke(
            [HumanMessage(content=formatted_user_intent_router_prompt)]
        )
        user_intent = user_intent_response.content.strip().upper()
        logger.info(f"Got the user intent: {user_intent}")
        
        # STEP 3: Check user intent and decide whether to skip insights
        if TO_EVOLUTION_HINT in user_intent:
            # User wants evolution directly - skip insights generation
            logger.info("User intent detected: Evolution requested. Skipping insights generation.")
            print("🧬 [INTENT ROUTING] Evolution requested - skipping insights analysis")
            # Set minimal state to indicate we're going straight to evolution
            state["insights"] = "User requested prompt evolution directly."
            state["messages"] = [
                HumanMessage(content=user_question),
                AIMessage(content="Routing to evolution engine...")
            ]
            state["spec_insights_marker"] = TO_EVOLUTION_HINT
            return state
        else:
            state["spec_insights_marker"] = CONTINUE_WITH_INSIGHTS_HINT
        # If not going to evolution, continue with insights generation
        logger.info("User intent: Continuing with insights generation")
        print("💡 [INTENT ROUTING] Continuing with insights analysis")
        
        # STEP 3: Create platform-specific insights agent dynamically
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

async def evolution_engine(state: UnifiedAgentState) -> UnifiedAgentState:
    """
    Enhanced evolution engine that displays patches before applying changes.
    """
    print("\n🧬 [EVOLUTION NODE] Starting prompt evolution analysis...")
    logger.info("In the EVOLUTION NODE - Analyzing insights for prompt optimization...")

    # Use the correct key from your state
    question = state.get("user_question", "")
    insights = state.get("insights", "")
    observability_platform = state.get("observability_platform", "")
    agent_repo = state.get("agent_repo", "")

    print(f"📋 [EVOLUTION] Question: {question}")
    print(f"💡 [EVOLUTION] Insights available: {len(insights)} characters")
    print(f"🔍 [EVOLUTION] Platform: {observability_platform}")

    if insights:
        logger.info(f"Received insights of length: {len(insights)}")

    print("🔍 [EVOLUTION] Creating optimization agent...")

    try:
        from evolution.prompt_evolution import OfflineOptimizationType

        optimization_agent = await prompt_evolution_system.create_optimization_agent(
            optimization_type=OfflineOptimizationType.SYSTEM_PROMPT
        )
        logger.info("Optimization agent created successfully")
        print("✅ [EVOLUTION] Optimization agent ready")

        optimization_prompt = f"""
Based on the following insights from {observability_platform}, 
optimize the agent code in {agent_repo}.

User Question: {question}

Insights from Analysis:
{insights}

Repository Path: {agent_repo}
"""

        logger.info("Invoking optimization agent for evolution...")
        print("🤖 [EVOLUTION] Invoking optimization agent (this may take a while)...")

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = optimization_agent.astream(
                    {"messages": [HumanMessage(content=optimization_prompt)]},
                    stream_mode=["updates", "messages"],
                )

                print("\n" + "=" * 80)
                print("EVOLUTION AGENT RESPONSE (streaming):")
                print("=" * 80 + "\n")

                full_response = ""

                async for stream_mode, chunk in response:
                    if stream_mode == "messages":
                        if hasattr(chunk, "content") and chunk.content:
                            if isinstance(chunk.content, str):
                                print(chunk.content, end="", flush=True)
                                full_response += chunk.content
                            elif isinstance(chunk.content, list):
                                for item in chunk.content:
                                    if (
                                        isinstance(item, dict)
                                        and item.get("type") == "text"
                                    ):
                                        text = item.get("text", "")
                                        print(text, end="", flush=True)
                                        full_response += text

                print("\n" + "=" * 80)
                state["optimization_result"] = full_response
                state["evolution_status"] = "completed"
                logger.info("Evolution completed successfully")
                return state
            except GraphInterrupt as e:
                # HITL interrupt detected - SHOW THE PATCH!
                logger.info("HITL Interrupt - File modification requested")
                print("\n" + "⚠️ " * 40)
                print("FILE MODIFICATION REQUESTED - SHOWING PATCH")
                print("⚠️ " * 40)

                # ---- 1) Robustly extract interrupt payload ----
                # Your logs show: (Interrupt(value={...}),)
                # So we need to unwrap nested tuples and then .value/.dict()
                raw_payload = None

                if e.args:
                    raw_payload = e.args[0]
                else:
                    raw_payload = e

                # Unwrap nested 1-element tuples until we get to the real object
                while isinstance(raw_payload, tuple) and len(raw_payload) == 1:
                    raw_payload = raw_payload[0]

                # If it's an Interrupt-like object with `.value`, unwrap that
                if hasattr(raw_payload, "value"):
                    raw_payload = raw_payload.value

                # If it's a Pydantic-ish object with `.dict()`, unwrap that
                if hasattr(raw_payload, "dict"):
                    raw_payload = raw_payload.dict()

                interrupt_payload = raw_payload

                # Now we expect a dict like:
                # {"action_requests": [...], "review_configs": [...]}
                if not isinstance(interrupt_payload, dict):
                    print("\n❓ Could not understand interrupt payload structure.")
                    print(f"Raw interrupt payload: {repr(interrupt_payload)}")
                    logger.error(
                        f"Unexpected interrupt payload type: {type(interrupt_payload)}"
                    )
                    # We cannot show a patch; bail to avoid blocking you
                    state["evolution_status"] = "error"
                    state["error"] = "Unexpected interrupt payload structure"
                    return state

                action_requests = interrupt_payload.get("action_requests", [])

                if not action_requests:
                    print("\n❓ No 'action_requests' found in interrupt payload.")
                    print("Raw interrupt payload:")
                    print(interrupt_payload)
                    logger.error(
                        "GraphInterrupt payload has no 'action_requests'; cannot build patch."
                    )
                    state["evolution_status"] = "error"
                    state["error"] = "No action_requests found in GraphInterrupt"
                    return state

                # ---- 2) Render patch for each write_file request ----
                for request in action_requests:
                    if request.get("name") != "write_file":
                        continue

                    args = request.get("args", {})
                    file_path = args.get("file_path")
                    new_content = args.get("content", "")

                    if not file_path:
                        continue

                    original_path = Path(file_path)

                    if original_path.exists():
                        # Existing file: show unified diff
                        try:
                            original_content = original_path.read_text()
                        except Exception as read_err:
                            print(
                                f"\n❌ Could not read original file {file_path}: {read_err}"
                            )
                            logger.error(
                                f"Could not read file {file_path}: {read_err}",
                                exc_info=True,
                            )
                            continue

                        diff = difflib.unified_diff(
                            original_content.splitlines(keepends=True),
                            new_content.splitlines(keepends=True),
                            fromfile=f"a/{file_path}",
                            tofile=f"b/{file_path}",
                            n=3,
                        )

                        patch_lines = list(diff)

                        print("\n" + "=" * 80)
                        print(f"📝 PATCH FOR: {file_path}")
                        print("=" * 80)

                        for line in patch_lines:
                            if line.startswith("+++") or line.startswith("---"):
                                print(f"\033[1m{line}\033[0m", end="")
                            elif line.startswith("@@"):
                                print(f"\033[36m{line}\033[0m", end="")
                            elif line.startswith("+"):
                                print(f"\033[92m{line}\033[0m", end="")
                            elif line.startswith("-"):
                                print(f"\033[91m{line}\033[0m", end="")
                            else:
                                print(line, end="")

                        print("\n" + "=" * 80)

                        orig_lines = original_content.splitlines()
                        new_lines = new_content.splitlines()
                        added = sum(
                            1
                            for line in patch_lines
                            if line.startswith("+") and not line.startswith("+++")
                        )
                        deleted = sum(
                            1
                            for line in patch_lines
                            if line.startswith("-") and not line.startswith("---")
                        )

                        print("\n📊 CHANGE STATISTICS:")
                        print(f"   Lines added:   \033[92m+{added}\033[0m")
                        print(f"   Lines deleted: \033[91m-{deleted}\033[0m")
                        print(
                            f"   Total lines:   {len(new_lines)} (was {len(orig_lines)})"
                        )
                        # Optional: open diff in VS Code if available
                        try:
                            import shutil
                            import subprocess
                            import tempfile

                            if shutil.which("code") is not None:
                                with tempfile.NamedTemporaryFile(
                                    mode="w",
                                    suffix=original_path.suffix,
                                    delete=False,
                                ) as tmp:
                                    tmp.write(new_content)
                                    temp_path = tmp.name

                                subprocess.run(
                                    ["code", "--diff", str(original_path), temp_path],
                                    check=False,
                                )
                                print(
                                    "\n📂 Opened in VS Code for visual comparison (via `code --diff`)."
                                )
                            else:
                                print(
                                    "\nℹ️ VS Code CLI (`code`) not found on PATH – "
                                    "skipping automatic editor diff. "
                                    "You can still see the full patch above."
                                )
                        except Exception as vs_err:
                            logger.warning(
                                f"Could not open VS Code diff: {vs_err}", exc_info=True
                            )
                            print(
                                "\n⚠️ Could not open VS Code diff. Patch is still printed above."
                            )

                    else:
                        # New file: show first 30 lines as additions
                        print(f"\n🆕 NEW FILE TO BE CREATED: {file_path}")
                        print("=" * 80)
                        lines = new_content.splitlines()
                        for i, line in enumerate(lines[:30], 1):
                            print(f"\033[92m+{i:4}: {line}\033[0m")
                        if len(lines) > 30:
                            print(f"\n... ({len(lines) - 30} more lines)")
                        print("=" * 80)

                # ---- 3) Ask for approval ----
                print("\n" + "=" * 80)
                print("PATCH APPROVAL REQUIRED")
                print("=" * 80)
                print("Review the patch above and choose:")
                print("  [a] Apply patch - Accept these changes")
                print("  [r] Reject patch - Discard these changes")
                print("=" * 80)

                while True:
                    choice = input("\nYour decision [a/r]: ").strip().lower()

                    if choice == "a":
                        print("\n✅ Patch APPROVED - Applying changes...")
                        for request in action_requests:
                            if request.get("name") != "write_file":
                                continue
                            args = request.get("args", {})
                            file_path = args.get("file_path")
                            content = args.get("content", "")
                            if not file_path:
                                continue
                            try:
                                with open(file_path, "w") as f:
                                    f.write(content)
                                logger.info(f"File written: {file_path}")
                                print(f"✅ Patch applied to: {file_path}")
                            except Exception as write_err:
                                logger.error(
                                    f"Failed to write file {file_path}: {write_err}",
                                    exc_info=True,
                                )
                                print(
                                    f"❌ Failed to write file {file_path}: {write_err}"
                                )

                        state["evolution_status"] = "completed_with_approval"
                        state["hitl_decision"] = "approved"
                        return state

                    elif choice == "r":
                        print("\n❌ Patch REJECTED - No changes made")
                        state["evolution_status"] = "rejected_by_user"
                        state["hitl_decision"] = "rejected"
                        logger.info("User rejected the proposed changes")
                        return state
                    else:
                        print("Invalid choice. Please enter 'a' or 'r'.")

            except Exception as e2:
                logger.error(f"Error during evolution: {e2}", exc_info=True)
                print(f"\n❌ Evolution failed: {e2}")
                state["evolution_status"] = "error"
                state["error"] = str(e2)
                return state

        state["evolution_status"] = "max_retries_exceeded"
        return state

    except Exception as e:
        logger.error(f"Error during prompt evolution: {e}", exc_info=True)
        print(f"\n❌ Evolution failed: {e}")
        state["evolution_status"] = "error"
        state["error"] = str(e)
        return state


def route_to_evolution(
    state: UnifiedAgentState
) -> str:
    """
    Routing function that determines whether to invoke the evolution agent.

    Uses a small LLM to analyze:
    1. Whether the insights indicate patterns that could benefit from prompt optimization
    2. Whether the user is asking for prompt improvements
    3. Whether there are performance issues that better prompting could address

    Args:
        state: UnifiedAgentState containing user_question and insights

    Returns:
        "evolve_prompts" to route to evolution agent, or END to skip it
    """
    print("\n🚦 [ROUTING] Deciding whether to invoke prompt evolution...")
    user_question = state.get("user_question", "")
    logger.info(f"Going to check if the user question requires prompt evolution: {user_question}")

    # Prepare the routing prompt
    routing_prompt = router_prompt_template.format(
        user_question=user_question
    )

    try:
        # Invoke the routing LLM
        logger.info("Invoking routing LLM to determine if evolution is needed...")
        print("🤔 [ROUTING] Analyzing user question...")
        response = router_llm.invoke([HumanMessage(content=routing_prompt)])
        routing_decision = response.content.strip().upper()
        logger.info(f"Routing LLM decision: {routing_decision}")

        # Parse the decision - checking for ROUTE_TO_RESEARCH which the prompt should return
        # for evolution requests (the prompt template may say "research" but we use it for evolution)
        if "ROUTE_TO_RESEARCH" in routing_decision or "ROUTE_TO_EVOLUTION" in routing_decision:
            print("➡️  [ROUTING] Decision: Routing to prompt evolution agent")
            logger.info("Routing to prompt evolution agent")
            return "evolve_prompts"
        elif TO_EVOLUTION_HINT in state["spec_insights_marker"]:
            print("➡️  [ROUTING] Decision: Routing to prompt evolution agent")
            logger.info("Routing to prompt evolution agent")
            return "evolve_prompts"
        else:
            print("✋ [ROUTING] Decision: Skipping evolution, ending workflow")
            logger.info("Skipping evolution agent, ending workflow")
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
    workflow.add_node("select_agent_repository", select_agent_repository)
    workflow.add_node("get_insights", get_insights)
    workflow.add_node("evolve_prompts", evolution_engine)

    # Define edges - platform selection happens first
    workflow.add_edge(START, "select_platform")
    workflow.add_edge("select_platform", "get_insights")
    
    # REMOVED: workflow.add_edge("get_insights", "select_agent_repository")
    
    # Conditional routing from get_insights
    workflow.add_conditional_edges(
        "get_insights",
        route_to_evolution,
        {
            "evolve_prompts": "select_agent_repository",  # Route to repo selection first
            END: END  # Skip repo selection if not evolving
        }
    )
    
    # After selecting repo, go to evolution
    workflow.add_edge("select_agent_repository", "evolve_prompts")
    workflow.add_edge("evolve_prompts", END)
    
    logger.info("Graph built successfully with conditional routing and memory checkpointer")
    return workflow.compile(checkpointer=memory)

# Build the graph
app = _build_graph()

@traceable
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
    print("Workflow: Insights Agent -> Evolution Agent")
    print("  1. Insights Agent: Analyzes observability traces and generates insights")
    print("  2. Evolution Agent: Optimizes system prompts based on agent performance")
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
