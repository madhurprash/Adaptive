"""
Adaptive - Orchestrator-Based Self-Healing AI Agent System

This is an intelligent orchestrator-based multi-agent system that provides:

1. Intelligent Routing: Orchestrator agent analyzes user intent and routes to specialized sub-agents
2. Insights Agent: Analyzes execution traces from observability platforms (LangSmith, Langfuse, MLflow)
3. Evolution Agent: Optimizes prompts and improves agent code based on performance patterns
4. Generate Tasks Agent: Creates synthetic test cases for comprehensive capability testing
5. Direct Responses: Handles simple queries (greetings, clarifications) without sub-agent invocation
6. Conversation Memory: Uses AgentCore Memory for semantic search and context retrieval

The orchestrator uses Claude Sonnet 4 for intelligent decision-making and can handle
both complex multi-agent workflows and simple conversational interactions.
"""
import os
import sys
import json
import uuid
import logging
import difflib
import asyncio
import argparse
from pathlib import Path

# Get the directory where this file is located
_current_dir = Path(__file__).parent.resolve()

# Add the current directory to sys.path if it's not already there
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

# Now import from utils and constants (which are in the same directory)
from utils import *
from constants import *
from typing import Annotated
from dotenv import load_dotenv
from langsmith import traceable
from typing_extensions import TypedDict
from langgraph.errors import GraphInterrupt
from typing import Any, Dict, List, Optional
from langchain_aws import ChatBedrockConverse
from langgraph.graph.message import add_messages
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

# ---------------- ORCHESTRATOR CONFIGURATION ----------------
"""
This section initializes the orchestrator agent that decides which sub-agents
to invoke based on the user's question. The orchestrator can route to:
1. Insights Agent - for trace analysis
2. Evolution Agent - for prompt optimization
3. Generate Tasks Agent - for synthetic task generation
4. Direct Response - for simple queries (greetings, clarifications)
"""
orchestrator_config: Dict = config_data.get('orchestrator_agent', {})
orchestrator_model_id = orchestrator_config.get('model_id', 'us.anthropic.claude-sonnet-4-20250514-v1:0')
orchestrator_inference_params = orchestrator_config.get('inference_parameters', {})

# Initialize orchestrator LLM (Sonnet 4 for intelligent routing)
orchestrator_llm = ChatBedrockConverse(
    model=orchestrator_model_id,
    temperature=orchestrator_inference_params.get('temperature', 0.3),
    max_tokens=orchestrator_inference_params.get('max_tokens', 2048),
    top_p=orchestrator_inference_params.get('top_p', 0.92),
)
logger.info(f"Initialized orchestrator LLM: {orchestrator_llm}")

# Load orchestrator prompt
orchestrator_prompt_path = orchestrator_config.get('prompt_template_path', 'prompt_templates/orchestrator_prompt.txt')
orchestrator_prompt_template = ""
try:
    orchestrator_prompt_template = load_system_prompt(orchestrator_prompt_path)
    logger.info(f"Loaded orchestrator prompt from: {orchestrator_prompt_path}")
except Exception as e:
    logger.error(f"Could not load orchestrator prompt: {e}")
    raise ValueError(f"Orchestrator prompt is required but failed to load: {e}")

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
    Unified state for orchestrator-based multi-agent system.

    This state allows:
    1. Orchestrator to route to appropriate sub-agents
    2. Sub-agents to populate their results
    3. Conversation memory through messages field
    4. User identification for AgentCore Memory
    5. Platform persistence across conversation turns
    """
    user_question: str
    session_id: Optional[str]
    user_id: str  # User/thread identifier for AgentCore Memory
    platform: Optional[str]  # Selected observability platform (persisted across session)
    raw_logs: Dict[str, Any]  # Raw logs from observability platforms
    insights: str  # Insights generated by insights agent
    research_results: str  # Research results from research agent
    output_file_path: str  # Path to output report file
    messages: Annotated[List[BaseMessage], add_messages]
    agent_repo: str  # Agent repository path for evolution
    orchestrator_decision: str  # Routing decision: DIRECT, INSIGHTS, EVOLUTION, GENERATE_TASKS
    orchestrator_response: str  # Direct response from orchestrator (for simple queries)
    orchestrator_reasoning: str  # Reasoning for routing decision

async def orchestrator_node(
    state: UnifiedAgentState
) -> UnifiedAgentState:
    """
    Orchestrator node that decides which sub-agent to invoke or responds directly.

    This node:
    1. Analyzes the user question with conversation history
    2. Uses AgentCore Memory for context retrieval
    3. Decides to route to: INSIGHTS, EVOLUTION, GENERATE_TASKS, or DIRECT response
    4. Stores decision and reasoning in state
    """
    print("\n🎯 [ORCHESTRATOR] Analyzing user request...")
    logger.info("Orchestrator analyzing user request and conversation context")

    user_question = state["user_question"]
    existing_messages = state.get("messages", [])
    user_id = state.get("user_id", "default_user")

    try:
        # STEP 1: Search memory for relevant context
        print("🧠 [ORCHESTRATOR] Retrieving relevant context from memory...")
        enriched_messages = _memory_search_hook(
            messages=list(existing_messages),
            user_id=user_id,
        )

        # STEP 2: Prepare orchestrator prompt with system instructions
        orchestrator_system_message = SystemMessage(content=orchestrator_prompt_template)

        # Add conversation history and current question
        messages_for_orchestrator = [orchestrator_system_message] + enriched_messages + [
            HumanMessage(content=f"User Question: {user_question}\n\nAnalyze this question and decide how to respond.")
        ]

        # STEP 3: Invoke orchestrator LLM
        print("🤔 [ORCHESTRATOR] Making routing decision...")
        logger.info(f"Invoking orchestrator with {len(messages_for_orchestrator)} messages")

        response = await orchestrator_llm.ainvoke(messages_for_orchestrator)
        orchestrator_output = response.content.strip()

        logger.info(f"Orchestrator output: {orchestrator_output}")

        # STEP 4: Parse orchestrator response
        # Expected format:
        # ROUTE: [DIRECT|INSIGHTS|EVOLUTION|GENERATE_TASKS]
        # RESPONSE: [text] (for DIRECT)
        # or
        # REASONING: [text] (for agent routing)

        lines = orchestrator_output.split('\n')
        routing_decision = "DIRECT"  # default
        response_text = ""
        reasoning = ""

        for idx, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith("ROUTE:"):
                routing_decision = line_stripped.replace("ROUTE:", "").strip().upper()
            elif line_stripped.startswith("RESPONSE:"):
                # Collect rest of the lines as part of response
                response_text = '\n'.join(lines[idx:]).replace("RESPONSE:", "").strip()
                break
            elif line_stripped.startswith("REASONING:"):
                # Collect rest of the lines as part of reasoning
                reasoning = '\n'.join(lines[idx:]).replace("REASONING:", "").strip()
                break

        # Store decision in state
        state["orchestrator_decision"] = routing_decision
        state["orchestrator_reasoning"] = reasoning

        if routing_decision == "DIRECT":
            # Direct response - no sub-agent needed
            state["orchestrator_response"] = response_text
            print(f"💬 [ORCHESTRATOR] Decision: Direct response")
            print(f"   Response: {response_text[:100]}...")
            logger.info("Orchestrator decided: DIRECT response")

            # Add to messages for memory
            state["messages"] = [
                HumanMessage(content=user_question),
                AIMessage(content=response_text)
            ]

            # Store in memory
            _memory_store_hook(
                user_id=user_id,
                session_id=state.get("session_id"),
                user_question=user_question,
                ai_response=response_text,
            )
        else:
            # Routing to sub-agent
            print(f"🚀 [ORCHESTRATOR] Decision: Route to {routing_decision}")
            print(f"   Reasoning: {reasoning}")
            logger.info(f"Orchestrator decided: Route to {routing_decision}")
            logger.info(f"Reasoning: {reasoning}")

        return state

    except Exception as e:
        logger.error(f"Error in orchestrator node: {e}", exc_info=True)
        print(f"❌ [ORCHESTRATOR] Error: {e}")
        # Fallback to insights on error
        state["orchestrator_decision"] = "INSIGHTS"
        state["orchestrator_reasoning"] = f"Error in orchestrator, defaulting to insights: {e}"
        return state


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
    print("   3. MLflow (Databricks)")
    while True:
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()

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
        elif choice == "3":
            selected_platform = "mlflow"
            print("✅ Selected: MLflow (Databricks)")
            logger.info("User selected MLflow platform")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

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
        
        # STEP 2: Create platform-specific insights agent dynamically
        print(f"🏗️  [AGENT FACTORY] Creating {platform} insights agent...")
        logger.info(f"Creating insights agent for platform: {platform}")

        from insights.insights_agents import ObservabilityPlatform

        # Convert string to enum
        if platform == PLATFORM_LANGSMITH:
            platform_enum = ObservabilityPlatform.LANGSMITH
        elif platform == PLATFORM_LANGFUSE:
            platform_enum = ObservabilityPlatform.LANGFUSE
        elif platform == PLATFORM_MLFLOW:
            platform_enum = ObservabilityPlatform.MLFLOW
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


async def generate_synthetic_tasks(state: UnifiedAgentState) -> UnifiedAgentState:
    """
    Generate synthetic tasks from execution history using self-questioning.

    This node runs after insights are generated to create targeted test cases
    that help identify capability gaps and untested areas.
    """
    print("\n🧪 [SELF-QUESTIONING] Starting synthetic task generation...")
    logger.info("Generating synthetic tasks from execution history...")

    try:
        # Import self-questioning module
        from evolution.self_questioning.self_questioning import SelfQuestioningModule
        from langchain_aws import BedrockEmbeddings

        # Get configuration
        sq_config = config_data.get('self_questioning', {})
        if not sq_config.get('enabled', False):
            logger.info("Self-questioning is disabled in configuration. Skipping task generation.")
            print("⏭️  [SELF-QUESTIONING] Disabled in configuration - skipping")
            return state

        # Get insights and platform from state
        insights = state.get("insights", "")
        platform = state.get("platform")
        user_id = state.get("user_id", "default_user")

        if not insights:
            logger.info("No insights available for task generation. Skipping.")
            print("⚠️  [SELF-QUESTIONING] No insights available - skipping")
            return state

        print(f"📊 [SELF-QUESTIONING] Analyzing insights for {platform}...")

        # Initialize LLM for task generation
        sq_agent_config = sq_config.get('self_questioning_agent', {})
        sq_llm = ChatBedrockConverse(
            model=sq_agent_config.get('model_id', 'us.anthropic.claude-sonnet-4-20250514-v1:0'),
            temperature=sq_agent_config.get('inference_parameters', {}).get('temperature', 0.7),
            max_tokens=sq_agent_config.get('inference_parameters', {}).get('max_tokens', 4096),
            top_p=sq_agent_config.get('inference_parameters', {}).get('top_p', 0.92),
        )

        # Initialize embedding model
        diversity_config = sq_config.get('diversity_enforcer', {})
        embedding_model = BedrockEmbeddings(
            model_id=diversity_config.get('embedding_model', 'us.amazon.titan-embed-text-v2:0')
        )

        # Initialize self-questioning module
        logger.info("Initializing SelfQuestioningModule...")
        sq_module = SelfQuestioningModule(
            llm=sq_llm,
            embedding_model=embedding_model,
            agentcore_memory=memory_store if agentcore_memory_enabled else None,
            config_dict=sq_config,
        )
        logger.info("✅ SelfQuestioningModule initialized")

        # Create execution history from insights
        # The module expects a list of execution dictionaries
        execution_history = [
            {
                "insights": insights,
                "platform": platform,
            "success": False,  # Assume failures since we're generating improvement tasks
                "task": "Agent execution analysis",
            }
        ]

        # Environment context
        environment_context = {
            "platform": platform,
            "user_id": user_id,
            "codebase_path": sq_module.default_agent_codebase_path,
        }

        # Generate tasks using the module (async call)
        print("🎯 [SELF-QUESTIONING] Generating targeted tasks...")
        generated_tasks = await sq_module.generate_tasks(
            environment_context=environment_context,
            agent_execution_history=execution_history,
            max_tasks=sq_config.get('task_generator', {}).get('max_tasks_per_session', 5),
        )
        logger.info(f"Generated {len(generated_tasks)} synthetic tasks")

        if generated_tasks:
            print(f"✅ [SELF-QUESTIONING] Generated {len(generated_tasks)} synthetic tasks")

            # Add summary to messages
            task_summary = f"\n\n💡 **Self-Questioning**: Generated {len(generated_tasks)} synthetic tasks to test capability gaps."
            if state.get("messages"):
                last_message = state["messages"][-1]
                if isinstance(last_message, AIMessage):
                    last_message.content += task_summary
        else:
            print("⚠️  [SELF-QUESTIONING] No tasks generated")

    except Exception as e:
        logger.exception(f"Error in self-questioning: {e}")
        print(f"❌ [SELF-QUESTIONING] Error: {e}")
        # Don't fail the whole workflow on self-questioning errors

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


def route_from_orchestrator(
    state: UnifiedAgentState
) -> str:
    """
    Routing function based on orchestrator decision.

    Routes to:
    - END if decision is DIRECT (orchestrator already responded)
    - "get_insights" if decision is INSIGHTS (platform already selected)
    - "select_agent_repository" if decision is EVOLUTION (need repo for code changes)
    - "generate_synthetic_tasks" if decision is GENERATE_TASKS

    Args:
        state: UnifiedAgentState containing orchestrator_decision

    Returns:
        Node name to route to, or END
    """
    decision = state.get("orchestrator_decision", "DIRECT")
    logger.info(f"Routing based on orchestrator decision: {decision}")

    if decision == "DIRECT":
        # Orchestrator handled it, end workflow
        return END
    elif decision == "INSIGHTS":
        # Platform already selected at start, go directly to insights
        return "get_insights"
    elif decision == "EVOLUTION":
        # Need repo selection first, then evolution
        return "select_agent_repository"
    elif decision == "GENERATE_TASKS":
        # Go directly to task generation
        return "generate_synthetic_tasks"
    else:
        # Unknown decision, default to END
        logger.warning(f"Unknown orchestrator decision: {decision}, defaulting to END")
        return END


def route_after_tasks(
    state: UnifiedAgentState  # noqa: ARG001
) -> str:
    """
    Routing function after task generation completes.
    Always ends the workflow since tasks are the final step.

    Args:
        state: UnifiedAgentState (unused but required by LangGraph)
    """
    return END


def _build_graph() -> StateGraph:
    """Build the orchestrator-based multi-agent graph with memory checkpointer."""
    logger.info("Building orchestrator-based multi-agent graph...")

    # Create graph with unified state
    workflow = StateGraph(UnifiedAgentState)

    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("select_platform", select_platform)
    workflow.add_node("select_agent_repository", select_agent_repository)
    workflow.add_node("get_insights", get_insights)
    workflow.add_node("generate_synthetic_tasks", generate_synthetic_tasks)
    workflow.add_node("adapt_prompts", evolution_engine)

    # Start with platform selection - run this first before anything else
    workflow.add_edge(START, "select_platform")

    # After platform selection, go to orchestrator
    workflow.add_edge("select_platform", "orchestrator")

    # Conditional routing from orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "get_insights": "get_insights",  # For INSIGHTS
            "select_agent_repository": "select_agent_repository",  # For EVOLUTION
            "generate_synthetic_tasks": "generate_synthetic_tasks",  # For GENERATE_TASKS
            END: END  # For DIRECT responses
        }
    )

    # After insights, end (insights is complete)
    workflow.add_edge("get_insights", END)

    # After repo selection, go to evolution
    workflow.add_edge("select_agent_repository", "adapt_prompts")

    # After evolution, end
    workflow.add_edge("adapt_prompts", END)

    # After task generation, end
    workflow.add_edge("generate_synthetic_tasks", END)

    logger.info("Graph built successfully with orchestrator-based routing and memory checkpointer")
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
        "insights": result.get("insights", ""),
        "research_results": result.get("research_results", ""),
        "output_file_path": result.get("output_file_path", ""),
        "orchestrator_response": result.get("orchestrator_response", ""),
        "orchestrator_decision": result.get("orchestrator_decision", ""),
        "orchestrator_reasoning": result.get("orchestrator_reasoning", ""),
    }


def _parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Optional list of arguments to parse. If None, parses sys.argv.
    """
    parser = argparse.ArgumentParser(
        description="Adaptive - Continuous optimization for AI agents: Analyze agent execution traces from LangSmith",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Interactive mode (default)
    python adaptive.py --session-id "my-session-123"

    # Interactive mode with environment variable
    export LANGSMITH_SESSION_ID="my-session-123"
    python adaptive.py

    # Enable debug logging
    python adaptive.py --session-id "abc123" --debug
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

    return parser.parse_args(args)


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
    print("Adaptive - Orchestrator-Based Multi-Agent System (Interactive Mode)")
    print("="*80)
    print("Architecture: Orchestrator → Sub-Agents")
    print("  • Orchestrator: Intelligently routes requests to specialized sub-agents")
    print("  • Sub-Agents:")
    print("    - Insights: Analyzes observability traces and generates insights")
    print("    - Evolution: Optimizes prompts and improves agent code")
    print("    - Generate Tasks: Creates synthetic test cases for capability testing")
    print("\nFeatures:")
    print("  • Handles simple queries directly (greetings, questions)")
    print("  • Routes complex tasks to specialized agents")
    print("  • Conversation memory with semantic search (AgentCore)")
    if session_id:
        print(f"\nSession ID: {session_id}")
    else:
        print("\nSession ID: Not provided (analysis may be limited)")
    print(f"Thread ID: {thread_id} (conversation memory enabled)")
    print("\nExamples:")
    print("  - 'Hello' → Direct response from orchestrator")
    print("  - 'What errors occurred?' → Routes to Insights agent")
    print("  - 'Optimize my prompts' → Routes to Evolution agent")
    print("  - 'Generate test cases' → Routes to Generate Tasks agent")
    print("\nCommands: 'quit', 'exit', or 'done' to exit")
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

                # Check if this was a direct response from orchestrator
                orchestrator_response = result.get('orchestrator_response')
                if orchestrator_response:
                    print(orchestrator_response)
                else:
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


def main(
    session_id: Optional[str] = None,
    debug: bool = False,
    parse_cli_args: bool = True
) -> None:
    """Main function to run the unified multi-agent workflow.

    Args:
        session_id: Optional session ID (overrides CLI args if provided)
        debug: Enable debug logging (overrides CLI args if provided)
        parse_cli_args: If True, parse CLI arguments. If False, use provided parameters.
    """
    if parse_cli_args:
        args = _parse_args()
        session_id = session_id or args.session_id
        debug = debug or args.debug

    # Set debug logging if requested
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Get session ID from CLI or environment
    session_id = _get_session_id(session_id)

    if not session_id:
        logger.warning("No session ID provided. Analysis may be limited without a specific session.")

    try:
        _run_interactive_session(session_id)
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 
