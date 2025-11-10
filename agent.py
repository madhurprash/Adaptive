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
)

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

# ---------------- INITIALIZE THE INSIGHTS AGENT ----------------
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
    ]
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

class AgentState(TypedDict):
    """State for the log curator agent with message memory."""
    user_question: str
    session_id: Optional[str]
    curated_logs: Dict[str, Any]
    answer: str
    messages: Annotated[List[BaseMessage], add_messages]

@traceable
def get_insights(
    state: AgentState
) -> AgentState:
    """
    Answer user question based on curated logs with conversation memory.

    This node uses an LLM to analyze the curated logs and answer the user's question
    while maintaining conversation context through the messages field.
    """
    logger.info("Generating answer based on curated logs...")
    user_question = state["user_question"]
    curated_logs = state["curated_logs"]
    existing_messages = state.get("messages", [])

    logger.info(f"Found {len(existing_messages)} existing messages in conversation history")

    # Check if there was an error during curation
    if "error" in curated_logs:
        error_msg = f"Unable to answer question: {curated_logs['error']}"
        state["answer"] = error_msg
        # Add error message to conversation history
        new_messages = [HumanMessage(content=user_question), AIMessage(content=error_msg)]
        state["messages"] = new_messages
        return state

    try:
        logger.info("In the GET INSIGHTS NODE...")
        # Create analysis prompt with context
        analysis_prompt = f"""
        ## User Question:
        {user_question}

        ## Curated Logs:
        {json.dumps(curated_logs, indent=2, default=str)}
        """
        logger.info(f"Constructed the insights agent prompt with {len(analysis_prompt)} characters")

        # Prepare messages for the agent: include conversation history + new question
        messages_to_send = list(existing_messages)  # Copy existing messages
        messages_to_send.append(HumanMessage(content=analysis_prompt))

        # Count tokens before sending
        try:
            input_token_count = insights_llm.get_num_tokens_from_messages(messages_to_send)
            logger.info(f"Input token count: {input_token_count}")
        except Exception as e:
            logger.warning(f"Could not count input tokens: {e}")
            input_token_count = None

        logger.info(f"Sending {len(messages_to_send)} messages to insights agent (including history)")

        # Invoke the insights agent with full conversation context
        response = insights_agent.invoke({"messages": messages_to_send})

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

        # Extract the answer from the response messages
        if "messages" in response and len(response["messages"]) > 0:
            # Get the last message (agent's response)
            last_message = response["messages"][-1]
            answer_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
            state["answer"] = answer_text

            # Add both the user question and agent answer to state messages
            # The add_messages reducer will append these to existing messages
            new_messages = [
                HumanMessage(content=user_question),
                AIMessage(content=answer_text)
            ]
            state["messages"] = new_messages
        else:
            answer_text = str(response)
            state["answer"] = answer_text
            new_messages = [
                HumanMessage(content=user_question),
                AIMessage(content=answer_text)
            ]
            state["messages"] = new_messages
        logger.info("Successfully generated answer and updated conversation history")
    except Exception as e:
        logger.error(f"Error generating answer: {e}", exc_info=True)
        error_msg = f"Error generating answer: {str(e)}"
        state["answer"] = error_msg
        # Add error to conversation history
        new_messages = [
            HumanMessage(content=user_question),
            AIMessage(content=error_msg)
        ]
        state["messages"] = new_messages
    return state


def _build_graph() -> StateGraph:
    """Build the log curator graph with memory checkpointer."""
    logger.info("Building log curator graph...")

    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("answer_question", get_insights)

    # Define edges
    workflow.add_edge(START, "answer_question")
    workflow.add_edge("answer_question", END)
    logger.info("Graph built successfully with memory checkpointer")
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

    initial_state: AgentState = {
        "user_question": user_question,
        "session_id": session_id,
        "curated_logs": {},
        "answer": "",
        "messages": [],
    }

    # Configure thread for memory persistence
    config = {"configurable": {"thread_id": thread_id}}

    result = app.invoke(initial_state, config=config)

    return {
        "question": result["user_question"],
        "answer": result["answer"],
        "curated_logs": result["curated_logs"],
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
    print("Self-Healing Agent - Log Curator (Interactive Mode)")
    print("="*80)
    if session_id:
        print(f"Session ID: {session_id}")
    else:
        print("Session ID: Not provided (analysis may be limited)")
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
                        print(f"    A: {entry['answer'][:200]}..." if len(entry['answer']) > 200 else f"    A: {entry['answer']}")
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
                    "answer": result["answer"]
                })

                # Display answer
                print("\n" + "-"*80)
                print(f"Agent: {result['answer']}")
                print("-"*80)

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
    """Main function to run the log curator agent."""
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
