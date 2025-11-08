# import langGraph specific libraries
import os
import json
import uuid
import boto3
import argparse
from tools import *
from utils import *
from constants import *
from dotenv import load_dotenv
from tavily import TavilyClient
from langsmith import traceable
from botocore.config import Config
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_aws import ChatBedrock
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import List, Optional, Any, List, Dict, Literal, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# we will add the basic and the advanced model based on the 
# message history size
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

"""
This is an agent that is responsible for autotuning lambda functions
in AWS accounts. This agent has access to certain lambda based tools that will
help this agent to 

1. First, list the log groups for lambda functions and look for lambda functions
in the AWS account. 

2. Second this agent can then fetch metrics for these lambda functions. These 
metrics are defined by the pydantic BaseModel class called MetricsData and tracks the 
following metrics:
- duration_p95_ms: This is the 95th percentile duration in milliseconds
- error_count: These are the total number of errors that the lambda function has had
- throttle_count: Total number of lambda throttles
- error_rate: This is the error rate (errors/invocations)

3. Third, this agent can fetch the lambda configuration. This lambda configuration will contain the 
metadata about the function's setup and runtime environment. This contains:

- Resource settings:
    - MemorySize: This is the memory allocated to the lambda function in MB
    - Timeout: This is the maximum execution time for the lambda function in seconds
    - EphemeralStorage: This is the temporary storage size

- Runtime and code settings:
    - Runtime: This is the runtime language (e.g., Python, Node.js)
    - Handler: This is the entry point for the lambda function
    - CodeSize: This is the size of the deployment package in bytes
    - CodeSha256: This is the SHA256 hash of the deployment package

- Identity and permissions: 
    - FunctionName: This is the name of the function
    - FunctionArn: This is the ARN of the function
    - Role: This is th IAM role ARN used by the function

- Networking settings:
    - VpcConfig: This is the VPC configuration details
    - Environment: This is the environment variables set for the function

- Other settings:
    - LastModified: This is the timestamp of the last modification
    - Layers: This is the list of layers associated with the function
    - Architectures: This is the CPU architecture (e.g., x86_64, arm64)

4. Analyze metrics and configuration: In this tool, the agent will be responsible to check for the
following analysis:

- High duration: If the duration_p95_ms exceeds a certain threshold, it may need more memory.
If the duration is very low, it may be over-provisioned.

- High error rate: If the error_rate exceeds the threshold, then 
investigate and consider increasing memory.

- Throttling: If there are significant throttle_count, consider increasing memory.

5. Decide action: In this, based on the analysis the agent will decide on what to do. 

6. Apply action: Finally, the agent will apply the decided action by updating the lambda
"""
print(f"Loading the env vars...")
load_dotenv()

config_data: Dict = load_config(CONFIG_FILE_FNAME)
print(f"Loaded config data: {json.dumps(config_data, indent=4)}")

# initialize the Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError(
        "TAVILY_API_KEY environment variable not set. "
        "Please set it with: export TAVILY_API_KEY='your-api-key' or use --tavily-api-key"
    )
tavily_client = TavilyClient(api_key=tavily_api_key)
logger.info("Initialized Tavily client for internet search")


# Initialize the bedrock configuration
bedrock_config = Config(
    read_timeout=12000, 
    connect_timeout=60,
    retries={
        'max_attempts': 3,
        'mode': 'standard'
    }
)

# Initialize the bedrock runtime client
bedrock_runtime_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=boto3.session.Session().region_name,
    config=bedrock_config
)

# initialize the bedrock model
basic_default_model = ChatBedrock(
    client=bedrock_runtime_client,
    model=config_data['basic_default_model_information']['model_id'],
    temperature=config_data['basic_default_model_information']["inference_parameters"]["temperature"],
    max_tokens=config_data['basic_default_model_information']["inference_parameters"]["max_tokens"],
    top_p=config_data['basic_default_model_information']["inference_parameters"]["top_p"],
)
print(f"Initialized the basic amazon Bedrock model: {config_data['basic_default_model_information']['model_id']}")

# let's also configure the advanced model to route the request to based on some intelligent routing criteria
advanced_model = ChatBedrock(
    client=bedrock_runtime_client,
    model=config_data['advanced_model_information']['model_id'],
    temperature=config_data['advanced_model_information']["inference_parameters"]["temperature"],
    max_tokens=config_data['advanced_model_information']["inference_parameters"]["max_tokens"],
    top_p=config_data['advanced_model_information']["inference_parameters"]["top_p"],
)
print(f"Initialized the advanced model that the requests will route to when the number of ")

# Next, we will initialize our agent prompt
lambda_autotuner_sys_prompt_fpath: str = config_data['agent_prompts'].get('system_prompt_fpath', 'prompts/lambda_autotuner_agent_system_prompt.txt')
lambda_autotuner_sys_prompt: str = load_system_prompt(lambda_autotuner_sys_prompt_fpath)
print(f"Loaded the lambda agent system prompt: {lambda_autotuner_sys_prompt}")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_default_model

    request.model = model
    return handler(request)


# Represents the internet search tool to be used by the agent
def internet_search(
        query: str,
        max_results: int = 5,
        include_raw_content: bool = False,
        topic: str = "general"):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# Represents the list of tools that the agent will have access to including the lambda tools 
# as well as the internet search tools
tools = [fetch_lambda_metrics, 
         get_lambda_configuration,
         analyze_metrics, 
         decide_action,
         apply_action,
         list_lambda_functions,
         internet_search]

# Create the agent using the create_agent primitive
agent = create_agent(
    model = basic_default_model, # default to the basic model
    tools = tools, 
    middleware=[dynamic_model_selection], 
    system_prompt=lambda_autotuner_sys_prompt
)
print(f"Created the lambda agent: {agent}")

# In this case, we will create a very simple graph -
# since this is not a workflow and more of a dynamic agent, we
# will start off with a base graph structure
class AgentState(TypedDict):
    """State definition for the lambda autotuner agent."""
    messages: List[Any]  # List of messages in the conversation

@traceable
def _invoke_agent(state: AgentState) -> AgentState:
    """
    This is the first node that is called as an entry point
    where the lambda agent is invoked.

    In this case, the agent is allowed to pick and choose the tool
    dynamically based on the user question.

    Args:
        state: Current agent state containing message history

    Returns:
        Updated agent state with new messages
    """
    try:
        # Get the current messages from state
        messages = state.get("messages", [])
        print(f"In AGENT INVOCATION NODE: Going to invoke the lambda agent with the context: {messages}")
        # Invoke the agent with the current messages
        response = agent.invoke({"messages": messages})
        print(f"RESPONSE: {response['messages']}")
        # Return updated state with agent response
        return {"messages": response["messages"]}
    except Exception as e:
        logger.error(f"Error invoking agent: {e}")
        error_message = AIMessage(
            content=f"I encountered an error: {str(e)}. Please try again."
        )
        return {"messages": messages + [error_message]}


# Initialize the checkpointer for session management
checkpointer = InMemorySaver()
logger.info("Initialized InMemorySaver checkpointer for session management")

# Create the graph
graph_builder = StateGraph(AgentState)

# Add the agent node
graph_builder.add_node("agent", _invoke_agent)

# Set the entry point
graph_builder.add_edge(START, "agent")

# Set the end point - agent completes after processing
graph_builder.add_edge("agent", END)

# Compile the graph with checkpointer
graph = graph_builder.compile(checkpointer=checkpointer)
logger.info("Compiled LangGraph with checkpointer")


def run_agent_with_session(
    user_message: str,
    session_id: str,
) -> Dict[str, Any]:
    """
    Run the agent with a user message and session ID for persistence.

    Args:
        user_message: The user's input message
        session_id: Unique identifier for this conversation session

    Returns:
        Dictionary containing the agent's response and full message history
    """
    logger.info(f"Running agent with session_id: {session_id}")
    logger.debug(f"User message: {user_message}")

    # Create the config with thread_id for session management
    config = {
        "configurable": {
            "thread_id": session_id
        }
    }

    # Create the user message
    user_msg = HumanMessage(content=user_message)

    # Invoke the graph with the user message
    result = graph.invoke(
        {"messages": [user_msg]},
        config=config
    )

    logger.info(f"Agent completed processing for session: {session_id}")

    # Extract the last message (agent's response)
    last_message = result["messages"][-1] if result["messages"] else None

    return {
        "response": last_message.content if last_message else "",
        "all_messages": result["messages"],
        "session_id": session_id
    }


def main():
    """
    Main function for CLI interaction with the Lambda Autotuner Agent.

    Enables multi-turn conversations with session persistence.
    """
    parser = argparse.ArgumentParser(
        description="Lambda Autotuner Agent - Optimize AWS Lambda functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Start an interactive session
    uv run python agent.py

    # Start with a specific session ID
    uv run python agent.py --session-id my-session-123

    # Enable debug logging
    uv run python agent.py --debug
        """
    )

    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID for conversation persistence (default: auto-generated UUID)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Generate or use provided session ID
    session_id = args.session_id or str(uuid.uuid4())

    print("\n" + "=" * 70)
    print("Lambda Autotuner Agent - Interactive Mode")
    print("=" * 70)
    print(f"Session ID: {session_id}")
    print("\nThis agent can help you optimize AWS Lambda functions.")
    print("Type 'exit', 'quit', or press Ctrl+C to end the conversation.")
    print("=" * 70 + "\n")

    turn_count = 0

    try:
        while True:
            # Get user input
            try:
                user_input = input("\nYou: ").strip()
            except EOFError:
                print("\nExiting...")
                break

            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nThank you for using Lambda Autotuner Agent. Goodbye!")
                break

            # Skip empty inputs
            if not user_input:
                continue

            turn_count += 1
            logger.info(f"Turn {turn_count}: Processing user message")

            # Run the agent with session management
            try:
                result = run_agent_with_session(
                    user_message=user_input,
                    session_id=session_id
                )

                # Display the agent's response
                print(f"\nAgent: {result['response']}")

                # Log conversation stats
                logger.debug(
                    f"Turn {turn_count} completed. "
                    f"Total messages in session: {len(result['all_messages'])}"
                )

            except Exception as e:
                logger.error(f"Error during agent invocation: {e}", exc_info=True)
                print(f"\nError: An error occurred - {str(e)}")
                print("Please try again or type 'exit' to quit.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")

    print(f"\nSession '{session_id}' ended after {turn_count} turns.")


if __name__ == "__main__":
    main()




    
















