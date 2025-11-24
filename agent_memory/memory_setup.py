"""
AgentCore Memory Setup

This module initializes and manages AgentCore Memory for the insights agent.
Uses default strategies (user preferences, semantics, session summaries) plus
a custom strategy for error analysis and insights extraction.
"""

import logging
from typing import Optional, Dict, Any

from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.memory.constants import StrategyType
from langgraph.store.base import BaseStore

from .memory_prompts import (
    ERROR_INSIGHTS_EXTRACTION_PROMPT,
    ERROR_INSIGHTS_CONSOLIDATION_PROMPT,
)

logger = logging.getLogger(__name__)


def initialize_agentcore_memory(
    memory_name: str,
    region_name: str,
    memory_execution_role_arn: str,
    model_id: str = "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    description: str = "Self-healing agent conversation memory",
) -> Dict[str, Any]:
    """
    Initialize or retrieve AgentCore Memory with default strategies plus custom error analysis.

    Creates a memory with:
    - Custom ERROR_INSIGHTS strategy (extracts errors, insights, and solutions)

    Args:
        memory_name: Name for the memory instance
        region_name: AWS region
        memory_execution_role_arn: IAM role ARN with permissions for memory operations
        model_id: Model ID for extraction and consolidation
        description: Description of the memory purpose

    Returns:
        Dictionary containing memory information including 'id'
    """
    logger.info(f"Initializing AgentCore Memory: {memory_name} in region {region_name}")

    try:
        # Initialize memory client
        client = MemoryClient(region_name=region_name)

        # Create or get memory with default strategies + custom error insights
        memory = client.create_or_get_memory(
            name=memory_name,
            description=description,
            memory_execution_role_arn=memory_execution_role_arn,
            strategies=[
                # Custom strategy: Error insights and analysis
                {
                    StrategyType.CUSTOM.value: {
                        "name": "ErrorInsights",
                        "description": "Extracts errors, insights, solutions, and research findings",
                        "namespaces": ["/{actorId}/errors_and_insights"],
                        "configuration": {
                            "userPreferenceOverride": {
                                "extraction": {
                                    "appendToPrompt": ERROR_INSIGHTS_EXTRACTION_PROMPT,
                                    "modelId": model_id,
                                },
                                "consolidation": {
                                    "appendToPrompt": ERROR_INSIGHTS_CONSOLIDATION_PROMPT,
                                    "modelId": model_id,
                                }
                            }
                        }
                    }
                },
            ]
        )
        memory_id = memory["id"]
        logger.info(f"Successfully initialized AgentCore Memory with ID: {memory_id}")
        logger.info(f"Memory strategies: USER_PREFERENCES, SEMANTICS, SESSION_SUMMARY, CUSTOM (ErrorInsights)")
        return memory
    except Exception as e:
        logger.error(f"Error initializing AgentCore Memory: {e}", exc_info=True)
        raise

def get_memory_store(
    memory_id: str,
    region_name: str,
) -> BaseStore:
    """
    Get a BaseStore instance connected to AgentCore Memory.

    This store can be used with LangGraph for conversation persistence
    and retrieval.

    Args:
        memory_id: AgentCore Memory ID
        region_name: AWS region

    Returns:
        BaseStore instance for LangGraph integration
    """
    try:
        from langgraph_checkpoint_aws import AgentCoreMemoryStore

        logger.info(f"Creating AgentCore Memory Store for memory_id: {memory_id}")

        store = AgentCoreMemoryStore(
            memory_id=memory_id,
            region_name=region_name,
        )

        logger.info("Successfully created AgentCore Memory Store")
        return store

    except ImportError as e:
        logger.error(
            "langgraph-checkpoint-aws not installed. "
            "Run: uv add langgraph-checkpoint-aws"
        )
        raise
    except Exception as e:
        logger.error(f"Error creating memory store: {e}", exc_info=True)
        raise


def get_memory_checkpointer(
    memory_id: str,
    region_name: str,
):
    """
    Get an AgentCore Memory checkpointer for LangGraph.

    This checkpointer saves conversation state and works seamlessly
    with the AgentCore Memory Store.

    Args:
        memory_id: AgentCore Memory ID
        region_name: AWS region

    Returns:
        AgentCoreMemorySaver instance
    """
    try:
        from langgraph_checkpoint_aws import AgentCoreMemorySaver

        logger.info(f"Creating AgentCore Memory Saver for memory_id: {memory_id}")

        checkpointer = AgentCoreMemorySaver(
            memory_id=memory_id,
            region_name=region_name,
        )

        logger.info("Successfully created AgentCore Memory Saver")
        return checkpointer

    except ImportError as e:
        logger.error(
            "langgraph-checkpoint-aws not installed. "
            "Run: uv add langgraph-checkpoint-aws"
        )
        raise
    except Exception as e:
        logger.error(f"Error creating memory checkpointer: {e}", exc_info=True)
        raise
