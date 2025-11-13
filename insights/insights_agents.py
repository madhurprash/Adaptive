"""
Insights Agents Factory with MCP Tool Integration

This module provides platform-specific insights agents (LangSmith, Langfuse)
that connect to their respective MCP servers for observability tool access.

The factory pattern allows the main evolve_agents.py to dynamically select
the appropriate agent based on the observability platform being used.
"""

import os
import json
import asyncio
import logging
from enum import Enum
from langchain_core.tools import Tool
from langchain.agents import create_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from typing import Dict, List, Optional, Any
from langchain_aws import ChatBedrockConverse

# Import middleware
from agent_middleware.context_middleware import (
    # this is all the context engineering middleware which will be used
    TokenLimitCheckMiddleware,
    tool_response_summarizer,
    PruneToolCallMiddleware,
)
from langchain.agents.middleware import (
    # this is the prebuild middleware
    SummarizationMiddleware,
    TodoListMiddleware,
)

from utils import load_system_prompt, load_config
from constants import *


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


class ObservabilityPlatform(Enum):
    """Supported observability platforms."""
    LANGSMITH = "langsmith"
    LANGFUSE = "langfuse"


class InsightsAgentFactory:
    """
    Factory for creating platform-specific insights agents with MCP tool integration.

    This factory:
    1. Loads configuration from config.yaml
    2. Initializes the appropriate MCP server based on platform
    3. Creates tools from MCP server
    4. Builds the insights agent with platform-specific prompt and tools
    5. Applies middleware stack for context management
    """

    def __init__(
        self,
        config_file_path: str = CONFIG_FILE_FPATH,
    ):
        """
        Initialize the factory with configuration.

        Args:
            config_file_path: Path to config.yaml file
        """
        self.config_data = load_config(config_file_path)
        logger.info(f"Loaded configuration from {config_file_path}")

        # Load context engineering and middleware configs
        self.context_engineering_info = self.config_data.get('context_engineering_info', {})
        self.insights_agent_config = self.config_data.get('insights_agent_model_information', {})

        # Initialize middleware components
        self._initialize_middleware()

    def _initialize_middleware(self) -> None:
        """Initialize all middleware components."""
        logger.info("Initializing middleware stack...")

        # Get summarization config
        summarization_config = self.context_engineering_info.get('summarization_middleware', {})

        # Initialize summarization LLM
        self.summarization_llm = ChatBedrockConverse(
            model=summarization_config.get("model_id"),
            temperature=summarization_config.get('temperature', 0.1),
            max_tokens=summarization_config.get('max_tokens', 2000),
        )
        logger.info(f"Initialized summarization LLM: {summarization_config.get('model_id')}")

        # Load custom summarization prompt
        summary_prompt_path = summarization_config.get('summary_prompt')
        self.summary_prompt_text = None
        if summary_prompt_path:
            try:
                self.summary_prompt_text = load_system_prompt(summary_prompt_path)
                logger.info(f"Loaded custom summarization prompt from: {summary_prompt_path}")
            except Exception as e:
                logger.warning(f"Could not load summary prompt: {e}, using default")

        # Create middleware components
        middleware_params = {
            "model": self.summarization_llm,
            "max_tokens_before_summary": summarization_config.get("max_tokens_before_summary", 4000),
            "messages_to_keep": summarization_config.get("messages_to_keep", 20),
        }
        if self.summary_prompt_text:
            middleware_params["summary_prompt"] = self.summary_prompt_text

        self.conversation_summarization_middleware = SummarizationMiddleware(**middleware_params)
        self.todo_list_middleware = TodoListMiddleware()
        self.prune_middleware = PruneToolCallMiddleware(
            tools_to_prune=None,
            max_error_length=500,
            max_input_length=200
        )

        logger.info("Middleware stack initialized successfully")

    def _create_insights_llm(self) -> ChatBedrockConverse:
        """Create the LLM for the insights agent."""
        model_id = self.insights_agent_config.get('model_id')
        inference_params = self.insights_agent_config.get('inference_parameters', {})

        if CLAUDE_4_5_SONNET_HINT not in model_id:
            llm = ChatBedrockConverse(
                model=model_id,
                temperature=inference_params.get("temperature", 0.1),
                max_tokens=inference_params.get("max_tokens", 8192),
                top_p=inference_params.get("top_p", 0.92),
            )
        else:
            # Claude 4.5 Sonnet does not accept temperature or top_p
            llm = ChatBedrockConverse(
                model=model_id,
                max_tokens=inference_params.get("max_tokens", 8192),
            )

        logger.info(f"Initialized insights agent LLM: {model_id}")
        return llm

    def _load_platform_prompt(
        self,
        platform: ObservabilityPlatform,
    ) -> str:
        """
        Load the insights agent system prompt (same for all platforms).

        Uses the framework-agnostic log_curator_base_prompt.txt for all platforms.

        Args:
            platform: Observability platform

        Returns:
            System prompt string
        """
        # Use the same prompt for all platforms (framework-agnostic)
        prompt_path = self.insights_agent_config.get('insights_agent_prompt')
        try:
            prompt = load_system_prompt(prompt_path)
            logger.info(f"Loaded insights agent prompt from: {prompt_path} (framework-agnostic)")
            return prompt
        except Exception as e:
            logger.error(f"Error loading prompt from {prompt_path}: {e}")

    def _wrap_async_tool_for_sync(self, tool: Tool) -> Tool:
        """
        Wrap an async-only tool to support both sync and async invocation.

        Args:
            tool: The async-only tool

        Returns:
            Tool that supports both sync and async invocation
        """
        from langchain_core.tools import StructuredTool

        # Create sync wrapper that runs async function in event loop
        # Accept both positional and keyword arguments to handle different invocation patterns
        def sync_wrapper(*args, **kwargs) -> Any:
            # If called with positional args, use first arg as input
            if args:
                input_data = args[0]
            # Otherwise, use kwargs as input
            else:
                input_data = kwargs
            return asyncio.run(tool.ainvoke(input_data))

        # Create async wrapper that delegates to original
        async def async_wrapper(*args, **kwargs) -> Any:
            # If called with positional args, use first arg as input
            if args:
                input_data = args[0]
            # Otherwise, use kwargs as input
            else:
                input_data = kwargs
            return await tool.ainvoke(input_data)

        # Return new tool with both sync and async support
        return StructuredTool(
            name=tool.name,
            description=tool.description,
            func=sync_wrapper,
            coroutine=async_wrapper,
            args_schema=tool.args_schema if hasattr(tool, 'args_schema') else None,
        )

    async def _create_mcp_tools(
        self,
        platform: ObservabilityPlatform,
    ) -> List[Tool]:
        """
        Create tools from MCP server for the specified platform.

        Args:
            platform: Observability platform

        Returns:
            List of LangChain tools with both sync and async support
        """
        logger.info(f"Creating MCP tools for {platform.value}...")

        try:
            # Determine MCP server script path based on platform using constants
            if platform == ObservabilityPlatform.LANGSMITH:
                server_script = LANGSMITH_MCP_SERVER_PATH
            elif platform == ObservabilityPlatform.LANGFUSE:
                server_script = LANGFUSE_MCP_SERVER_PATH
            else:
                raise ValueError(f"Unsupported platform: {platform}")

            logger.info(f"Connecting to MCP server: {server_script}")

            # Create MCP server parameters
            server_params = StdioServerParameters(
                command=MCP_SERVER_COMMAND,
                args=MCP_SERVER_BASE_ARGS + [server_script],
                env=os.environ.copy(),
            )

            # Create MCP client connection and load tools
            # The load_mcp_tools function starts the MCP server as a subprocess
            # and returns tools that can be used by the agent
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    # Load MCP tools (these are async-only by default)
                    mcp_tools = await load_mcp_tools(session)

            logger.info(f"Successfully created {len(mcp_tools)} MCP tools from {platform.value} server")

            # Wrap tools to support both sync and async invocation
            wrapped_tools = [self._wrap_async_tool_for_sync(tool) for tool in mcp_tools]
            logger.info(f"Wrapped {len(wrapped_tools)} tools for sync/async compatibility")

            # Log tool names for debugging
            tool_names = [tool.name for tool in wrapped_tools]
            logger.info(f"Available tools: {', '.join(tool_names)}")

            return wrapped_tools

        except Exception as e:
            logger.error(f"Error creating MCP tools for {platform.value}: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to initialize MCP tools for {platform.value}. "
                f"Ensure the MCP server is properly configured. Error: {e}"
            )

    def create_insights_agent(
        self,
        platform: ObservabilityPlatform,
    ):
        """
        Create a platform-specific insights agent with MCP tools.

        Args:
            platform: Observability platform (LangSmith or Langfuse)

        Returns:
            Configured insights agent
        """
        logger.info(f"Creating insights agent for {platform.value}...")

        # Create LLM
        insights_llm = self._create_insights_llm()

        # Load platform-specific prompt
        system_prompt = self._load_platform_prompt(platform)

        # Create MCP tools (async operation)
        mcp_tools = asyncio.run(self._create_mcp_tools(platform))

        # Initialize platform-specific middleware (token limit check needs LLM)
        token_limit_middleware = TokenLimitCheckMiddleware(
            model=insights_llm,
            summarization_llm=self.summarization_llm,
            token_threshold=self.context_engineering_info.get('token_threshold', 100000),
            summary_prompt=self.summary_prompt_text,
        )

        tool_summarizer_middleware = tool_response_summarizer(
            summarization_llm=self.summarization_llm,
            token_threshold=self.context_engineering_info.get('token_threshold', 100000),
            store_full_responses=True,
            summary_prompt=self.summary_prompt_text,
        )

        # Build complete middleware stack
        middleware_stack = [
            token_limit_middleware,
            tool_summarizer_middleware,
            self.todo_list_middleware,
            self.conversation_summarization_middleware,
            self.prune_middleware,
        ]

        # Create the agent
        logger.info(f"Building {platform.value} insights agent with {len(mcp_tools)} tools and {len(middleware_stack)} middleware...")

        insights_agent = create_agent(
            model=insights_llm,
            tools=mcp_tools,
            system_prompt=system_prompt,
            middleware=middleware_stack,
        )

        logger.info(
            f"✅ Created {platform.value} insights agent successfully!\n"
            f"   - Tools: {len(mcp_tools)}\n"
            f"   - Middleware: {len(middleware_stack)}\n"
            f"   - Model: {self.insights_agent_config.get('model_id')}"
        )

        return insights_agent


def create_insights_agent_for_platform(
    platform: str,
    config_file_path: str = CONFIG_FILE_FPATH,
):
    """
    Convenience function to create an insights agent for a platform.

    Args:
        platform: Platform name ("langsmith" or "langfuse")
        config_file_path: Path to config.yaml

    Returns:
        Configured insights agent

    Example:
        >>> agent = create_insights_agent_for_platform("langsmith")
        >>> # Use agent with LangGraph...
    """
    # Parse platform string to enum
    platform_lower = platform.lower()

    if platform_lower == "langsmith":
        platform_enum = ObservabilityPlatform.LANGSMITH
    elif platform_lower == "langfuse":
        platform_enum = ObservabilityPlatform.LANGFUSE
    else:
        raise ValueError(
            f"Unsupported platform: {platform}. "
            f"Supported platforms: langsmith, langfuse"
        )

    # Create factory and agent
    factory = InsightsAgentFactory(config_file_path=config_file_path)
    return factory.create_insights_agent(platform=platform_enum)


def get_platform_from_config(
    config_file_path: str = CONFIG_FILE_FPATH,
) -> str:
    """
    Determine which observability platform to use from config.

    Args:
        config_file_path: Path to config.yaml

    Returns:
        Platform name ("langsmith" or "langfuse")
    """
    config_data = load_config(config_file_path)

    # Check for explicit platform configuration
    insights_config = config_data.get('insights_agent_model_information', {})
    platform = insights_config.get('observability_platform', 'langsmith').lower()

    logger.info(f"Using observability platform from config: {platform}")
    return platform


def get_platform_from_env() -> str:
    """
    Determine platform from environment variables.

    Checks for:
    - OBSERVABILITY_PLATFORM env var
    - LANGSMITH_API_KEY presence -> langsmith
    - LANGFUSE_PUBLIC_KEY presence -> langfuse

    Returns:
        Platform name ("langsmith" or "langfuse")
    """
    # Check explicit env var
    platform = os.getenv("OBSERVABILITY_PLATFORM", "").lower()
    if platform in ["langsmith", "langfuse"]:
        logger.info(f"Using platform from OBSERVABILITY_PLATFORM: {platform}")
        return platform

    # Auto-detect from API keys
    if os.getenv("LANGSMITH_API_KEY"):
        logger.info("Detected LANGSMITH_API_KEY, using langsmith platform")
        return "langsmith"

    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        logger.info("Detected LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, using langfuse platform")
        return "langfuse"

    # Default to langsmith
    logger.warning("No platform detected, defaulting to langsmith")
    return "langsmith"


def create_insights_agent_auto(
    config_file_path: str = CONFIG_FILE_FPATH,
    platform: Optional[str] = None,
):
    """
    Auto-detect platform and create appropriate insights agent.

    Priority order:
    1. Explicit platform parameter
    2. OBSERVABILITY_PLATFORM env var
    3. Config file
    4. API key detection
    5. Default to langsmith

    Args:
        config_file_path: Path to config.yaml
        platform: Optional explicit platform override

    Returns:
        Configured insights agent

    Example:
        >>> # Auto-detect from config/env
        >>> agent = create_insights_agent_auto()
        >>>
        >>> # Explicit platform
        >>> agent = create_insights_agent_auto(platform="langfuse")
    """
    if platform:
        selected_platform = platform.lower()
        logger.info(f"Using explicitly provided platform: {selected_platform}")
    else:
        # Try environment first, then config
        try:
            selected_platform = get_platform_from_env()
        except Exception as e:
            logger.warning(f"Could not detect platform from env: {e}")
            try:
                selected_platform = get_platform_from_config(config_file_path)
            except Exception as e2:
                logger.warning(f"Could not detect platform from config: {e2}")
                selected_platform = "langsmith"

    logger.info(f"🎯 Creating insights agent for platform: {selected_platform}")
    return create_insights_agent_for_platform(
        platform=selected_platform,
        config_file_path=config_file_path,
    )


if __name__ == "__main__":
    """Test the insights agent factory."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Insights Agent Factory")
    parser.add_argument(
        "--platform",
        choices=["langsmith", "langfuse"],
        help="Observability platform to test"
    )
    args = parser.parse_args()

    try:
        # Test auto-detection
        logger.info("Testing auto-detection...")
        agent = create_insights_agent_auto(platform=args.platform)
        logger.info("✅ Successfully created insights agent!")

        # Print agent info
        print("\n" + "="*80)
        print("INSIGHTS AGENT CREATED SUCCESSFULLY")
        print("="*80)
        print(f"Platform: {args.platform or 'auto-detected'}")
        print(f"Agent type: {type(agent).__name__}")
        print("="*80 + "\n")

    except Exception as e:
        logger.error(f"❌ Failed to create insights agent: {e}", exc_info=True)
        import sys
        sys.exit(1)
