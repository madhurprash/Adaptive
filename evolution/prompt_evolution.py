"""
System Prompt Optimization Agents Factory

This module provides optimization agents that can analyze insights from observability
platforms and optimize system prompts accordingly. The agents have file system access
to read and modify prompts in the local repository.

The factory pattern allows dynamic creation of optimization agents based on the
optimization type (code_optimization, performance_optimization, etc.).

Design Pattern:
- Similar to InsightsAgentFactory
- Supports multiple optimization types via enum
- Includes file tools for repository access
- Multi-turn conversation capability
- Middleware stack for context management
"""

import os
import json
import yaml
import asyncio
import logging
from enum import Enum
from constants import *
from langchain_core.tools import Tool
from langchain.agents import create_agent
from typing import Dict, List, Optional, Any
from langchain_aws import ChatBedrockConverse
from utils import load_system_prompt, load_config
# This middleware adds a `write_todos` tool that allows agents to create and manage
# structured task lists for complex multi-step operations. It's designed to help
# agents track progress, organize complex tasks, and provide users with visibility
# into task completion status.
from langchain.agents.middleware import TodoListMiddleware, HumanInTheLoopMiddleware
# Import file tools for repository access
from agent_tools.file_tools import read_file, write_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

class OfflineOptimizationType(Enum):
    """Supported optimization types."""
    # more optimizations can be added here
    SYSTEM_PROMPT = "system_prompt"

class PromptOptimizationAgentFactory:
    """
    Factory for creating optimization agents with file system access.
    This factory:
    1. Loads configuration from config.yaml
    2. Initializes file tools for repository access
    3. Creates optimization-specific agents based on optimization type
    4. Builds agents with context-aware prompts
    5. Applies middleware stack for context management
    Usage:
        factory = PromptOptimizationAgentFactory(config_file_path="config.yaml")
        agent = await factory.create_optimization_agent(
            optimization_type=OptimizationType.SYSTEM_PROMPT
        )
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
        # Load optimization agent configuration
        self.optimization_agent_config = self.config_data.get(
            'prompt_optimization_agent_model_information',
            {}
        )
        # Get repository configuration
        self.repository_config = self.optimization_agent_config.get(
            'agent_repository',
            {}
        )
        self.repository_path = self.repository_config.get(
            'local_path',
            os.getcwd()
        )
        logger.info(f"Repository path: {self.repository_path}")
        # Initialize middleware components
        self._initialize_middleware()

    def _initialize_middleware(self) -> None:
        """Initialize all middleware components including HITL."""
        logger.info("Initializing middleware stack...")
        self.todo_list_middleware = TodoListMiddleware()

        # Initialize HITL middleware if enabled
        hitl_config = self.optimization_agent_config.get('hitl', {})
        hitl_enabled = hitl_config.get('enabled', False)

        if hitl_enabled:
            hitl_config_file = hitl_config.get('config_file', 'hitl_config.yaml')
            self.hitl_middleware = self._load_hitl_middleware(hitl_config_file)
            logger.info(f"✅ HITL middleware enabled from config: {hitl_config_file}")
        else:
            self.hitl_middleware = None
            logger.info("HITL middleware disabled")

    def _load_hitl_middleware(
        self,
        config_file: str
    ) -> HumanInTheLoopMiddleware:
        """
        Load HITL middleware configuration from YAML file.

        Args:
            config_file: Path to HITL config file

        Returns:
            Configured HumanInTheLoopMiddleware instance
        """
        try:
            # Load HITL config file
            with open(config_file, 'r') as f:
                hitl_config = yaml.safe_load(f)

            logger.info(f"Loaded HITL config from: {config_file}")

            # Check if HITL is enabled globally
            if not hitl_config.get('enable_hitl_config', False):
                logger.warning("HITL config found but enable_hitl_config is False")
                return None

            # Build interrupt_on dictionary from tool_names config
            interrupt_on = {}
            tool_names = hitl_config.get('tool_names', {})

            for tool_name, tool_config in tool_names.items():
                if isinstance(tool_config, dict):
                    # Tool has allowed_decisions configuration
                    interrupt_on[tool_name] = tool_config
                    logger.debug(f"  - {tool_name}: {tool_config}")
                elif isinstance(tool_config, bool):
                    # Tool has boolean configuration
                    interrupt_on[tool_name] = tool_config
                    logger.debug(f"  - {tool_name}: {tool_config}")

            logger.info(f"Configured HITL for {len(interrupt_on)} tools")

            # Create and return HITL middleware
            return HumanInTheLoopMiddleware(interrupt_on=interrupt_on)

        except FileNotFoundError:
            logger.error(f"HITL config file not found: {config_file}")
            return None
        except Exception as e:
            logger.error(f"Error loading HITL config: {e}", exc_info=True)
            return None

    def _create_optimization_llm(self) -> ChatBedrockConverse:
        """Create the LLM for the optimization agent."""
        model_id = self.optimization_agent_config.get('model_id')
        inference_params = self.optimization_agent_config.get(
            'inference_parameters',
            {}
        )
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

        logger.info(f"Initialized optimization agent LLM: {model_id}")
        return llm

    def _load_optimization_prompt(
        self,
        optimization_type: OfflineOptimizationType,
    ) -> str:
        """
        Load the optimization-specific system prompt.

        Args:
            optimization_type: Type of optimization to perform

        Returns:
            System prompt string

        Raises:
            ValueError: If prompt configuration is missing or invalid
        """
        # Get optimization-specific prompts configuration
        prompts_config = self.optimization_agent_config.get('prompts', {})
        # Map optimization type enum to config key
        optimization_key = optimization_type.value
        # Get the prompt path for this optimization type
        prompt_path = prompts_config.get(optimization_key)
        if not prompt_path:
            raise ValueError(
                f"No prompt configured for optimization type '{optimization_key}'. "
                f"Please add 'prompts.{optimization_key}' to config.yaml"
            )
        try:
            prompt = load_system_prompt(prompt_path)
            logger.info(
                f"✅ Loaded {optimization_key} optimization prompt from: {prompt_path}"
            )
            return prompt
        except Exception as e:
            logger.error(f"❌ Error loading prompt from {prompt_path}: {e}")
            raise ValueError(
                f"Failed to load {optimization_key} prompt from {prompt_path}: {e}"
            ) from e

    def _create_file_tools(self) -> List[Tool]:
        """
        Create file system tools for repository access.

        Returns:
            List of file operation tools
        """
        logger.info("Creating file system tools for repository access...")

        # The file tools from agent_tools.file_tools are already configured
        # They provide: read_file, write_file
        file_tools = [read_file, write_file]

        logger.info(f"Created {len(file_tools)} file system tools")
        logger.info(f"Available tools: {', '.join([tool.name for tool in file_tools])}")

        return file_tools

    async def create_optimization_agent(
        self,
        optimization_type: OfflineOptimizationType,
    ):
        """
        Create an optimization agent with file system access.
        Args:
            optimization_type: Type of optimization to perform
        Returns:
            Configured optimization agent
        Example:
            >>> factory = PromptOptimizationAgentFactory()
            >>> agent = await factory.create_optimization_agent(
            ...     optimization_type=OptimizationType.SYSTEM_PROMPT
            ... )
            >>> # Use agent with LangGraph...
        """
        logger.info(f"Creating optimization agent for {optimization_type.value}...")
        # Create LLM
        optimization_llm = self._create_optimization_llm()
        # Load optimization-specific prompt
        system_prompt = self._load_optimization_prompt(optimization_type)
        # Create file tools for repository access
        file_tools = self._create_file_tools()
        # Build complete middleware stack
        middleware_stack = [self.todo_list_middleware]

        # Add HITL middleware if enabled
        if self.hitl_middleware:
            middleware_stack.append(self.hitl_middleware)
            logger.info("Added HITL middleware to stack")

        # Create the agent
        logger.info(f"Building {optimization_type.value} optimization agent with "
            f"{len(file_tools)} tools and {len(middleware_stack)} middleware...")
        optimization_agent = create_agent(
            model=optimization_llm,
            tools=file_tools,
            system_prompt=system_prompt,
            middleware=middleware_stack)
        logger.info(
            f"✅ Created {optimization_type.value} optimization agent successfully!\n"
            f"   - Tools: {len(file_tools)}\n"
            f"   - Middleware: {len(middleware_stack)}\n"
            f"   - Model: {self.optimization_agent_config.get('model_id')}\n"
            f"   - Repository: {self.repository_path}"
        )
        return optimization_agent

async def create_optimization_agent_for_type(
    optimization_type: str,
    config_file_path: str = CONFIG_FILE_FPATH,):
    """
    Convenience function to create an optimization agent for a specific type.

    Args:
        optimization_type: Optimization type ("system_prompt", "code_optimization", etc.)
        config_file_path: Path to config.yaml

    Returns:
        Configured optimization agent

    Example:
        >>> agent = await create_optimization_agent_for_type("system_prompt")
        >>> # Use agent with LangGraph...
    """
    # Parse optimization type string to enum
    optimization_type_lower = optimization_type.lower()

    if optimization_type_lower == "system_prompt":
        optimization_enum = OfflineOptimizationType.SYSTEM_PROMPT
    else:
        raise ValueError(
            f"Unsupported optimization type: {optimization_type}. "
            f"Supported types: system_prompt, code_optimization, performance_optimization"
        )

    # Create factory and agent
    factory = PromptOptimizationAgentFactory(config_file_path=config_file_path)
    return await factory.create_optimization_agent(optimization_type=optimization_enum)

def get_optimization_type_from_config(
    config_file_path: str = CONFIG_FILE_FPATH,
) -> str:
    """
    Determine which optimization type to use from config.

    Args:
        config_file_path: Path to config.yaml

    Returns:
        Optimization type ("system_prompt", "code_optimization", etc.)
    """
    config_data = load_config(config_file_path)

    # Check for explicit optimization type configuration
    optimization_config = config_data.get('prompt_optimization_agent_model_information')
    optimization_type = optimization_config.get('default_optimization_type', 'system_prompt')

    logger.info(f"Using optimization type from config: {optimization_type}")
    return optimization_type

async def create_optimization_agent_auto(
    config_file_path: str = CONFIG_FILE_FPATH,
    optimization_type: Optional[str] = None,
):
    """
    Auto-detect optimization type and create appropriate agent.

    Priority order:
    1. Explicit optimization_type parameter
    2. Config file
    3. Default to system_prompt

    Args:
        config_file_path: Path to config.yaml
        optimization_type: Optional explicit optimization type override

    Returns:
        Configured optimization agent

    Example:
        >>> # Auto-detect from config
        >>> agent = await create_optimization_agent_auto()
        >>>
        >>> # Explicit type
        >>> agent = await create_optimization_agent_auto(
        ...     optimization_type="system_prompt"
        ... )
    """
    if optimization_type:
        selected_type = optimization_type.lower()
        logger.info(f"Using explicitly provided optimization type: {selected_type}")
    else:
        # Try config, then default
        try:
            selected_type = get_optimization_type_from_config(config_file_path)
        except Exception as e:
            logger.warning(
                f"Could not detect optimization type from config: {e}, "
                f"defaulting to system_prompt"
            )
            selected_type = "system_prompt"

    logger.info(f"🎯 Creating optimization agent for type: {selected_type}")
    return await create_optimization_agent_for_type(
        optimization_type=selected_type,
        config_file_path=config_file_path,
    )

if __name__ == "__main__":
    """Test the optimization agent factory."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Prompt Optimization Agent Factory"
    )
    parser.add_argument(
        "--optimization-type",
        choices=["system_prompt", "code_optimization", "performance_optimization"],
        help="Optimization type to test",
        default="system_prompt"
    )
    args = parser.parse_args()

    async def main():
        """Main async function for testing."""
        try:
            # Test agent creation
            logger.info(f"Testing {args.optimization_type} optimization agent...")
            agent = await create_optimization_agent_auto(
                optimization_type=args.optimization_type
            )
            logger.info("✅ Successfully created optimization agent!")

            # Print agent info
            print("\n" + "="*80)
            print("OPTIMIZATION AGENT CREATED SUCCESSFULLY")
            print("="*80)
            print(f"Optimization Type: {args.optimization_type}")
            print(f"Agent type: {type(agent).__name__}")
            print("="*80 + "\n")

        except Exception as e:
            logger.error(
                f"❌ Failed to create optimization agent: {e}",
                exc_info=True
            )
            import sys
            sys.exit(1)

    # Run the async main function
    asyncio.run(main())