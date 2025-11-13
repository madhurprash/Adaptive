"""
Agent Memory Module

This module provides AgentCore Memory integration for the insights agent,
enabling efficient context management with semantic search and session summaries.
"""

from .memory_setup import (
    # these are the two functions that we will be using from the memory module
    initialize_agentcore_memory,
    get_memory_store,
)

__all__ = [
    "initialize_agentcore_memory",
    "get_memory_store",
]
