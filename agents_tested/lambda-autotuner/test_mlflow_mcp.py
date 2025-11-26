#!/usr/bin/env python3
"""
Simple test script to verify MLflow MCP server functionality.
Tests the basic connection and tool availability.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


async def test_mlflow_mcp_server():
    """Test the MLflow MCP server connection and basic functionality."""

    logger.info("=" * 80)
    logger.info("MLflow MCP Server Test")
    logger.info("=" * 80)

    # Check environment variables
    logger.info("\n1. Checking environment variables...")
    required_env_vars = {
        "DATABRICKS_HOST": os.getenv("DATABRICKS_HOST"),
        "DATABRICKS_TOKEN": os.getenv("DATABRICKS_TOKEN"),
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
    }

    for key, value in required_env_vars.items():
        if value:
            masked_value = value[:10] + "..." if len(value) > 10 else value
            logger.info(f"   ✓ {key}: {masked_value}")
        else:
            logger.warning(f"   ✗ {key}: NOT SET")

    # Import MCP client
    logger.info("\n2. Importing MCP client libraries...")
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        logger.info("   ✓ Successfully imported MultiServerMCPClient")
    except ImportError as e:
        logger.error(f"   ✗ Failed to import MCP client: {e}")
        sys.exit(1)

    # Configure MCP server
    logger.info("\n3. Configuring MLflow MCP server...")
    server_config = {
        "mlflow": {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "--with", "mlflow[mcp]>=3.5.1", "mlflow", "mcp", "run"],
            "env": {
                "DATABRICKS_HOST": os.getenv("DATABRICKS_HOST", ""),
                "DATABRICKS_TOKEN": os.getenv("DATABRICKS_TOKEN", ""),
                "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "databricks"),
            }
        }
    }

    # Remove empty env vars
    server_config["mlflow"]["env"] = {
        k: v for k, v in server_config["mlflow"]["env"].items() if v
    }

    logger.info(f"   Command: {server_config['mlflow']['command']} {' '.join(server_config['mlflow']['args'])}")
    logger.info(f"   Env vars: {list(server_config['mlflow']['env'].keys())}")

    # Initialize MCP client with timeout
    logger.info("\n4. Initializing MCP client (with 60s timeout)...")
    start_time = datetime.now()

    try:
        mcp_client = MultiServerMCPClient(server_config)
        logger.info("   ✓ MCP client created")

        # Get tools with timeout
        logger.info("\n5. Fetching available tools from MCP server...")
        logger.info("   (This may take a while on first run - installing MLflow...)")

        tools = await asyncio.wait_for(
            mcp_client.get_tools(),
            timeout=60.0
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"   ✓ Successfully connected in {elapsed:.1f}s")
        logger.info(f"   ✓ Found {len(tools)} tools")

        # List available tools
        logger.info("\n6. Available MLflow MCP tools:")
        for i, tool in enumerate(tools, 1):
            logger.info(f"   {i}. {tool.name}")
            if hasattr(tool, 'description') and tool.description:
                logger.info(f"      Description: {tool.description[:100]}...")

        logger.info("\n" + "=" * 80)
        logger.info("✅ SUCCESS: MLflow MCP server is working correctly!")
        logger.info("=" * 80)

        return True

    except asyncio.TimeoutError:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"\n   ✗ Timeout after {elapsed:.1f}s waiting for MCP server")
        logger.error("\n" + "=" * 80)
        logger.error("❌ FAILED: MCP server initialization timed out")
        logger.error("=" * 80)
        logger.error("\nPossible causes:")
        logger.error("  1. First-time installation of mlflow[mcp] is taking too long")
        logger.error("  2. Databricks connection is slow or timing out")
        logger.error("  3. Network connectivity issues")
        logger.error("\nTroubleshooting:")
        logger.error("  - Try running manually: uv run --with 'mlflow[mcp]>=3.5.1' mlflow mcp run")
        logger.error("  - Check Databricks credentials and connectivity")
        logger.error("  - Verify MLFLOW_TRACKING_URI is set correctly")
        return False

    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"\n   ✗ Error after {elapsed:.1f}s: {e}")
        logger.error("\n" + "=" * 80)
        logger.error("❌ FAILED: MCP server initialization failed")
        logger.error("=" * 80)
        logger.exception("Full error details:")
        return False


def main():
    """Main entry point."""
    try:
        success = asyncio.run(test_mlflow_mcp_server())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("\n\nTest interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
