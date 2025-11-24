# these are the constants that will be used across the
# agent file
CONFIG_FILE_FPATH: str = "configs/config.yaml"
EVOLUTION_ENGINE_CONFIG_FILE: str = "configs/prompt_optimization_config.yaml"


# File operation constants
# Base directory for all file operations (relative to project root)
DEFAULT_FILE_BASE_DIR: str = "."

# Valid absolute path prefixes (paths starting with these are allowed)
# All other paths starting with '/' will be converted to relative paths
VALID_ABSOLUTE_PATH_PREFIXES: tuple = (
    '/Users/',
    '/home/',
    '/opt/',
    '/var/',
    '/tmp/',
)

# Whether to use sudo for file operations (not recommended, kept for reference)
# Instead, files should be written to user-accessible directories
USE_SUDO_FOR_FILE_OPS: bool = False

# Default output directory for reports and analysis files
DEFAULT_OUTPUT_DIR: str = "reports"

# This is the namespace where the memory will be stored and retrieved from
ERRORS_AND_INSIGHTS_NAMESPACE: str = "errors_and_insightsn"


# =============================================================================
# MCP SERVER CONSTANTS
# =============================================================================

# MCP Server script paths (relative to project root)
LANGSMITH_MCP_SERVER_PATH: str = "agent_tools/langsmith_mcp_server.py"
LANGFUSE_MCP_SERVER_PATH: str = "agent_tools/langfuse_mcp_server.py"

# MCP Server command and args
MCP_SERVER_COMMAND: str = "uv"
MCP_SERVER_BASE_ARGS: list[str] = ["run", "python"]


# =============================================================================
# OBSERVABILITY PLATFORM CONSTANTS
# =============================================================================

# Supported observability platforms
PLATFORM_LANGSMITH: str = "langsmith"
PLATFORM_LANGFUSE: str = "langfuse"

SUPPORTED_PLATFORMS: tuple[str, ...] = (
    PLATFORM_LANGSMITH,
    PLATFORM_LANGFUSE,
)

# Default platform if none specified
DEFAULT_PLATFORM: str = PLATFORM_LANGSMITH


# =============================================================================
# MODEL CONSTANTS
# =============================================================================

# Model identifier hints for conditional logic
CLAUDE_4_5_SONNET_HINT: str = "sonnet-4-5"

# =============================================================================
# USER INTENT CONSTANTS
# =============================================================================

# These are the user intent constants
TO_EVOLUTION_HINT: str = "ADAPT"
CONTINUE_WITH_INSIGHTS_HINT: str = "INSIGHTS"