# these are the constants that will be used across the
# agent file
CONFIG_FILE_FPATH: str = "config.yaml"


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

# This is the Claude model id hint to check to not provide the temperature as a part of the 
# LLM initialization
CLAUDE_4_5_SONNET_HINT: str = "sonnet-4-5"

# This is the namespace where the memory will be stored and retrieved from
ERRORS_AND_INSIGHTS_NAMESPACE: str = "errors_and_insights"