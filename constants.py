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