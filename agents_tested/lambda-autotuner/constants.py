# these are the constants that will be used as a part of this
# agent code creation - for tools and agent development

# TOOL CONSTANTS
DEFAULT_METRIC_PERIOD_MINUTES: int = 30
DEFAULT_MEMORY_STEP_MB: int = 128
MIN_MEMORY_MB: int = 128
MAX_MEMORY_MB: int = 10240
DEFAULT_DURATION_BUDGET_PERCENT: float = 0.70
DEFAULT_ERROR_RATE_THRESHOLD: float = 0.01

# AGENT CONFIGURATION INFORMATION
CONFIG_FILE_FNAME: str = "config.yaml"