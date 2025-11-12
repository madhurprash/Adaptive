#!/bin/bash

# Exit on error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load environment variables from .env file
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
else
    echo "Warning: .env file not found at $SCRIPT_DIR/.env"
    exit 1
fi

# Check if DD_API_KEY is set
if [ -z "$DD_API_KEY" ]; then
    echo "Error: DD_API_KEY is not set in .env file"
    exit 1
fi

# Set Datadog LLM Observability environment variables
export DD_LLMOBS_ENABLED=1
export DD_LLMOBS_ML_APP=${DD_LLMOBS_ML_APP:-"lambda-autotuner"}
export DD_SITE=${DD_SITE:-"datadoghq.com"}

# Configure agentless mode (send directly to Datadog, no local agent needed)
export DD_TRACE_AGENT_URL=""  # Disable local agent connection attempts
export DD_LLMOBS_AGENTLESS_ENABLED=1  # Enable agentless mode for LLMObs

echo "=========================================="
echo "Starting Lambda Autotuner Agent with Datadog LLM Observability"
echo "=========================================="
echo "DD_LLMOBS_ENABLED: $DD_LLMOBS_ENABLED"
echo "DD_LLMOBS_ML_APP: $DD_LLMOBS_ML_APP"
echo "DD_SITE: $DD_SITE"
echo "DD_LLMOBS_AGENTLESS_ENABLED: $DD_LLMOBS_AGENTLESS_ENABLED"
echo "DD_API_KEY: ${DD_API_KEY:0:10}..." # Show only first 10 chars for security
echo "=========================================="

# Run the agent with ddtrace-run for automatic instrumentation
cd "$SCRIPT_DIR"
uv run ddtrace-run python run_synthetic_questions.py "$@"
