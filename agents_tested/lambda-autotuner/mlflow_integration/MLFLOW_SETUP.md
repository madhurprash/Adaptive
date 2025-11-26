# MLflow Tracing Setup for Lambda Autotuner

This guide explains how to set up and use MLflow tracing with Databricks for the Lambda Autotuner agent to monitor and analyze agent performance.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Running with MLflow](#running-with-mlflow)
- [What Gets Tracked](#what-gets-tracked)
- [Viewing Data in MLflow](#viewing-data-in-mlflow)
- [Troubleshooting](#troubleshooting)

## Overview

MLflow Tracing provides comprehensive monitoring for AI agents and LLM applications. For the Lambda Autotuner agent, we track:

- **Agent Workflows**: Complete agent execution flows with timing and success metrics
- **Synthetic Questions**: Individual question evaluation with category and difficulty tracking
- **LLM Calls**: Automatic instrumentation of all LangChain LLM calls via autologging
- **Tool Executions**: Tracking of AWS Lambda tool calls (metrics fetching, configuration updates, etc.)
- **Performance Metrics**: Execution times, token usage, error rates, and response quality

## Prerequisites

1. **Databricks Account**: You need a Databricks workspace with MLflow enabled
   - Sign up for free: https://databricks.com/try-databricks
   - Or use your existing Databricks workspace

2. **Databricks Personal Access Token**: Get your access token
   - Navigate to: User Settings > Developer > Access Tokens
   - Click "Generate New Token"
   - Save the token securely (you'll use this as `DATABRICKS_TOKEN`)
   - Documentation: https://docs.databricks.com/en/dev-tools/auth.html#personal-access-tokens

3. **Python Environment**: The project uses `uv` for dependency management
   - Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Setup Instructions

### 1. Configure Environment Variables

Copy the example environment file and update with your credentials:

```bash
cd /path/to/lambda-autotuner
cp .env.example .env
```

Edit `.env` and set the following MLflow variables:

```bash
# MLflow Tracing Configuration
MLFLOW_TRACKING_URI=databricks
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi34f18c415ed0ac5abfd655d8bd281064
MLFLOW_EXPERIMENT_NAME=lambda-autotuner-agent
MLFLOW_ENABLE_TRACING=true
```

**Important Configuration Options:**

- `MLFLOW_TRACKING_URI`: Set to `databricks` to use Databricks MLflow
- `DATABRICKS_HOST`: Your Databricks workspace URL (e.g., `https://adb-1234567890123456.7.azuredatabricks.net`)
- `DATABRICKS_TOKEN`: Your Databricks personal access token (starts with `dapi`)
- `MLFLOW_EXPERIMENT_NAME`: Name for organizing your runs (default: `lambda-autotuner-agent`)
- `MLFLOW_ENABLE_TRACING`: Set to `true` to enable, `false` to disable

### 2. Install Dependencies

The project includes `mlflow>=2.18.0` in the dependencies. Install all dependencies:

```bash
cd /path/to/self-healing-agent
uv sync
```

### 3. Verify Setup

Check that your configuration is correct:

```bash
# From the lambda-autotuner directory
cd /path/to/lambda-autotuner
source .env
echo $DATABRICKS_HOST  # Should show your workspace URL
echo $DATABRICKS_TOKEN  # Should show your token (first few chars)
```

## Running with MLflow

### Running Synthetic Questions

Run the synthetic questions evaluation with MLflow tracing:

```bash
cd /path/to/lambda-autotuner
uv run python run_synthetic_questions.py
```

You should see:

```
================================================================================
Initializing MLflow Tracing...
================================================================================
✅ MLflow tracing enabled
================================================================================
```

### Running Individual Agent Queries

You can also run the agent directly:

```bash
uv run python agent.py
```

The agent will automatically initialize MLflow tracing at startup.

## What Gets Tracked

### Automatic Tracking

MLflow automatically captures:

1. **LangChain Operations**: All LangChain LLM calls via `mlflow.langchain.autolog()`
2. **Input/Output**: Question inputs and agent responses
3. **Timing**: Execution duration for each operation
4. **Metadata**: Model parameters, token counts, and more

### Custom Tracking

The Lambda Autotuner agent adds custom tracking for:

1. **Question Metadata**:
   - Question ID
   - Category (e.g., "basic_listing", "metrics_analysis")
   - Difficulty level (easy, medium, hard)

2. **Performance Metrics**:
   - Execution time in seconds
   - Response length in characters
   - Success/failure status

3. **Error Information**:
   - Error messages and stack traces
   - Error types and context

## Viewing Data in MLflow

### In Databricks UI

1. Navigate to your Databricks workspace
2. Go to **Machine Learning** → **Experiments**
3. Find your experiment: `lambda-autotuner-agent`
4. Click on individual runs to see:
   - Input questions and agent responses
   - Execution traces with detailed span information
   - Metrics and parameters
   - Error logs

### Trace Visualization

MLflow provides a trace visualization UI showing:

- **Span Hierarchy**: Tree view of all operations
- **Timing Information**: Duration of each span
- **Attributes**: Custom metadata for each operation
- **Inputs/Outputs**: Full content of requests and responses

### Querying Traces

You can programmatically query traces using the MLflow SDK:

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("databricks")

# Get experiment
experiment = mlflow.get_experiment_by_name("lambda-autotuner-agent")

# Search runs
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.success = 'True'",
    order_by=["start_time DESC"],
)

print(runs)
```

## Troubleshooting

### Issue: "MLflow tracing not configured"

**Cause**: Missing or invalid Databricks credentials

**Solution**:
1. Verify `DATABRICKS_HOST` is set in `.env` with your full workspace URL
2. Verify `DATABRICKS_TOKEN` is set with a valid personal access token
3. Check the token hasn't expired (tokens can have expiration dates)
4. Ensure `.env` file is being loaded correctly

```bash
# Check if environment variables are set
echo $DATABRICKS_HOST
echo $DATABRICKS_TOKEN

# If empty, load from .env
export $(grep -v '^#' .env | xargs)
```

### Issue: Authentication Errors (401/403)

**Error Message**: `PERMISSION_DENIED` or `INVALID_PARAMETER_VALUE: Invalid access token`

**Cause**: Invalid or expired Databricks token

**Solution**:
1. Generate a new personal access token in Databricks:
   - User Settings → Developer → Access Tokens → Generate New Token
2. Update your `.env` file with the new token:
   ```bash
   DATABRICKS_TOKEN=dapi_your_new_token_here
   ```
3. Restart your application

### Issue: Cannot Find Experiment

**Error Message**: `Experiment 'lambda-autotuner-agent' not found`

**Cause**: Experiment doesn't exist yet or you don't have access

**Solution**:
- The integration automatically creates the experiment on first run
- If you still see errors, create it manually:
  1. Go to Databricks UI → Machine Learning → Experiments
  2. Click "Create Experiment"
  3. Name it `lambda-autotuner-agent`
  4. Set location to `/Users/your-username/lambda-autotuner-agent`

### Issue: No Traces Appearing

**Possible causes and solutions**:

1. **MLflow Not Initialized**:
   - Check startup logs for "✅ MLflow tracing enabled"
   - If you see "⚠️ MLflow tracing not configured", check credentials

2. **Tracing Disabled**:
   - Verify `MLFLOW_ENABLE_TRACING=true` in `.env`

3. **Network Issues**:
   - Check network connectivity to Databricks
   - Verify no firewall blocking HTTPS to `*.databricks.com`

4. **Experiment Access**:
   - Ensure you have write permissions to the experiment
   - Check with workspace admin if needed

### Issue: Import Error for mlflow_integration

**Cause**: Python can't find the mlflow_integration module

**Solution**:
```bash
# Ensure you're in the correct directory
cd /path/to/lambda-autotuner

# Check if __init__.py exists
ls -la mlflow_integration/__init__.py

# If missing, create it
touch mlflow_integration/__init__.py
```

### Viewing Logs

Enable debug logging to see detailed MLflow instrumentation info:

```bash
# Set debug level in config.yaml
# evaluation:
#   debug_logging: true

# Or set in code
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

### Testing the Connection

Test your MLflow connection with this simple script:

```python
import os
import mlflow

# Set credentials
os.environ["DATABRICKS_HOST"] = "https://your-workspace.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = "dapi_your_token_here"

# Configure MLflow
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("test-experiment")

# Test logging
with mlflow.start_run():
    mlflow.log_param("test", "value")
    print("✅ Successfully connected to MLflow!")
```

## Advanced Configuration

### Custom Span Annotations

Add custom metadata to spans in your code:

```python
from mlflow_integration import create_trace_context

# During execution
with create_trace_context(
    name="custom_operation",
    span_type="TOOL",
    inputs={"param": "value"},
    attributes={
        "custom_tag": "my_value",
        "environment": "production",
    },
):
    # Your code here
    result = do_work()
```

### Using Different Tracking URIs

You can use MLflow with different backends:

```bash
# Databricks (recommended)
MLFLOW_TRACKING_URI=databricks

# Local MLflow server
MLFLOW_TRACKING_URI=http://localhost:5000

# Remote MLflow server
MLFLOW_TRACKING_URI=https://mlflow.yourcompany.com
```

### Filtering and Organizing Runs

Use tags to organize your runs:

```python
import mlflow

mlflow.set_tags({
    "environment": "production",
    "version": "1.0.0",
    "team": "platform",
})
```

Then filter in the UI or via API:

```python
runs = mlflow.search_runs(
    filter_string="tags.environment = 'production'",
)
```

## Additional Resources

- **MLflow Tracing Documentation**: https://mlflow.org/docs/latest/genai/tracing/
- **Databricks MLflow Guide**: https://docs.databricks.com/mlflow/
- **Personal Access Tokens**: https://docs.databricks.com/en/dev-tools/auth.html#personal-access-tokens
- **MLflow Python API**: https://mlflow.org/docs/latest/python_api/

## Integration Benefits

The MLflow integration provides:

1. **Comprehensive Visibility**: Full trace of agent execution from question to response
2. **Performance Insights**: Detailed timing and resource usage metrics
3. **Error Tracking**: Complete error context for debugging
4. **Experiment Tracking**: Compare different configurations and models
5. **Production Monitoring**: Track agent behavior in production
6. **Team Collaboration**: Share traces and insights with your team via Databricks

The integration is fully automatic and requires minimal configuration. All agent operations are traced automatically, with custom spans for Lambda-specific operations.
