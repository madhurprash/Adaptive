"""
Langfuse MCP Server for monitoring agent execution and retrieving trace data.

This MCP server exposes Langfuse API tools for analyzing execution traces,
extracting insights, and monitoring performance metrics from Langfuse projects.
"""

import os
import json
import logging
import httpx
from datetime import datetime, timedelta
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Constants
LANGFUSE_API_BASE_URL: str = "https://cloud.langfuse.com"
LANGFUSE_US_BASE_URL: str = "https://us.cloud.langfuse.com"
LANGFUSE_HIPAA_BASE_URL: str = "https://hipaa.cloud.langfuse.com"
DEFAULT_LIMIT: int = 50


class LangfuseConfig(BaseModel):
    """Configuration for Langfuse API access."""

    public_key: str = Field(..., description="Langfuse public API key")
    secret_key: str = Field(..., description="Langfuse secret API key")
    base_url: str = Field(
        default=LANGFUSE_API_BASE_URL, description="Langfuse API base URL"
    )

    @classmethod
    def from_env(cls) -> "LangfuseConfig":
        """Create configuration from environment variables."""
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        base_url = os.getenv("LANGFUSE_HOST", LANGFUSE_API_BASE_URL)

        if not public_key:
            raise ValueError(
                "LANGFUSE_PUBLIC_KEY environment variable must be set. "
                "Get your API key from your Langfuse project settings"
            )
        if not secret_key:
            raise ValueError(
                "LANGFUSE_SECRET_KEY environment variable must be set. "
                "Get your API key from your Langfuse project settings"
            )
        return cls(public_key=public_key, secret_key=secret_key, base_url=base_url)


async def _make_request(
    endpoint: str,
    config: LangfuseConfig,
    method: str = "GET",
    params: Optional[dict[str, Any]] = None,
    json_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Make an authenticated request to the Langfuse API."""
    url = f"{config.base_url}{endpoint}"
    headers = {"Content-Type": "application/json"}

    logger.debug(f"Making {method} request to {url}")

    async with httpx.AsyncClient() as client:
        response = await client.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_data,
            auth=(config.public_key, config.secret_key),  # Basic Auth
        )
        response.raise_for_status()

        if response.content:
            return response.json()
        return {}


# Initialize MCP server
mcp = FastMCP("Langfuse")


# =============================================================================
# PROJECT MANAGEMENT TOOLS
# =============================================================================


@mcp.tool()
async def list_projects() -> list[dict[str, Any]]:
    """
    List all Langfuse projects accessible with the current API key.

    Returns:
        List of project dictionaries with id, name, created_at, updated_at
    """
    try:
        logger.info("Listing all Langfuse projects...")
        config = LangfuseConfig.from_env()
        response = await _make_request("/api/public/projects", config)
        # Extract the 'data' field from the response if it exists
        if isinstance(response, dict) and 'data' in response:
            return response['data']
        return response if isinstance(response, list) else []
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        raise


# =============================================================================
# TRACE MANAGEMENT TOOLS
# =============================================================================


@mcp.tool()
async def list_traces(
    page: int = 1,
    limit: int = DEFAULT_LIMIT,
    user_id: Optional[str] = None,
    name: Optional[str] = None,
    session_id: Optional[str] = None,
    from_timestamp: Optional[str] = None,
    to_timestamp: Optional[str] = None,
) -> dict[str, Any]:
    """
    List traces with optional filtering and pagination.

    Args:
        page: Page number for pagination (default: 1)
        limit: Number of traces per page (default: 50, max: 100)
        user_id: Filter by user ID
        name: Filter by trace name
        session_id: Filter by session ID
        from_timestamp: Filter traces after this timestamp (ISO 8601)
        to_timestamp: Filter traces before this timestamp (ISO 8601)

    Returns:
        Dictionary with 'data' (list of traces) and 'meta' (pagination info)
    """
    try:
        logger.info("Listing traces...")
        config = LangfuseConfig.from_env()

        params: dict[str, Any] = {"page": page, "limit": min(limit, 100)}

        if user_id:
            params["userId"] = user_id
        if name:
            params["name"] = name
        if session_id:
            params["sessionId"] = session_id
        if from_timestamp:
            params["fromTimestamp"] = from_timestamp
        if to_timestamp:
            params["toTimestamp"] = to_timestamp

        logger.debug(f"Fetching traces with params: {params}")
        return await _make_request("/api/public/traces", config, params=params)
    except Exception as e:
        logger.error(f"Error listing traces: {e}")
        raise


@mcp.tool()
async def get_trace(trace_id: str) -> dict[str, Any]:
    """
    Get detailed information about a specific trace.

    Args:
        trace_id: UUID of the trace

    Returns:
        Dictionary containing trace details, observations, and metadata
    """
    try:
        logger.info(f"Fetching trace: {trace_id}")
        config = LangfuseConfig.from_env()
        return await _make_request(f"/api/public/traces/{trace_id}", config)
    except Exception as e:
        logger.error(f"Error fetching trace {trace_id}: {e}")
        raise


# =============================================================================
# OBSERVATION TOOLS (SPANS, EVENTS, GENERATIONS)
# =============================================================================


@mcp.tool()
async def list_observations(
    page: int = 1,
    limit: int = DEFAULT_LIMIT,
    trace_id: Optional[str] = None,
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    observation_type: Optional[str] = None,
    from_start_time: Optional[str] = None,
    to_start_time: Optional[str] = None,
) -> dict[str, Any]:
    """
    List observations (spans, events, generations) with filtering.

    Args:
        page: Page number for pagination (default: 1)
        limit: Number of observations per page (default: 50, max: 100)
        trace_id: Filter by trace ID
        name: Filter by observation name
        user_id: Filter by user ID
        observation_type: Filter by type (SPAN, EVENT, GENERATION)
        from_start_time: Filter observations after this time (ISO 8601)
        to_start_time: Filter observations before this time (ISO 8601)

    Returns:
        Dictionary with 'data' (list of observations) and 'meta' (pagination info)
    """
    try:
        logger.info("Listing observations...")
        config = LangfuseConfig.from_env()

        params: dict[str, Any] = {"page": page, "limit": min(limit, 100)}

        if trace_id:
            params["traceId"] = trace_id
        if name:
            params["name"] = name
        if user_id:
            params["userId"] = user_id
        if observation_type:
            params["type"] = observation_type
        if from_start_time:
            params["fromStartTime"] = from_start_time
        if to_start_time:
            params["toStartTime"] = to_start_time

        logger.debug(f"Fetching observations with params: {params}")
        return await _make_request("/api/public/observations", config, params=params)
    except Exception as e:
        logger.error(f"Error listing observations: {e}")
        raise


@mcp.tool()
async def get_observation(observation_id: str) -> dict[str, Any]:
    """
    Get detailed information about a specific observation.

    Args:
        observation_id: UUID of the observation

    Returns:
        Dictionary containing observation details (type, input, output, metadata, etc.)
    """
    try:
        logger.info(f"Fetching observation: {observation_id}")
        config = LangfuseConfig.from_env()
        return await _make_request(f"/api/public/observations/{observation_id}", config)
    except Exception as e:
        logger.error(f"Error fetching observation {observation_id}: {e}")
        raise


@mcp.tool()
async def list_generations(
    page: int = 1,
    limit: int = DEFAULT_LIMIT,
    trace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    name: Optional[str] = None,
    from_start_time: Optional[str] = None,
    to_start_time: Optional[str] = None,
) -> dict[str, Any]:
    """
    List LLM generations with filtering.

    Args:
        page: Page number for pagination (default: 1)
        limit: Number of generations per page (default: 50, max: 100)
        trace_id: Filter by trace ID
        user_id: Filter by user ID
        name: Filter by generation name
        from_start_time: Filter generations after this time (ISO 8601)
        to_start_time: Filter generations before this time (ISO 8601)

    Returns:
        Dictionary with 'data' (list of generations with token usage) and 'meta'
    """
    try:
        logger.info("Listing generations...")
        config = LangfuseConfig.from_env()

        params: dict[str, Any] = {"page": page, "limit": min(limit, 100)}

        if trace_id:
            params["traceId"] = trace_id
        if user_id:
            params["userId"] = user_id
        if name:
            params["name"] = name
        if from_start_time:
            params["fromStartTime"] = from_start_time
        if to_start_time:
            params["toStartTime"] = to_start_time

        logger.debug(f"Fetching generations with params: {params}")
        return await _make_request("/api/public/observations", config, params=params)
    except Exception as e:
        logger.error(f"Error listing generations: {e}")
        raise


# =============================================================================
# SESSION TOOLS
# =============================================================================


@mcp.tool()
async def list_sessions(
    page: int = 1,
    limit: int = DEFAULT_LIMIT,
    from_timestamp: Optional[str] = None,
    to_timestamp: Optional[str] = None,
) -> dict[str, Any]:
    """
    List all sessions with optional time filtering.

    Args:
        page: Page number for pagination (default: 1)
        limit: Number of sessions per page (default: 50, max: 100)
        from_timestamp: Filter sessions after this timestamp (ISO 8601)
        to_timestamp: Filter sessions before this timestamp (ISO 8601)

    Returns:
        Dictionary with 'data' (list of sessions) and 'meta' (pagination info)
    """
    try:
        logger.info("Listing sessions...")
        config = LangfuseConfig.from_env()

        params: dict[str, Any] = {"page": page, "limit": min(limit, 100)}

        if from_timestamp:
            params["fromTimestamp"] = from_timestamp
        if to_timestamp:
            params["toTimestamp"] = to_timestamp

        logger.debug(f"Fetching sessions with params: {params}")
        return await _make_request("/api/public/sessions", config, params=params)
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise


@mcp.tool()
async def get_session(session_id: str) -> dict[str, Any]:
    """
    Get detailed information about a specific session.

    Args:
        session_id: Session ID

    Returns:
        Dictionary containing session details and list of traces
    """
    try:
        logger.info(f"Fetching session: {session_id}")
        config = LangfuseConfig.from_env()
        return await _make_request(f"/api/public/sessions/{session_id}", config)
    except Exception as e:
        logger.error(f"Error fetching session {session_id}: {e}")
        raise


# =============================================================================
# SCORE TOOLS (EVALUATION & FEEDBACK)
# =============================================================================


@mcp.tool()
async def list_scores(
    page: int = 1,
    limit: int = DEFAULT_LIMIT,
    trace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    name: Optional[str] = None,
    from_timestamp: Optional[str] = None,
    to_timestamp: Optional[str] = None,
) -> dict[str, Any]:
    """
    List scores (evaluations and feedback) with filtering.

    Args:
        page: Page number for pagination (default: 1)
        limit: Number of scores per page (default: 50, max: 100)
        trace_id: Filter by trace ID
        user_id: Filter by user ID
        name: Filter by score name
        from_timestamp: Filter scores after this timestamp (ISO 8601)
        to_timestamp: Filter scores before this timestamp (ISO 8601)

    Returns:
        Dictionary with 'data' (list of scores) and 'meta' (pagination info)
    """
    try:
        logger.info("Listing scores...")
        config = LangfuseConfig.from_env()

        params: dict[str, Any] = {"page": page, "limit": min(limit, 100)}

        if trace_id:
            params["traceId"] = trace_id
        if user_id:
            params["userId"] = user_id
        if name:
            params["name"] = name
        if from_timestamp:
            params["fromTimestamp"] = from_timestamp
        if to_timestamp:
            params["toTimestamp"] = to_timestamp

        logger.debug(f"Fetching scores with params: {params}")
        return await _make_request("/api/public/scores", config, params=params)
    except Exception as e:
        logger.error(f"Error listing scores: {e}")
        raise


@mcp.tool()
async def get_score(score_id: str) -> dict[str, Any]:
    """
    Get detailed information about a specific score.

    Args:
        score_id: UUID of the score

    Returns:
        Dictionary containing score details (name, value, data_type, etc.)
    """
    try:
        logger.info(f"Fetching score: {score_id}")
        config = LangfuseConfig.from_env()
        return await _make_request(f"/api/public/scores/{score_id}", config)
    except Exception as e:
        logger.error(f"Error fetching score {score_id}: {e}")
        raise


@mcp.tool()
async def create_score(
    trace_id: str,
    name: str,
    value: float | str | bool,
    observation_id: Optional[str] = None,
    comment: Optional[str] = None,
    data_type: str = "NUMERIC",
) -> dict[str, Any]:
    """
    Create a new score for a trace or observation.

    Args:
        trace_id: UUID of the trace to score
        name: Score name (e.g., "accuracy", "relevance")
        value: Score value (number, string, or boolean)
        observation_id: Optional observation ID to score specific observation
        comment: Optional comment explaining the score
        data_type: Score data type (NUMERIC, CATEGORICAL, BOOLEAN)

    Returns:
        Dictionary containing the created score
    """
    try:
        logger.info(f"Creating score for trace: {trace_id}")
        config = LangfuseConfig.from_env()

        score_data = {
            "traceId": trace_id,
            "name": name,
            "value": value,
            "dataType": data_type,
        }

        if observation_id:
            score_data["observationId"] = observation_id
        if comment:
            score_data["comment"] = comment

        return await _make_request("/api/public/scores", config, method="POST", json_data=score_data)
    except Exception as e:
        logger.error(f"Error creating score: {e}")
        raise


# =============================================================================
# DATASET TOOLS
# =============================================================================


@mcp.tool()
async def list_datasets(
    page: int = 1,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """
    List all datasets in the current project.

    Args:
        page: Page number for pagination (default: 1)
        limit: Number of datasets per page (default: 50, max: 100)

    Returns:
        Dictionary with 'data' (list of datasets) and 'meta' (pagination info)
    """
    try:
        logger.info("Listing datasets...")
        config = LangfuseConfig.from_env()

        params = {"page": page, "limit": min(limit, 100)}
        return await _make_request("/api/public/v2/datasets", config, params=params)
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise


@mcp.tool()
async def get_dataset(dataset_name: str) -> dict[str, Any]:
    """
    Get detailed information about a specific dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dictionary containing dataset details (id, name, description, metadata, etc.)
    """
    try:
        logger.info(f"Fetching dataset: {dataset_name}")
        config = LangfuseConfig.from_env()
        return await _make_request(f"/api/public/v2/datasets/{dataset_name}", config)
    except Exception as e:
        logger.error(f"Error fetching dataset {dataset_name}: {e}")
        raise


@mcp.tool()
async def create_dataset(
    name: str,
    description: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Create a new dataset.

    Args:
        name: Dataset name
        description: Optional dataset description
        metadata: Optional custom metadata

    Returns:
        Dictionary containing the created dataset
    """
    try:
        logger.info(f"Creating dataset: {name}")
        config = LangfuseConfig.from_env()

        dataset_data = {"name": name}
        if description:
            dataset_data["description"] = description
        if metadata:
            dataset_data["metadata"] = metadata

        return await _make_request("/api/public/v2/datasets", config, method="POST", json_data=dataset_data)
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise


@mcp.tool()
async def list_dataset_items(
    dataset_name: str,
    page: int = 1,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """
    List all items in a dataset.

    Args:
        dataset_name: Name of the dataset
        page: Page number for pagination (default: 1)
        limit: Number of items per page (default: 50, max: 100)

    Returns:
        Dictionary with 'data' (list of dataset items) and 'meta' (pagination info)
    """
    try:
        logger.info(f"Listing items for dataset: {dataset_name}")
        config = LangfuseConfig.from_env()

        params = {"page": page, "limit": min(limit, 100)}
        return await _make_request(
            f"/api/public/v2/datasets/{dataset_name}/items", config, params=params
        )
    except Exception as e:
        logger.error(f"Error listing dataset items: {e}")
        raise


@mcp.tool()
async def create_dataset_item(
    dataset_name: str,
    input_data: dict[str, Any],
    expected_output: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Create a new item in a dataset.

    Args:
        dataset_name: Name of the dataset
        input_data: Input data for the test case
        expected_output: Expected output data
        metadata: Optional custom metadata

    Returns:
        Dictionary containing the created dataset item
    """
    try:
        logger.info(f"Creating item for dataset: {dataset_name}")
        config = LangfuseConfig.from_env()

        item_data = {"input": input_data}
        if expected_output:
            item_data["expectedOutput"] = expected_output
        if metadata:
            item_data["metadata"] = metadata

        return await _make_request(
            f"/api/public/v2/datasets/{dataset_name}/items",
            config,
            method="POST",
            json_data=item_data,
        )
    except Exception as e:
        logger.error(f"Error creating dataset item: {e}")
        raise


@mcp.tool()
async def get_dataset_runs(
    dataset_name: str,
    page: int = 1,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """
    List all runs (evaluations) for a dataset.

    Args:
        dataset_name: Name of the dataset
        page: Page number for pagination (default: 1)
        limit: Number of runs per page (default: 50, max: 100)

    Returns:
        Dictionary with 'data' (list of dataset runs) and 'meta' (pagination info)
    """
    try:
        logger.info(f"Fetching runs for dataset: {dataset_name}")
        config = LangfuseConfig.from_env()

        params = {"page": page, "limit": min(limit, 100)}
        return await _make_request(
            f"/api/public/v2/datasets/{dataset_name}/runs", config, params=params
        )
    except Exception as e:
        logger.error(f"Error fetching dataset runs: {e}")
        raise


# =============================================================================
# PROMPT MANAGEMENT TOOLS
# =============================================================================


@mcp.tool()
async def list_prompts(
    page: int = 1,
    limit: int = DEFAULT_LIMIT,
    name: Optional[str] = None,
    label: Optional[str] = None,
    tag: Optional[str] = None,
) -> dict[str, Any]:
    """
    List all prompts with optional filtering.

    Args:
        page: Page number for pagination (default: 1)
        limit: Number of prompts per page (default: 50, max: 100)
        name: Filter by prompt name
        label: Filter by label (e.g., "production", "staging")
        tag: Filter by tag

    Returns:
        Dictionary with 'data' (list of prompts with versions) and 'meta'
    """
    try:
        logger.info("Listing prompts...")
        config = LangfuseConfig.from_env()

        params: dict[str, Any] = {"page": page, "limit": min(limit, 100)}

        if name:
            params["name"] = name
        if label:
            params["label"] = label
        if tag:
            params["tag"] = tag

        logger.debug(f"Fetching prompts with params: {params}")
        return await _make_request("/api/public/v2/prompts", config, params=params)
    except Exception as e:
        logger.error(f"Error listing prompts: {e}")
        raise


@mcp.tool()
async def get_prompt(
    prompt_name: str,
    version: Optional[int] = None,
    label: Optional[str] = None,
) -> dict[str, Any]:
    """
    Get a specific prompt by name and version or label.

    Args:
        prompt_name: Name of the prompt
        version: Specific version number (optional)
        label: Label to fetch (e.g., "production") - alternative to version

    Returns:
        Dictionary containing prompt details (name, version, prompt, config, etc.)
    """
    try:
        logger.info(f"Fetching prompt: {prompt_name}")
        config = LangfuseConfig.from_env()

        params: dict[str, Any] = {}
        if version is not None:
            params["version"] = version
        if label:
            params["label"] = label

        logger.debug(f"Fetching prompt with params: {params}")
        return await _make_request(
            f"/api/public/v2/prompts/{prompt_name}", config, params=params
        )
    except Exception as e:
        logger.error(f"Error fetching prompt {prompt_name}: {e}")
        raise


@mcp.tool()
async def create_prompt(
    name: str,
    prompt: str,
    prompt_config: Optional[dict[str, Any]] = None,
    labels: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Create a new prompt or new version of an existing prompt.

    Args:
        name: Prompt name
        prompt: Prompt template/content
        prompt_config: Optional configuration (model, temperature, etc.)
        labels: Optional labels (e.g., ["production", "latest"])
        tags: Optional tags for categorization

    Returns:
        Dictionary containing the created prompt
    """
    try:
        logger.info(f"Creating prompt: {name}")
        config = LangfuseConfig.from_env()

        prompt_data: dict[str, Any] = {
            "name": name,
            "prompt": prompt,
        }

        if prompt_config:
            prompt_data["config"] = prompt_config
        if labels:
            prompt_data["labels"] = labels
        if tags:
            prompt_data["tags"] = tags

        return await _make_request("/api/public/v2/prompts", config, method="POST", json_data=prompt_data)
    except Exception as e:
        logger.error(f"Error creating prompt: {e}")
        raise


# =============================================================================
# METRICS & ANALYTICS TOOLS
# =============================================================================


@mcp.tool()
async def get_metrics(
    view: str = "traces",
    metrics: Optional[list[dict[str, str]]] = None,
    from_timestamp: Optional[str] = None,
    to_timestamp: Optional[str] = None,
    trace_name: Optional[str] = None,
    user_id: Optional[str] = None,
    dimensions: Optional[list[dict[str, str]]] = None,
    granularity: Optional[str] = None,
) -> dict[str, Any]:
    """
    Get aggregated metrics for traces with optional filtering.

    This endpoint requires a complex query structure. By default, it returns
    basic trace count metrics for the last 7 days.

    Args:
        view: Data view to query. Options: "traces", "observations", "scores-numeric", "scores-categorical" (default: "traces")
        metrics: List of metrics to compute. Each metric needs 'measure' and 'aggregation'.
                 Example: [{"measure": "traces", "aggregation": "count"}]
                 If not provided, defaults to trace count.
        from_timestamp: Start of time range (ISO 8601). Defaults to 7 days ago.
        to_timestamp: End of time range (ISO 8601). Defaults to now.
        trace_name: Filter by trace name
        user_id: Filter by user ID
        dimensions: List of dimension fields to group by. Example: [{"field": "name"}]
        granularity: Time dimension granularity. Options: "hour", "day", "week", "month"

    Returns:
        Dictionary containing aggregated metrics based on the query

    Example:
        # Get trace count for last 7 days
        get_metrics()

        # Get trace count by name
        get_metrics(dimensions=[{"field": "name"}])

        # Get observation latency metrics
        get_metrics(
            view="observations",
            metrics=[{"measure": "latency", "aggregation": "avg"}]
        )
    """
    try:
        logger.info("Fetching metrics...")
        config = LangfuseConfig.from_env()

        # Set default time range if not provided (last 7 days)
        if not to_timestamp:
            to_timestamp = datetime.utcnow().isoformat() + "Z"
        if not from_timestamp:
            from_dt = datetime.utcnow() - timedelta(days=7)
            from_timestamp = from_dt.isoformat() + "Z"

        # Set default metrics if not provided
        if not metrics:
            metrics = [{"measure": "traces", "aggregation": "count"}]

        # Build the query object
        query: dict[str, Any] = {
            "view": view,
            "metrics": metrics,
            "fromTimestamp": from_timestamp,
            "toTimestamp": to_timestamp,
        }

        # Add optional dimensions
        if dimensions:
            query["dimensions"] = dimensions

        # Add optional time granularity
        if granularity:
            query["timeDimension"] = {"granularity": granularity}

        # Add filters if provided
        filters = []
        if trace_name:
            filters.append({
                "column": "name",
                "operator": "=",
                "value": trace_name,
                "type": "string"
            })
        if user_id:
            filters.append({
                "column": "userId",
                "operator": "=",
                "value": user_id,
                "type": "string"
            })

        if filters:
            query["filters"] = filters

        # Convert query to JSON string
        query_json = json.dumps(query)

        logger.debug(f"Fetching metrics with query: {query_json}")
        return await _make_request(
            "/api/public/metrics",
            config,
            params={"query": query_json}
        )
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise


# =============================================================================
# MODEL MANAGEMENT TOOLS
# =============================================================================


@mcp.tool()
async def list_models(
    page: int = 1,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """
    List all models configured in the project.

    Args:
        page: Page number for pagination (default: 1)
        limit: Number of models per page (default: 50, max: 100)

    Returns:
        Dictionary with 'data' (list of models with pricing) and 'meta'
    """
    try:
        logger.info("Listing models...")
        config = LangfuseConfig.from_env()

        params = {"page": page, "limit": min(limit, 100)}
        return await _make_request("/api/public/models", config, params=params)
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise


# =============================================================================
# COMMENT TOOLS
# =============================================================================


@mcp.tool()
async def create_comment(
    trace_id: str,
    content: str,
    observation_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create a comment on a trace or observation.

    Args:
        trace_id: UUID of the trace
        content: Comment text
        observation_id: Optional observation ID to comment on specific observation

    Returns:
        Dictionary containing the created comment
    """
    try:
        logger.info(f"Creating comment for trace: {trace_id}")
        config = LangfuseConfig.from_env()

        comment_data = {
            "traceId": trace_id,
            "content": content,
        }

        if observation_id:
            comment_data["observationId"] = observation_id

        return await _make_request("/api/public/comments", config, method="POST", json_data=comment_data)
    except Exception as e:
        logger.error(f"Error creating comment: {e}")
        raise


# =============================================================================
# HEALTH CHECK
# =============================================================================


@mcp.tool()
async def health_check() -> dict[str, Any]:
    """
    Check the health status of the Langfuse API.

    Returns:
        Dictionary containing health status
    """
    try:
        logger.info("Checking API health...")
        config = LangfuseConfig.from_env()
        return await _make_request("/api/public/health", config)
    except Exception as e:
        logger.error(f"Error checking health: {e}")
        raise


if __name__ == "__main__":
    mcp.run(transport="stdio")
