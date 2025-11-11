"""
LangSmith API tools for monitoring agent execution and retrieving trace data.

These tools enable the self-healing agent to analyze execution traces, extract insights,
and monitor performance metrics from LangSmith projects.
"""

import os
import re
import logging
import requests
from typing import Any, Optional
from datetime import datetime
from langchain_core.tools import tool

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Constants
LANGSMITH_API_BASE_URL: str = "https://api.smith.langchain.com"
DEFAULT_LIMIT: int = 100

class LangSmithConfig(BaseModel):
    """Configuration for LangSmith API access."""

    api_key: str = Field(..., description="LangSmith API key")
    base_url: str = Field(
        default=LANGSMITH_API_BASE_URL, description="LangSmith API base URL"
    )

    @classmethod
    def from_env(cls) -> "LangSmithConfig":
        """Create configuration from environment variables."""
        api_key = os.getenv("LANGSMITH_API_KEY")
        if not api_key:
            raise ValueError(
                "LANGSMITH_API_KEY environment variable must be set. "
                "Get your API key from https://smith.langchain.com/settings"
            )
        return cls(api_key=api_key)


def _make_request(
    endpoint: str,
    config: LangSmithConfig,
    method: str = "GET",
    params: Optional[dict[str, Any]] = None,
    json_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Make an authenticated request to the LangSmith API.

    Args:
        endpoint: API endpoint path (e.g., "/api/v1/sessions")
        config: LangSmith configuration with API key
        method: HTTP method (GET, POST, etc.)
        params: Query parameters
        json_data: JSON body for POST requests

    Returns:
        Response data as dictionary

    Raises:
        requests.HTTPError: If the request fails
    """
    url = f"{config.base_url}{endpoint}"
    headers = {"x-api-key": config.api_key, "Content-Type": "application/json"}
    print(f"Making {method} request to {url}")
    response = requests.request(
        method=method, url=url, headers=headers, params=params, json=json_data
    )
    response.raise_for_status()
    return response.json()


def _resolve_session_id(
    session_identifier: str,
    config: LangSmithConfig,
) -> str:
    """
    Resolve a session name or UUID to a valid session UUID.

    If the identifier looks like a UUID, return it as-is.
    Otherwise, search for a project with that name and return its UUID.

    Args:
        session_identifier: Either a session UUID or project name
        config: LangSmith configuration

    Returns:
        The session UUID

    Raises:
        ValueError: If the session name is not found
    """
    # Check if it looks like a UUID (contains hyphens and hex characters)
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)

    if uuid_pattern.match(session_identifier):
        logger.debug(f"'{session_identifier}' appears to be a UUID, using directly")
        return session_identifier

    # It's a name, so we need to resolve it to a UUID
    logger.info(f"Resolving project name '{session_identifier}' to UUID...")

    try:
        # List sessions to find the matching name
        response = _make_request("/api/v1/sessions", config, params={"limit": 100})

        # Response could be a list or a dict with a sessions key
        sessions = response if isinstance(response, list) else response.get("sessions", [])

        for session in sessions:
            if session.get("name") == session_identifier:
                session_uuid = session.get("id")
                logger.info(f"Resolved '{session_identifier}' to UUID: {session_uuid}")
                return session_uuid

        # If we didn't find it, raise an error with helpful message
        available_names = [s.get("name") for s in sessions if s.get("name")]
        raise ValueError(
            f"Project '{session_identifier}' not found. "
            f"Available projects: {', '.join(available_names[:10])}"
        )

    except Exception as e:
        logger.error(f"Error resolving session name to UUID: {e}")
        raise ValueError(
            f"Unable to resolve project name '{session_identifier}'. "
            f"Please check the name or provide the project UUID directly. Error: {e}"
        )

@tool
def get_session_info(
    session_id: str,
    config: Optional[LangSmithConfig] = None,
    include_stats: bool = True,
) -> dict[str, Any]:
    """
    Get detailed information about a specific LangSmith session/project.

    This tool retrieves comprehensive information about a tracing session, including
    metadata, run counts, and execution statistics. Use this to understand the scope
    and current state of an agent execution project.

    Args:
        session_id: UUID or name of the LangSmith session/project
        config: LangSmith configuration (uses env vars if not provided)
        include_stats: Whether to include run statistics

    Returns:
        Dictionary containing session details:
        - id: Session UUID
        - name: Session name
        - description: Session description
        - created_at: Creation timestamp
        - run_count: Total number of runs
        - latency_p50/p99: Latency percentiles (if include_stats=True)
        - error_rate: Error rate percentage (if include_stats=True)

    Example:
        >>> config = LangSmithConfig.from_env()
        >>> info = get_session_info("my-agent-project", config)
        >>> print(f"Session has {info['run_count']} runs")
    """
    try:
        print(f"In the GET_SESSION_INFO tool, going to get information about specific session/project...")
        if config is None:
            config = LangSmithConfig.from_env()

        # Resolve the session name to UUID if needed
        resolved_session_id = _resolve_session_id(session_id, config)

        params = {"include_stats": str(include_stats).lower()}
        print(f"Fetching info for session: {session_id} (resolved to: {resolved_session_id})")

        try:
            return _make_request(f"/api/v1/sessions/{resolved_session_id}", config, params=params)
        except requests.exceptions.HTTPError as e:
            # If we get a 500 error with include_stats=true, retry without stats
            if e.response.status_code == 500 and include_stats:
                logger.warning(
                    f"Failed to get session info with stats (500 error), retrying without stats. "
                    f"Session: {session_id}"
                )
                params = {"include_stats": "false"}
                return _make_request(f"/api/v1/sessions/{resolved_session_id}", config, params=params)
            else:
                raise
    except Exception as e:
        logger.error(f"An error occurred while retrieving the information about the session or project: {e}")
        raise

@tool
def list_sessions(
    config: Optional[LangSmithConfig] = None,
    name_contains: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    include_stats: bool = False,
) -> list[dict[str, Any]]:
    """
    List all LangSmith sessions/projects with optional filtering.

    This tool retrieves a list of all tracing sessions, useful for discovering
    which projects are available for analysis. Filter by name to find specific
    agent projects.

    Args:
        config: LangSmith configuration (uses env vars if not provided)
        name_contains: Filter sessions by name substring (case-insensitive)
        limit: Maximum number of sessions to return (default: 100)
        offset: Number of sessions to skip for pagination
        include_stats: Whether to include statistics for each session

    Returns:
        List of session dictionaries, each containing:
        - id: Session UUID
        - name: Session name
        - created_at: Creation timestamp
        - run_count: Number of runs (if include_stats=True)

    Example:
        >>> config = LangSmithConfig.from_env()
        >>> sessions = list_sessions(config, name_contains="production")
        >>> for session in sessions:
        ...     print(f"{session['name']}: {session['run_count']} runs")
    """
    try:
        print(f"In the LIST_SESSIONS tool, going to list the sessions...")
        if config is None:
            config = LangSmithConfig.from_env()
        params = {
            "limit": limit,
            "offset": offset,
            "include_stats": str(include_stats).lower(),
        }
        if name_contains:
            params["name_contains"] = name_contains

        print(f"Listing sessions with filters: {params}")
        return _make_request("/api/v1/sessions", config, params=params)
    except Exception as e:
        logger.error(f"An error occurred while listing the sessions: {e}")
        raise

@tool
def get_session_metadata(
    session_id: str,
    config: Optional[LangSmithConfig] = None,
    k: int = 10,
    root_runs_only: bool = True,
) -> dict[str, list[str]]:
    """
    Get top metadata keys and values from runs in a session.

    This tool extracts the most common metadata from runs, helping you understand
    what contextual information is being tracked (e.g., user IDs, environment,
    model versions). Essential for identifying patterns in agent execution.

    Args:
        session_id: UUID or name of the LangSmith session
        config: LangSmith configuration (uses env vars if not provided)
        k: Number of top values to return per metadata key (default: 10)
        root_runs_only: Only analyze root-level runs (default: True)

    Returns:
        Dictionary mapping metadata keys to lists of top values:
        - environment: ["production", "staging"]
        - user_id: ["user_123", "user_456"]
        - model: ["gpt-4", "claude-3"]

    Example:
        >>> config = LangSmithConfig.from_env()
        >>> metadata = get_session_metadata("my-agent-project", config)
        >>> print(f"Top environments: {metadata.get('environment', [])}")
    """
    try:
        if config is None:
            config = LangSmithConfig.from_env()

        # Resolve the session name to UUID if needed
        resolved_session_id = _resolve_session_id(session_id, config)
        print(f"In the GET SESSION METADATA TOOL, going to get the metadata of the provided session id...")
        params = {"k": k, "root_runs_only": str(root_runs_only).lower()}
        print(f"Fetching metadata for session: {session_id} (resolved to: {resolved_session_id})")
        return _make_request(
            f"/api/v1/sessions/{resolved_session_id}/metadata", config, params=params
        )
    except Exception as e:
        logger.error(f"An error occurred while getting the session metadata: {e}")
        raise

@tool
def get_runs_from_session(
    session_id: str,
    config: Optional[LangSmithConfig] = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    filter_query: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Query and retrieve runs from a specific session.

    This tool fetches detailed run data including inputs, outputs, errors, and
    execution traces. Use this to analyze individual agent executions and identify
    failure patterns or performance bottlenecks.

    Args:
        session_id: UUID or name of the LangSmith session
        config: LangSmith configuration (uses env vars if not provided)
        limit: Maximum number of runs to return (default: 100)
        offset: Number of runs to skip for pagination
        filter_query: LangSmith filter query (e.g., 'eq(status, "error")')

    Returns:
        List of run dictionaries, each containing:
        - id: Run UUID
        - name: Run name/type
        - inputs: Input data passed to the agent
        - outputs: Output data from the agent
        - error: Error message if run failed
        - start_time: Execution start timestamp
        - end_time: Execution end timestamp
        - latency: Execution duration in seconds
        - metadata: Associated metadata
        - feedback_stats: Feedback scores

    Example:
        >>> config = LangSmithConfig.from_env()
        >>> # Get failed runs
        >>> runs = get_runs_from_session(
        ...     "my-agent-project",
        ...     config,
        ...     filter_query='eq(status, "error")'
        ... )
        >>> for run in runs:
        ...     print(f"Error: {run.get('error')}")
    """
    try:
        print(f"Going to GET THE RUNS FROM A SPECIFIC SESSION...")
        if config is None:
            config = LangSmithConfig.from_env()

        # Resolve the session name to UUID if needed
        resolved_session_id = _resolve_session_id(session_id, config)

        params = {
            "session": [resolved_session_id],
            "limit": limit,
            "offset": offset,
        }
        if filter_query:
            params["filter"] = filter_query
        print(f"Fetching runs for session: {session_id} (resolved to: {resolved_session_id})")
        runs_from_session: dict = _make_request("/api/v1/runs/query", config, method="POST", json_data=params)
        print(f"Retrieved the RUNS FROM THE PROVIDED SESSION ID: {runs_from_session}")
        return runs_from_session
    except Exception as e:
        logger.error(f"An error occurred while getting the runs from the session: {e}")
        raise e


@tool
def list_session_runs_summary(
    session_id: str,
    config: Optional[LangSmithConfig] = None,
    limit: int = 20,
    offset: int = 0,
    filter_query: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    List runs from a session with lightweight metadata only (no full inputs/outputs).

    This tool returns a summary of runs to avoid context window overflow. Use this
    first to see available runs, then use get_run_details() for specific runs.

    Args:
        session_id: UUID or name of the LangSmith session
        config: LangSmith configuration (uses env vars if not provided)
        limit: Maximum number of runs to return (default: 20, max: 100)
        offset: Number of runs to skip for pagination
        filter_query: LangSmith filter query (e.g., 'eq(status, "error")')

    Returns:
        List of lightweight run summaries, each containing:
        - id: Run UUID
        - name: Run name/type
        - status: Run status (success, error, etc.)
        - start_time: Execution start timestamp
        - end_time: Execution end timestamp
        - latency: Execution duration in milliseconds
        - error: Error message (truncated to 200 chars if present)
        - run_type: Type of run (chain, llm, tool, etc.)

    Example:
        >>> config = LangSmithConfig.from_env()
        >>> runs = list_session_runs_summary("my-agent-project", config, limit=10)
        >>> for run in runs:
        ...     print(f"{run['name']}: {run['status']} ({run['latency']}ms)")
    """
    try:
        print(f"Listing runs summary for session (lightweight, no full data)...")
        if config is None:
            config = LangSmithConfig.from_env()

        # Resolve the session name to UUID if needed
        resolved_session_id = _resolve_session_id(session_id, config)

        # Limit to max 100 to prevent context overflow
        actual_limit = min(limit, 100)

        params = {
            "session": [resolved_session_id],
            "limit": actual_limit,
            "offset": offset,
        }
        if filter_query:
            params["filter"] = filter_query

        print(f"Fetching {actual_limit} run summaries for session: {session_id}")
        response = _make_request("/api/v1/runs/query", config, method="POST", json_data=params)
        # Extract only lightweight fields from each run
        runs = response.get("runs", []) if isinstance(response, dict) else response
        lightweight_runs = []
        for run in runs:
            # Only include essential metadata, not full inputs/outputs
            summary = {
                "id": run.get("id"),
                "name": run.get("name"),
                "status": run.get("status"),
                "start_time": run.get("start_time"),
                "end_time": run.get("end_time"),
                "latency": run.get("latency"),
                "run_type": run.get("run_type"),
            }

            # Include error message but truncate it
            if run.get("error"):
                error_msg = str(run.get("error"))
                summary["error"] = error_msg[:200] + "..." if len(error_msg) > 200 else error_msg

            lightweight_runs.append(summary)

        print(f"Returning {len(lightweight_runs)} lightweight run summaries")
        print(f"Provided below are the summaries of the runs for the session below: \n {lightweight_runs}")
        return lightweight_runs
    except Exception as e:
        logger.error(f"An error occurred while listing run summaries: {e}")
        raise


@tool
def get_run_details(
    run_id: str,
    config: Optional[LangSmithConfig] = None,
) -> dict[str, Any]:
    """
    Get full details for a specific run by ID.

    Use this after list_session_runs_summary() to get complete information
    about a specific run, including inputs, outputs, and full error traces.

    Args:
        run_id: UUID of the specific run to fetch
        config: LangSmith configuration (uses env vars if not provided)

    Returns:
        Dictionary containing complete run details:
        - id: Run UUID
        - name: Run name/type
        - inputs: Input data passed to the agent
        - outputs: Output data from the agent
        - error: Full error message and stack trace if run failed
        - start_time: Execution start timestamp
        - end_time: Execution end timestamp
        - latency: Execution duration in milliseconds
        - metadata: Associated metadata
        - feedback_stats: Feedback scores
        - child_runs: List of child run IDs
        - tags: Run tags

    Example:
        >>> config = LangSmithConfig.from_env()
        >>> # First list runs to find the ID
        >>> runs = list_session_runs_summary("my-project", config)
        >>> run_id = runs[0]["id"]
        >>> # Then get full details
        >>> details = get_run_details(run_id, config)
        >>> print(f"Input: {details['inputs']}")
        >>> print(f"Output: {details['outputs']}")
    """
    try:
        print(f"Fetching full details for run: {run_id}")
        if config is None:
            config = LangSmithConfig.from_env()

        # Fetch the specific run by ID
        response = _make_request(f"/api/v1/runs/{run_id}", config)

        logger.info(f"Successfully fetched full details for run {run_id}")
        return response

    except Exception as e:
        logger.error(f"An error occurred while getting run details: {e}")
        raise


@tool
def get_latest_run(
    session_id: str,
    config: Optional[LangSmithConfig] = None,
    include_details: bool = False,
) -> dict[str, Any]:
    """
    Get the most recent run from a session (lightweight by default).

    This tool fetches ONLY the latest run, avoiding context window overflow.
    By default returns lightweight summary, set include_details=True for full data.

    Args:
        session_id: UUID or name of the LangSmith session
        config: LangSmith configuration (uses env vars if not provided)
        include_details: If True, fetches full run details including inputs/outputs

    Returns:
        Dictionary containing the latest run:
        - id: Run UUID
        - name: Run name/type
        - status: Run status (success, error, etc.)
        - start_time: Execution start timestamp
        - end_time: Execution end timestamp
        - latency: Execution duration in milliseconds
        - error: Error message (if present and include_details=False, truncated to 500 chars)
        - Full details if include_details=True

    Example:
        >>> config = LangSmithConfig.from_env()
        >>> latest = get_latest_run("my-agent-project", config)
        >>> print(f"Latest run status: {latest['status']}")
        >>> if latest.get('error'):
        ...     print(f"Error: {latest['error']}")
    """
    try:
        print(f"Fetching latest run from session: {session_id}")
        if config is None:
            config = LangSmithConfig.from_env()

        # Resolve the session name to UUID if needed
        resolved_session_id = _resolve_session_id(session_id, config)

        # Query for just 1 run, sorted by start_time descending
        params = {
            "session": [resolved_session_id],
            "limit": 1,
            "offset": 0,
        }

        response = _make_request("/api/v1/runs/query", config, method="POST", json_data=params)
        runs = response.get("runs", []) if isinstance(response, dict) else response

        if not runs:
            logger.warning(f"No runs found in session {session_id}")
            return {"error": "No runs found in this session"}

        latest_run = runs[0]

        if include_details:
            # Return full details
            logger.info(f"Returning full details for latest run {latest_run['id']}")
            return latest_run
        else:
            # Return lightweight summary
            summary = {
                "id": latest_run.get("id"),
                "name": latest_run.get("name"),
                "status": latest_run.get("status"),
                "start_time": latest_run.get("start_time"),
                "end_time": latest_run.get("end_time"),
                "latency": latest_run.get("latency"),
                "run_type": latest_run.get("run_type"),
            }

            # Include error message but truncate it
            if latest_run.get("error"):
                error_msg = str(latest_run.get("error"))
                summary["error"] = error_msg[:500] + "..." if len(error_msg) > 500 else error_msg

            logger.info(f"Returning lightweight summary for latest run {latest_run['id']}")
            return summary

    except Exception as e:
        logger.error(f"An error occurred while getting latest run: {e}")
        raise


@tool
def get_latest_error_run(
    session_id: str,
    config: Optional[LangSmithConfig] = None,
    include_full_trace: bool = False,
) -> dict[str, Any]:
    """
    Get the most recent FAILED run from a session.

    This tool specifically filters for error runs and returns the latest one,
    perfect for debugging the most recent failure without loading all runs.

    Args:
        session_id: UUID or name of the LangSmith session
        config: LangSmith configuration (uses env vars if not provided)
        include_full_trace: If True, includes full error trace; else truncates to 1000 chars

    Returns:
        Dictionary containing the latest error run:
        - id: Run UUID
        - name: Run name/type
        - status: Run status (will be "error")
        - start_time: Execution start timestamp
        - end_time: Execution end timestamp
        - error: Error message and trace
        - inputs: Input data that caused the error
        - run_type: Type of run

    Example:
        >>> config = LangSmithConfig.from_env()
        >>> latest_error = get_latest_error_run("my-agent-project", config)
        >>> print(f"Latest error: {latest_error['error']}")
        >>> print(f"Inputs that caused it: {latest_error['inputs']}")
    """
    try:
        print(f"Fetching latest ERROR run from session: {session_id}")
        if config is None:
            config = LangSmithConfig.from_env()

        # Resolve the session name to UUID if needed
        resolved_session_id = _resolve_session_id(session_id, config)

        # Query for just 1 error run, sorted by start_time descending
        params = {
            "session": [resolved_session_id],
            "limit": 1,
            "offset": 0,
            "filter": 'eq(status, "error")',
        }

        response = _make_request("/api/v1/runs/query", config, method="POST", json_data=params)
        runs = response.get("runs", []) if isinstance(response, dict) else response

        if not runs:
            logger.warning(f"No error runs found in session {session_id}")
            return {"message": "No failed runs found in this session"}

        latest_error = runs[0]

        # Extract relevant error information
        error_summary = {
            "id": latest_error.get("id"),
            "name": latest_error.get("name"),
            "status": latest_error.get("status"),
            "start_time": latest_error.get("start_time"),
            "end_time": latest_error.get("end_time"),
            "run_type": latest_error.get("run_type"),
            "inputs": latest_error.get("inputs"),  # Include inputs to see what caused the error
        }

        # Handle error message
        if latest_error.get("error"):
            error_msg = str(latest_error.get("error"))
            if include_full_trace:
                error_summary["error"] = error_msg
            else:
                # Truncate to 1000 chars for readability
                error_summary["error"] = error_msg[:1000] + "..." if len(error_msg) > 1000 else error_msg

        logger.info(f"Returning error info for latest failed run {latest_error['id']}")
        return error_summary

    except Exception as e:
        logger.error(f"An error occurred while getting latest error run: {e}")
        raise


@tool
def get_run_error_only(
    run_id: str,
    config: Optional[LangSmithConfig] = None,
    include_full_trace: bool = True,
) -> dict[str, Any]:
    """
    Get ONLY the error information from a specific run.

    This tool fetches a run but returns only error-relevant fields,
    avoiding large context from inputs/outputs. Perfect for error analysis.

    Args:
        run_id: UUID of the specific run
        config: LangSmith configuration (uses env vars if not provided)
        include_full_trace: If True, returns full error trace; else truncates to 2000 chars

    Returns:
        Dictionary containing error information:
        - id: Run UUID
        - name: Run name/type
        - status: Run status
        - error: Full or truncated error message
        - start_time: When the run started
        - end_time: When the run ended
        - run_type: Type of run

    Example:
        >>> config = LangSmithConfig.from_env()
        >>> error_info = get_run_error_only("run-uuid-here", config)
        >>> print(f"Error details: {error_info['error']}")
    """
    try:
        print(f"Fetching ONLY error info for run: {run_id}")
        if config is None:
            config = LangSmithConfig.from_env()

        # Fetch the specific run by ID
        full_run = _make_request(f"/api/v1/runs/{run_id}", config)

        # Extract only error-relevant fields
        error_info = {
            "id": full_run.get("id"),
            "name": full_run.get("name"),
            "status": full_run.get("status"),
            "start_time": full_run.get("start_time"),
            "end_time": full_run.get("end_time"),
            "run_type": full_run.get("run_type"),
        }

        # Handle error message
        if full_run.get("error"):
            error_msg = str(full_run.get("error"))
            if include_full_trace:
                error_info["error"] = error_msg
            else:
                error_info["error"] = error_msg[:2000] + "..." if len(error_msg) > 2000 else error_msg
        else:
            error_info["error"] = "No error found in this run"

        logger.info(f"Returning error-only info for run {run_id}")
        return error_info

    except Exception as e:
        logger.error(f"An error occurred while getting error info for run: {e}")
        raise


@tool
def get_session_insights(
    session_id: str,
    config: Optional[LangSmithConfig] = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Get AI-generated insights and clustering analysis for a session.

    This tool retrieves beta insights features including run clustering,
    pattern detection, and automated analysis. Use this to discover common
    execution patterns, failure modes, and optimization opportunities.

    Args:
        session_id: UUID or name of the LangSmith session
        config: LangSmith configuration (uses env vars if not provided)
        limit: Maximum number of insight jobs to return (default: 20)

    Returns:
        List of insight job dictionaries, each containing:
        - id: Insight job UUID
        - status: Job status (completed, running, failed)
        - clusters: Identified run clusters
        - patterns: Common execution patterns
        - anomalies: Detected anomalies or outliers

    Example:
        >>> config = LangSmithConfig.from_env()
        >>> insights = get_session_insights("my-agent-project", config)
        >>> for insight in insights:
        ...     if insight['status'] == 'completed':
        ...         print(f"Found {len(insight['clusters'])} clusters")
    """
    try:
        print(f'Getting AI DRIVEN session insights...')
        if config is None:
            config = LangSmithConfig.from_env()

        # Resolve the session name to UUID if needed
        resolved_session_id = _resolve_session_id(session_id, config)

        params = {"limit": min(limit, 100)}  # API max is 100

        logger.info(f"Fetching insights for session: {session_id} (resolved to: {resolved_session_id})")
        return _make_request(
            f"/api/v1/sessions/{resolved_session_id}/insights", config, params=params
        )
    except Exception as e:
        logger.error(f"An error occurred while getting the AI insights from langsmith: {e}")
        raise