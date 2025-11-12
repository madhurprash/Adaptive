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
    calculate_all_time_stats: bool = False,
) -> dict[str, Any]:
    """
    Get detailed information about a specific LangSmith session/project.

    This tool retrieves comprehensive information about a tracing session, including
    metadata, run counts, and execution statistics. Use this to understand the scope
    and current state of an agent execution project.

    Args:
        session_id: UUID or name of the LangSmith session/project
        config: LangSmith configuration (uses env vars if not provided)
        include_stats: Whether to include run statistics from the API
        calculate_all_time_stats: If True, fetches ALL runs and calculates error rate 
                                   manually (all-time statistics). This overrides 
                                   include_stats and may be slower for large sessions.

    Returns:
        Dictionary containing session details:
        - id: Session UUID
        - name: Session name
        - description: Session description
        - created_at: Creation timestamp
        - run_count: Total number of runs
        - latency_p50/p99: Latency percentiles (if include_stats=True)
        - error_rate: Error rate percentage (if include_stats=True or calculate_all_time_stats=True)
        - total_runs: Total runs analyzed (if calculate_all_time_stats=True)
        - error_runs: Number of error runs (if calculate_all_time_stats=True)
        - success_runs: Number of successful runs (if calculate_all_time_stats=True)

    Example:
        >>> config = LangSmithConfig.from_env()
        >>> # Get basic info with API stats (may be null for inactive sessions)
        >>> info = get_session_info("my-agent-project", config)
        >>> print(f"Session has {info['run_count']} runs")
        >>> 
        >>> # Get all-time error rate by analyzing all runs
        >>> info = get_session_info("my-agent-project", config, calculate_all_time_stats=True)
        >>> print(f"All-time error rate: {info['error_rate']}%")
    """
    try:
        print(f"In the GET_SESSION_INFO tool, going to get information about specific session/project...")
        if config is None:
            config = LangSmithConfig.from_env()

        # Resolve the session name to UUID if needed
        resolved_session_id = _resolve_session_id(session_id, config)

        # If calculate_all_time_stats is True, fetch all runs and calculate manually
        if calculate_all_time_stats:
            print(f"Calculating all-time statistics by fetching all runs for session: {session_id}")
            
            # Get basic session info first (without stats to avoid 500 errors)
            params = {"include_stats": "false"}
            session_info = _make_request(f"/api/v1/sessions/{resolved_session_id}", config, params=params)
            
            # Fetch all runs to calculate error rate
            all_runs = []
            offset = 0
            limit = 100
            
            while True:
                query_params = {
                    "session": [resolved_session_id],
                    "limit": limit,
                    "offset": offset,
                }
                
                response = _make_request("/api/v1/runs/query", config, method="POST", json_data=query_params)
                runs = response.get("runs", []) if isinstance(response, dict) else response
                
                if not runs:
                    break
                    
                all_runs.extend(runs)
                
                # If we got fewer runs than the limit, we've reached the end
                if len(runs) < limit:
                    break
                    
                offset += limit
                
                # Safety check to avoid infinite loops
                if offset > 10000:
                    logger.warning(f"Reached pagination limit at {offset} runs")
                    break
            
            # Calculate statistics from all runs
            total_runs = len(all_runs)
            error_runs = sum(1 for run in all_runs if run.get("status") == "error")
            success_runs = sum(1 for run in all_runs if run.get("status") == "success")
            error_rate = (error_runs / total_runs * 100) if total_runs > 0 else 0.0
            
            # Add calculated stats to session info
            session_info["total_runs"] = total_runs
            session_info["error_runs"] = error_runs
            session_info["success_runs"] = success_runs
            session_info["error_rate"] = round(error_rate, 2)
            session_info["stats_type"] = "all_time_calculated"
            
            logger.info(f"Calculated all-time stats for {session_id}: {error_rate:.2f}% error rate ({error_runs}/{total_runs} runs)")
            return session_info
        
        # Otherwise, use the API stats (original behavior)
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
    Query and retrieve runs from a specific session with full details.

    This tool fetches detailed run data including inputs, outputs, errors,
    latency, and token usage. Use this to analyze individual agent executions
    and identify failure patterns or performance bottlenecks.

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
        - outputs: Output data from the agent (may include token usage in usage_metadata)
        - error: Error message if run failed
        - start_time: Execution start timestamp
        - end_time: Execution end timestamp
        - latency: Execution duration in seconds
        - metadata: Associated metadata
        - feedback_stats: Feedback scores
        - extra: Additional data (may include token usage in extra.metadata)

    Example:
        >>> config = LangSmithConfig.from_env()
        >>> # Get all runs with inputs, errors, latency, and tokens
        >>> runs = get_runs_from_session("my-agent-project", config)
        >>> for run in runs:
        ...     print(f"Input: {run.get('inputs')}")
        ...     print(f"Error: {run.get('error')}")
        ...     print(f"Latency: {run.get('latency')} seconds")
        ...     # Token usage may be in outputs or extra.metadata
        ...     tokens = run.get('outputs', {}).get('usage_metadata', {})
        ...     print(f"Tokens: {tokens}")
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
    [WORKFLOW STEP 1: START BROAD] List runs from a session with lightweight metadata only.

    USE THIS TOOL WHEN:
    - User asks "what runs are available?" or "show me recent runs"
    - You need an overview of execution history before diving into details
    - Starting analysis of a session - this should be your FIRST tool call
    - You want to identify interesting runs to investigate further

    This tool returns a summary of runs to avoid context window overflow. Use this
    first to see available runs, then use get_run_details() or get_run_trace() for specific runs.

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
        - run_type: Type of run (chain, llm, tool, etc.)
        - inputs: Input data passed to the run (full data included)
        - outputs: Output data from the run (truncated to 500 chars if string)
        - error: Error message (truncated to 200 chars if present)

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
            # Include essential metadata AND inputs (needed for understanding user queries)
            summary = {
                "id": run.get("id"),
                "name": run.get("name"),
                "status": run.get("status"),
                "start_time": run.get("start_time"),
                "end_time": run.get("end_time"),
                "latency": run.get("latency"),
                "run_type": run.get("run_type"),
            }

            # Include inputs - this is critical for understanding what the user asked
            if run.get("inputs"):
                summary["inputs"] = run.get("inputs")

            # Include outputs for completeness (truncate if too large)
            if run.get("outputs"):
                outputs = run.get("outputs")
                # If outputs is a dict, keep it; if it's a large string, truncate
                if isinstance(outputs, str) and len(outputs) > 500:
                    summary["outputs"] = outputs[:500] + "..."
                else:
                    summary["outputs"] = outputs

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
    [WORKFLOW STEP 2: DIVE DEEPER] Get full details for a specific run by ID.

    USE THIS TOOL WHEN:
    - User asks "what are the details of run X?" or "show me the full run data"
    - You've identified an interesting run from list_session_runs_summary() and need complete info
    - You need to see the actual inputs/outputs of a specific execution
    - Investigating a specific run's complete data including metadata and feedback

    Use this after list_session_runs_summary() or get_latest_run() to get complete information
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
    [WORKFLOW STEP 1: START BROAD] Get the most recent run from a session (lightweight by default).

    USE THIS TOOL WHEN:
    - User asks "what's the latest run?" or "show me the most recent execution"
    - You want to quickly check current status without loading all runs
    - Starting investigation and need to see if latest run succeeded/failed
    - You need a quick snapshot of the most recent activity

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
        - run_type: Type of run
        - inputs: Input data passed to the run (full data included)
        - outputs: Output data from the run (truncated to 500 chars if string)
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

            # Include inputs - critical for understanding what the user asked
            if latest_run.get("inputs"):
                summary["inputs"] = latest_run.get("inputs")

            # Include outputs for completeness (truncate if too large)
            if latest_run.get("outputs"):
                outputs = latest_run.get("outputs")
                if isinstance(outputs, str) and len(outputs) > 500:
                    summary["outputs"] = outputs[:500] + "..."
                else:
                    summary["outputs"] = outputs

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
def get_run_trace(
    run_id: str,
    config: Optional[LangSmithConfig] = None,
    include_full_details: bool = False,
) -> dict[str, Any]:
    """
    [WORKFLOW STEP 2: DIVE DEEPER] Get the complete execution trace hierarchy for a specific run.

    USE THIS TOOL WHEN:
    - User asks "what's the trace for this run?" or "show me the execution flow"
    - You need to understand the parent-child relationship structure
    - Investigating how a complex workflow executed step-by-step
    - You want to see the trace tree before getting child run details

    This tool fetches a run and its full trace tree, showing parent-child
    relationships, execution flow, and nested operations. Essential for
    understanding complex agent workflows and debugging multi-step processes.
    Use get_child_runs() after this to get details of specific child operations.

    Args:
        run_id: UUID of the run to trace
        config: LangSmith configuration (uses env vars if not provided)
        include_full_details: If True, includes full inputs/outputs for all runs in trace

    Returns:
        Dictionary containing the run trace:
        - id: Run UUID
        - name: Run name/type
        - status: Run status
        - child_runs: List of child run IDs
        - parent_run_id: Parent run ID (if nested)
        - trace_id: Trace ID linking related runs
        - Full run details if include_full_details=True

    Example:
        >>> config = LangSmithConfig.from_env()
        >>> trace = get_run_trace("run-uuid-here", config)
        >>> print(f"Run has {len(trace.get('child_runs', []))} child runs")
        >>> for child_id in trace.get('child_runs', []):
        ...     print(f"Child run: {child_id}")
    """
    try:
        print(f"Fetching execution trace for run: {run_id}")
        if config is None:
            config = LangSmithConfig.from_env()

        # Fetch the specific run by ID
        run = _make_request(f"/api/v1/runs/{run_id}", config)

        if include_full_details:
            logger.info(f"Returning full trace with details for run {run_id}")
            return run
        else:
            # Return lightweight trace structure
            trace = {
                "id": run.get("id"),
                "name": run.get("name"),
                "status": run.get("status"),
                "run_type": run.get("run_type"),
                "start_time": run.get("start_time"),
                "end_time": run.get("end_time"),
                "latency": run.get("latency"),
                "parent_run_id": run.get("parent_run_id"),
                "child_run_ids": run.get("child_run_ids", []),
                "trace_id": run.get("trace_id"),
            }

            if run.get("error"):
                error_msg = str(run.get("error"))
                trace["error"] = error_msg[:500] + "..." if len(error_msg) > 500 else error_msg

            logger.info(f"Returning lightweight trace structure for run {run_id}")
            return trace

    except Exception as e:
        logger.error(f"An error occurred while getting run trace: {e}")
        raise


@tool
def get_child_runs(
    run_id: str,
    config: Optional[LangSmithConfig] = None,
    include_details: bool = False,
) -> list[dict[str, Any]]:
    """
    [WORKFLOW STEP 3: ANALYZE STRUCTURE] Get all child runs of a specific parent run.

    USE THIS TOOL WHEN:
    - User asks "what are the child runs?" or "show me the sub-operations"
    - You've seen a run has child_run_ids from get_run_trace() and want their details
    - Analyzing the steps within a complex agent operation
    - Understanding what sub-operations were executed in a workflow

    This tool fetches the immediate children of a run, useful for analyzing
    the steps within a complex agent operation. Shows what sub-operations
    were executed as part of a larger workflow. Use after get_run_trace()
    to dive into specific nested operations.

    Args:
        run_id: UUID of the parent run
        config: LangSmith configuration (uses env vars if not provided)
        include_details: If True, includes full details for each child run

    Returns:
        List of child run dictionaries, each containing:
        - id: Child run UUID
        - name: Child run name/type
        - status: Run status
        - run_type: Type of run (tool, llm, chain, etc.)
        - start_time: Execution start
        - end_time: Execution end
        - latency: Duration in milliseconds
        - Full details if include_details=True

    Example:
        >>> config = LangSmithConfig.from_env()
        >>> children = get_child_runs("parent-run-uuid", config)
        >>> for child in children:
        ...     print(f"{child['name']}: {child['status']} ({child['run_type']})")
    """
    try:
        print(f"Fetching child runs for parent run: {run_id}")
        if config is None:
            config = LangSmithConfig.from_env()

        # First get the parent run to extract child run IDs
        parent_run = _make_request(f"/api/v1/runs/{run_id}", config)
        child_run_ids = parent_run.get("child_run_ids", [])

        if not child_run_ids:
            logger.info(f"No child runs found for run {run_id}")
            return []

        print(f"Found {len(child_run_ids)} child runs, fetching details...")

        # Fetch details for each child run
        child_runs = []
        for child_id in child_run_ids:
            try:
                child_run = _make_request(f"/api/v1/runs/{child_id}", config)

                if include_details:
                    child_runs.append(child_run)
                else:
                    # Return lightweight summary
                    summary = {
                        "id": child_run.get("id"),
                        "name": child_run.get("name"),
                        "status": child_run.get("status"),
                        "run_type": child_run.get("run_type"),
                        "start_time": child_run.get("start_time"),
                        "end_time": child_run.get("end_time"),
                        "latency": child_run.get("latency"),
                    }

                    if child_run.get("error"):
                        error_msg = str(child_run.get("error"))
                        summary["error"] = error_msg[:200] + "..." if len(error_msg) > 200 else error_msg

                    child_runs.append(summary)

            except Exception as e:
                logger.warning(f"Failed to fetch child run {child_id}: {e}")
                continue

        logger.info(f"Successfully fetched {len(child_runs)} child runs")
        return child_runs

    except Exception as e:
        logger.error(f"An error occurred while getting child runs: {e}")
        raise


@tool
def compare_runs(
    run_ids: list[str],
    config: Optional[LangSmithConfig] = None,
    comparison_fields: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    [WORKFLOW STEP 4: COMPARE] Compare multiple runs side-by-side to identify differences and patterns.

    USE THIS TOOL WHEN:
    - User asks "compare these runs" or "what's different between run X and Y?"
    - You want to identify patterns across multiple executions
    - Investigating performance regressions or improvements
    - Analyzing how different inputs affect outputs or errors

    This tool fetches multiple runs and presents them in a comparable format,
    highlighting differences in inputs, outputs, errors, and performance.
    Perfect for A/B testing, debugging regressions, or analyzing variations.
    Use after list_session_runs_summary() to identify runs worth comparing.

    Args:
        run_ids: List of run UUIDs to compare (2-5 runs recommended)
        config: LangSmith configuration (uses env vars if not provided)
        comparison_fields: Specific fields to compare (default: status, latency, error, run_type)

    Returns:
        Dictionary containing comparison data:
        - runs: List of run summaries with compared fields
        - differences: Highlighted differences between runs
        - summary: Statistical comparison (avg latency, error rates, etc.)

    Example:
        >>> config = LangSmithConfig.from_env()
        >>> comparison = compare_runs(["run-1-uuid", "run-2-uuid"], config)
        >>> print(f"Latency difference: {comparison['summary']['latency_delta']}ms")
        >>> for diff in comparison['differences']:
        ...     print(f"Difference: {diff}")
    """
    try:
        print(f"Comparing {len(run_ids)} runs...")
        if config is None:
            config = LangSmithConfig.from_env()

        if len(run_ids) < 2:
            raise ValueError("Need at least 2 runs to compare")

        if len(run_ids) > 10:
            logger.warning(f"Comparing {len(run_ids)} runs may be slow, consider reducing")

        # Default comparison fields
        if comparison_fields is None:
            comparison_fields = ["status", "latency", "error", "run_type", "start_time"]

        # Fetch all runs
        runs = []
        for run_id in run_ids:
            try:
                run = _make_request(f"/api/v1/runs/{run_id}", config)
                runs.append(run)
            except Exception as e:
                logger.warning(f"Failed to fetch run {run_id}: {e}")
                continue

        if len(runs) < 2:
            raise ValueError("Failed to fetch enough runs for comparison")

        # Extract comparison data
        comparison_data = {
            "runs": [],
            "differences": [],
            "summary": {}
        }

        for run in runs:
            run_summary = {
                "id": run.get("id"),
                "name": run.get("name"),
            }

            for field in comparison_fields:
                if field == "error" and run.get("error"):
                    # Truncate error for readability
                    error_msg = str(run.get("error"))
                    run_summary[field] = error_msg[:300] + "..." if len(error_msg) > 300 else error_msg
                else:
                    run_summary[field] = run.get(field)

            comparison_data["runs"].append(run_summary)

        # Calculate summary statistics
        latencies = [r.get("latency", 0) for r in runs if r.get("latency")]
        if latencies:
            comparison_data["summary"]["avg_latency_ms"] = sum(latencies) / len(latencies)
            comparison_data["summary"]["min_latency_ms"] = min(latencies)
            comparison_data["summary"]["max_latency_ms"] = max(latencies)
            comparison_data["summary"]["latency_delta_ms"] = max(latencies) - min(latencies)

        # Count status distribution
        statuses = [r.get("status") for r in runs]
        comparison_data["summary"]["status_distribution"] = {
            status: statuses.count(status) for status in set(statuses)
        }

        # Identify key differences
        for field in comparison_fields:
            values = [r.get(field) for r in runs]
            unique_values = set(str(v) for v in values if v is not None)
            if len(unique_values) > 1:
                comparison_data["differences"].append(f"{field}: {len(unique_values)} different values")

        logger.info(f"Successfully compared {len(runs)} runs")
        return comparison_data

    except Exception as e:
        logger.error(f"An error occurred while comparing runs: {e}")
        raise


@tool
def get_session_ai_insights(
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