"""
LangSmith MCP Server for monitoring agent execution and retrieving trace data.

This MCP server exposes LangSmith API tools for analyzing execution traces,
extracting insights, and monitoring performance metrics from LangSmith projects.
"""

import os
import re
import logging
import requests
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
    """Make an authenticated request to the LangSmith API."""
    url = f"{config.base_url}{endpoint}"
    headers = {"x-api-key": config.api_key, "Content-Type": "application/json"}

    logger.debug(f"Making {method} request to {url}")

    response = requests.request(
        method=method, url=url, headers=headers, params=params, json=json_data
    )
    response.raise_for_status()
    return response.json()


def _resolve_session_id(
    session_identifier: str,
    config: LangSmithConfig,
) -> str:
    """Resolve a session name or UUID to a valid session UUID."""
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)

    if uuid_pattern.match(session_identifier):
        logger.debug(f"'{session_identifier}' appears to be a UUID, using directly")
        return session_identifier

    # It's a name, so we need to resolve it to a UUID
    logger.info(f"Resolving project name '{session_identifier}' to UUID...")

    try:
        response = _make_request("/api/v1/sessions", config, params={"limit": 100})
        sessions = response if isinstance(response, list) else response.get("sessions", [])

        for session in sessions:
            if session.get("name") == session_identifier:
                session_uuid = session.get("id")
                logger.info(f"Resolved '{session_identifier}' to UUID: {session_uuid}")
                return session_uuid

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


# Initialize MCP server
mcp = FastMCP("LangSmith")


# =============================================================================
# SESSION/PROJECT MANAGEMENT TOOLS
# =============================================================================


@mcp.tool()
def get_session_info(
    session_id: str,
    include_stats: bool = True,
    calculate_all_time_stats: bool = False,
) -> dict[str, Any]:
    """
    Get detailed information about a specific LangSmith session/project.

    Args:
        session_id: UUID or name of the LangSmith session/project
        include_stats: Whether to include run statistics from the API
        calculate_all_time_stats: If True, fetches ALL runs and calculates error rate manually

    Returns:
        Dictionary containing session details (id, name, run_count, error_rate, etc.)
    """
    try:
        logger.info(f"Getting session info for: {session_id}")
        config = LangSmithConfig.from_env()
        resolved_session_id = _resolve_session_id(session_id, config)

        if calculate_all_time_stats:
            logger.info(f"Calculating all-time statistics for session: {session_id}")

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

                if len(runs) < limit:
                    break

                offset += limit

                if offset > 10000:
                    logger.warning(f"Reached pagination limit at {offset} runs")
                    break

            # Calculate statistics
            total_runs = len(all_runs)
            error_runs = sum(1 for run in all_runs if run.get("status") == "error")
            success_runs = sum(1 for run in all_runs if run.get("status") == "success")
            error_rate = (error_runs / total_runs * 100) if total_runs > 0 else 0.0

            session_info["total_runs"] = total_runs
            session_info["error_runs"] = error_runs
            session_info["success_runs"] = success_runs
            session_info["error_rate"] = round(error_rate, 2)
            session_info["stats_type"] = "all_time_calculated"

            logger.info(f"Calculated all-time stats: {error_rate:.2f}% error rate ({error_runs}/{total_runs} runs)")
            return session_info

        # Use API stats
        params = {"include_stats": str(include_stats).lower()}
        logger.debug(f"Fetching session info with params: {params}")

        try:
            return _make_request(f"/api/v1/sessions/{resolved_session_id}", config, params=params)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 500 and include_stats:
                logger.warning(f"Failed with stats, retrying without stats for session: {session_id}")
                params = {"include_stats": "false"}
                return _make_request(f"/api/v1/sessions/{resolved_session_id}", config, params=params)
            else:
                raise
    except Exception as e:
        logger.error(f"Error retrieving session info: {e}")
        raise


@mcp.tool()
def list_sessions(
    name_contains: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    include_stats: bool = False,
) -> list[dict[str, Any]]:
    """
    List all LangSmith sessions/projects with optional filtering.

    Args:
        name_contains: Filter sessions by name substring (case-insensitive)
        limit: Maximum number of sessions to return (default: 100)
        offset: Number of sessions to skip for pagination
        include_stats: Whether to include statistics for each session

    Returns:
        List of session dictionaries with id, name, created_at, run_count
    """
    try:
        logger.info("Listing sessions...")
        config = LangSmithConfig.from_env()

        params = {
            "limit": limit,
            "offset": offset,
            "include_stats": str(include_stats).lower(),
        }
        if name_contains:
            params["name_contains"] = name_contains

        logger.debug(f"Listing sessions with params: {params}")
        return _make_request("/api/v1/sessions", config, params=params)
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise


@mcp.tool()
def get_session_metadata(
    session_id: str,
    k: int = 10,
    root_runs_only: bool = True,
) -> dict[str, list[str]]:
    """
    Get top metadata keys and values from runs in a session.

    Args:
        session_id: UUID or name of the LangSmith session
        k: Number of top values to return per metadata key (default: 10)
        root_runs_only: Only analyze root-level runs (default: True)

    Returns:
        Dictionary mapping metadata keys to lists of top values
    """
    try:
        logger.info(f"Getting session metadata for: {session_id}")
        config = LangSmithConfig.from_env()
        resolved_session_id = _resolve_session_id(session_id, config)

        params = {"k": k, "root_runs_only": str(root_runs_only).lower()}
        logger.debug(f"Fetching metadata with params: {params}")

        return _make_request(
            f"/api/v1/sessions/{resolved_session_id}/metadata", config, params=params
        )
    except Exception as e:
        logger.error(f"Error getting session metadata: {e}")
        raise


# =============================================================================
# RUN MANAGEMENT TOOLS
# =============================================================================


@mcp.tool()
def get_runs_from_session(
    session_id: str,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    filter_query: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Query and retrieve runs from a specific session with full details.

    Args:
        session_id: UUID or name of the LangSmith session
        limit: Maximum number of runs to return (default: 100)
        offset: Number of runs to skip for pagination
        filter_query: LangSmith filter query (e.g., 'eq(status, "error")')

    Returns:
        List of run dictionaries with inputs, outputs, errors, latency, token usage
    """
    try:
        logger.info(f"Getting runs from session: {session_id}")
        config = LangSmithConfig.from_env()
        resolved_session_id = _resolve_session_id(session_id, config)

        params = {
            "session": [resolved_session_id],
            "limit": limit,
            "offset": offset,
        }
        if filter_query:
            params["filter"] = filter_query

        logger.debug(f"Fetching runs with params: {params}")
        runs_from_session = _make_request("/api/v1/runs/query", config, method="POST", json_data=params)
        logger.debug(f"Retrieved runs from session")
        return runs_from_session
    except Exception as e:
        logger.error(f"Error getting runs from session: {e}")
        raise


@mcp.tool()
def list_session_runs_summary(
    session_id: str,
    limit: int = 20,
    offset: int = 0,
    filter_query: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    List runs from a session with lightweight metadata only (no full inputs/outputs).

    Args:
        session_id: UUID or name of the LangSmith session
        limit: Maximum number of runs to return (default: 20, max: 100)
        offset: Number of runs to skip for pagination
        filter_query: LangSmith filter query (e.g., 'eq(status, "error")')

    Returns:
        List of lightweight run summaries with id, name, status, latency, error
    """
    try:
        logger.info(f"Listing run summaries for session: {session_id}")
        config = LangSmithConfig.from_env()
        resolved_session_id = _resolve_session_id(session_id, config)

        actual_limit = min(limit, 100)

        params = {
            "session": [resolved_session_id],
            "limit": actual_limit,
            "offset": offset,
        }
        if filter_query:
            params["filter"] = filter_query

        logger.debug(f"Fetching {actual_limit} run summaries")
        response = _make_request("/api/v1/runs/query", config, method="POST", json_data=params)

        runs = response.get("runs", []) if isinstance(response, dict) else response
        lightweight_runs = []

        for run in runs:
            summary = {
                "id": run.get("id"),
                "name": run.get("name"),
                "status": run.get("status"),
                "start_time": run.get("start_time"),
                "end_time": run.get("end_time"),
                "latency": run.get("latency"),
                "run_type": run.get("run_type"),
            }

            if run.get("inputs"):
                summary["inputs"] = run.get("inputs")

            if run.get("outputs"):
                outputs = run.get("outputs")
                if isinstance(outputs, str) and len(outputs) > 500:
                    summary["outputs"] = outputs[:500] + "..."
                else:
                    summary["outputs"] = outputs

            if run.get("error"):
                error_msg = str(run.get("error"))
                summary["error"] = error_msg[:200] + "..." if len(error_msg) > 200 else error_msg

            lightweight_runs.append(summary)

        logger.info(f"Returning {len(lightweight_runs)} lightweight run summaries")
        return lightweight_runs
    except Exception as e:
        logger.error(f"Error listing run summaries: {e}")
        raise


@mcp.tool()
def get_run_details(run_id: str) -> dict[str, Any]:
    """
    Get full details for a specific run by ID.

    Args:
        run_id: UUID of the specific run to fetch

    Returns:
        Dictionary containing complete run details (inputs, outputs, error, metadata, etc.)
    """
    try:
        logger.info(f"Fetching full details for run: {run_id}")
        config = LangSmithConfig.from_env()
        response = _make_request(f"/api/v1/runs/{run_id}", config)
        logger.info(f"Successfully fetched full details for run {run_id}")
        return response
    except Exception as e:
        logger.error(f"Error getting run details: {e}")
        raise


@mcp.tool()
def get_latest_run(
    session_id: str,
    include_details: bool = False,
) -> dict[str, Any]:
    """
    Get the most recent run from a session (lightweight by default).

    Args:
        session_id: UUID or name of the LangSmith session
        include_details: If True, fetches full run details including inputs/outputs

    Returns:
        Dictionary containing the latest run (lightweight summary or full details)
    """
    try:
        logger.info(f"Fetching latest run from session: {session_id}")
        config = LangSmithConfig.from_env()
        resolved_session_id = _resolve_session_id(session_id, config)

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

            if latest_run.get("inputs"):
                summary["inputs"] = latest_run.get("inputs")

            if latest_run.get("outputs"):
                outputs = latest_run.get("outputs")
                if isinstance(outputs, str) and len(outputs) > 500:
                    summary["outputs"] = outputs[:500] + "..."
                else:
                    summary["outputs"] = outputs

            if latest_run.get("error"):
                error_msg = str(latest_run.get("error"))
                summary["error"] = error_msg[:500] + "..." if len(error_msg) > 500 else error_msg

            logger.info(f"Returning lightweight summary for latest run {latest_run['id']}")
            return summary
    except Exception as e:
        logger.error(f"Error getting latest run: {e}")
        raise


@mcp.tool()
def get_latest_error_run(
    session_id: str,
    include_full_trace: bool = False,
) -> dict[str, Any]:
    """
    Get the most recent FAILED run from a session.

    Args:
        session_id: UUID or name of the LangSmith session
        include_full_trace: If True, includes full error trace; else truncates to 1000 chars

    Returns:
        Dictionary containing the latest error run with error message and inputs
    """
    try:
        logger.info(f"Fetching latest ERROR run from session: {session_id}")
        config = LangSmithConfig.from_env()
        resolved_session_id = _resolve_session_id(session_id, config)

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

        error_summary = {
            "id": latest_error.get("id"),
            "name": latest_error.get("name"),
            "status": latest_error.get("status"),
            "start_time": latest_error.get("start_time"),
            "end_time": latest_error.get("end_time"),
            "run_type": latest_error.get("run_type"),
            "inputs": latest_error.get("inputs"),
        }

        if latest_error.get("error"):
            error_msg = str(latest_error.get("error"))
            if include_full_trace:
                error_summary["error"] = error_msg
            else:
                error_summary["error"] = error_msg[:1000] + "..." if len(error_msg) > 1000 else error_msg

        logger.info(f"Returning error info for latest failed run {latest_error['id']}")
        return error_summary
    except Exception as e:
        logger.error(f"Error getting latest error run: {e}")
        raise


@mcp.tool()
def get_run_error_only(
    run_id: str,
    include_full_trace: bool = True,
) -> dict[str, Any]:
    """
    Get ONLY the error information from a specific run.

    Args:
        run_id: UUID of the specific run
        include_full_trace: If True, returns full error trace; else truncates to 2000 chars

    Returns:
        Dictionary containing error information (id, name, status, error, timestamps)
    """
    try:
        logger.info(f"Fetching ONLY error info for run: {run_id}")
        config = LangSmithConfig.from_env()
        full_run = _make_request(f"/api/v1/runs/{run_id}", config)

        error_info = {
            "id": full_run.get("id"),
            "name": full_run.get("name"),
            "status": full_run.get("status"),
            "start_time": full_run.get("start_time"),
            "end_time": full_run.get("end_time"),
            "run_type": full_run.get("run_type"),
        }

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
        logger.error(f"Error getting error info for run: {e}")
        raise


@mcp.tool()
def get_run_trace(
    run_id: str,
    include_full_details: bool = False,
) -> dict[str, Any]:
    """
    Get the complete execution trace hierarchy for a specific run.

    Args:
        run_id: UUID of the run to trace
        include_full_details: If True, includes full inputs/outputs for all runs in trace

    Returns:
        Dictionary containing the run trace with parent-child relationships
    """
    try:
        logger.info(f"Fetching execution trace for run: {run_id}")
        config = LangSmithConfig.from_env()
        run = _make_request(f"/api/v1/runs/{run_id}", config)

        if include_full_details:
            logger.info(f"Returning full trace with details for run {run_id}")
            return run
        else:
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
        logger.error(f"Error getting run trace: {e}")
        raise


@mcp.tool()
def get_child_runs(
    run_id: str,
    include_details: bool = False,
) -> list[dict[str, Any]]:
    """
    Get all child runs of a specific parent run.

    Args:
        run_id: UUID of the parent run
        include_details: If True, includes full details for each child run

    Returns:
        List of child run dictionaries (lightweight summaries or full details)
    """
    try:
        logger.info(f"Fetching child runs for parent run: {run_id}")
        config = LangSmithConfig.from_env()

        parent_run = _make_request(f"/api/v1/runs/{run_id}", config)
        child_run_ids = parent_run.get("child_run_ids", [])

        if not child_run_ids:
            logger.info(f"No child runs found for run {run_id}")
            return []

        logger.info(f"Found {len(child_run_ids)} child runs, fetching details...")

        child_runs = []
        for child_id in child_run_ids:
            try:
                child_run = _make_request(f"/api/v1/runs/{child_id}", config)

                if include_details:
                    child_runs.append(child_run)
                else:
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
        logger.error(f"Error getting child runs: {e}")
        raise


@mcp.tool()
def compare_runs(
    run_ids: list[str],
    comparison_fields: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Compare multiple runs side-by-side to identify differences and patterns.

    Args:
        run_ids: List of run UUIDs to compare (2-5 runs recommended)
        comparison_fields: Specific fields to compare (default: status, latency, error, run_type)

    Returns:
        Dictionary containing comparison data with runs, differences, and summary stats
    """
    try:
        logger.info(f"Comparing {len(run_ids)} runs...")
        config = LangSmithConfig.from_env()

        if len(run_ids) < 2:
            raise ValueError("Need at least 2 runs to compare")

        if len(run_ids) > 10:
            logger.warning(f"Comparing {len(run_ids)} runs may be slow, consider reducing")

        if comparison_fields is None:
            comparison_fields = ["status", "latency", "error", "run_type", "start_time"]

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
        logger.error(f"Error comparing runs: {e}")
        raise


@mcp.tool()
def get_session_ai_insights(
    session_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Get AI-generated insights and clustering analysis for a session.

    Args:
        session_id: UUID or name of the LangSmith session
        limit: Maximum number of insight jobs to return (default: 20)

    Returns:
        List of insight job dictionaries with clusters, patterns, and anomalies
    """
    try:
        logger.info(f'Getting AI insights for session: {session_id}')
        config = LangSmithConfig.from_env()
        resolved_session_id = _resolve_session_id(session_id, config)

        params = {"limit": min(limit, 100)}

        logger.info(f"Fetching insights for session: {session_id} (resolved to: {resolved_session_id})")
        return _make_request(
            f"/api/v1/sessions/{resolved_session_id}/insights", config, params=params
        )
    except Exception as e:
        logger.error(f"Error getting AI insights: {e}")
        raise


if __name__ == "__main__":
    mcp.run(transport="stdio")
