"""
MLflow MCP Server for comprehensive LLM observability and experiment tracking.

This MCP server exposes MLflow API tools for analyzing execution traces,
tracking experiments, monitoring metrics, managing artifacts, and providing
complete observability for LLM and agent workflows.
"""

import os
import re
import json
import logging
import requests
from typing import Any, Optional, Dict, List
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Constants
MLFLOW_API_VERSION: str = "/api/2.0/mlflow"
DEFAULT_LIMIT: int = 100
MAX_RESULTS: int = 1000


class MLflowConfig(BaseModel):
    """Configuration for MLflow API access."""

    tracking_uri: str = Field(..., description="MLflow tracking server URI")
    token: Optional[str] = Field(None, description="Databricks token or auth token")

    @classmethod
    def from_env(cls) -> "MLflowConfig":
        """Create configuration from environment variables."""
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise ValueError(
                "MLFLOW_TRACKING_URI environment variable must be set. "
                "For Databricks: set to 'databricks' or your workspace URL. "
                "For local: set to 'http://localhost:5000'"
            )

        # Handle 'databricks' shorthand
        if tracking_uri == "databricks":
            databricks_host = os.getenv("DATABRICKS_HOST")
            if not databricks_host:
                raise ValueError(
                    "DATABRICKS_HOST must be set when using MLFLOW_TRACKING_URI='databricks'"
                )
            tracking_uri = databricks_host.rstrip("/")

        token = os.getenv("DATABRICKS_TOKEN") or os.getenv("MLFLOW_TRACKING_TOKEN")
        return cls(tracking_uri=tracking_uri.rstrip("/"), token=token)


def _make_request(
    endpoint: str,
    config: MLflowConfig,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Make an authenticated request to the MLflow tracking server."""
    url = f"{config.tracking_uri}{MLFLOW_API_VERSION}{endpoint}"
    headers = {"Content-Type": "application/json"}

    if config.token:
        headers["Authorization"] = f"Bearer {config.token}"

    logger.debug(f"Making {method} request to {url}")

    response = requests.request(
        method=method,
        url=url,
        headers=headers,
        params=params,
        json=json_data,
        timeout=30,
    )
    response.raise_for_status()

    if response.text:
        return response.json()
    return {}


def _resolve_experiment_id(
    experiment_identifier: str,
    config: MLflowConfig,
) -> str:
    """Resolve experiment name or ID to a valid experiment ID."""
    # If it's numeric, assume it's already an ID
    if experiment_identifier.isdigit():
        logger.debug(f"Using experiment ID directly: {experiment_identifier}")
        return experiment_identifier

    # Try to find experiment by name
    logger.info(f"Resolving experiment name '{experiment_identifier}' to ID...")

    try:
        # Use search experiments endpoint
        response = _make_request(
            "/experiments/search",
            config,
            method="POST",
            json_data={"max_results": 1000},
        )

        experiments = response.get("experiments", [])

        for exp in experiments:
            if exp.get("name") == experiment_identifier:
                exp_id = exp.get("experiment_id")
                logger.info(f"Resolved '{experiment_identifier}' to ID: {exp_id}")
                return exp_id

        available_names = [e.get("name") for e in experiments if e.get("name")]
        raise ValueError(
            f"Experiment '{experiment_identifier}' not found. "
            f"Available experiments: {', '.join(available_names[:10])}"
        )
    except Exception as e:
        logger.error(f"Error resolving experiment: {e}")
        raise ValueError(
            f"Unable to resolve experiment '{experiment_identifier}'. Error: {e}"
        )


# Initialize MCP server
mcp = FastMCP("MLflow")


# =============================================================================
# EXPERIMENT MANAGEMENT TOOLS
# =============================================================================


@mcp.tool()
def list_experiments(
    name_contains: Optional[str] = None,
    max_results: int = DEFAULT_LIMIT,
    view_type: str = "ACTIVE_ONLY",
) -> List[Dict[str, Any]]:
    """
    List all MLflow experiments with optional filtering.

    Args:
        name_contains: Filter experiments by name substring (case-insensitive)
        max_results: Maximum number of experiments to return (default: 100)
        view_type: Filter by lifecycle stage - ACTIVE_ONLY, DELETED_ONLY, or ALL

    Returns:
        List of experiment dictionaries with id, name, artifact_location, lifecycle_stage
    """
    try:
        logger.info("Listing MLflow experiments...")
        config = MLflowConfig.from_env()

        payload = {"max_results": min(max_results, MAX_RESULTS), "view_type": view_type}

        if name_contains:
            # Use ILIKE for case-insensitive matching
            payload["filter"] = f"name ILIKE '%{name_contains}%'"

        response = _make_request(
            "/experiments/search", config, method="POST", json_data=payload
        )

        experiments = response.get("experiments", [])
        logger.info(f"Found {len(experiments)} experiments")
        return experiments
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise


@mcp.tool()
def get_experiment_info(
    experiment_id_or_name: str,
    include_run_stats: bool = True,
) -> Dict[str, Any]:
    """
    Get detailed information about a specific experiment.

    Args:
        experiment_id_or_name: Experiment ID or name
        include_run_stats: If True, includes count of runs and recent activity

    Returns:
        Dictionary containing experiment details including tags, creation time, artifact location
    """
    try:
        logger.info(f"Getting experiment info for: {experiment_id_or_name}")
        config = MLflowConfig.from_env()
        exp_id = _resolve_experiment_id(experiment_id_or_name, config)

        # Get experiment details
        response = _make_request("/experiments/get", config, params={"experiment_id": exp_id})
        experiment = response.get("experiment", {})

        if include_run_stats:
            # Get run statistics
            runs_response = _make_request(
                "/runs/search",
                config,
                method="POST",
                json_data={"experiment_ids": [exp_id], "max_results": 1},
            )
            total_runs = runs_response.get("total_count", 0)
            experiment["total_runs"] = total_runs

            # Get recent runs
            if total_runs > 0:
                recent_runs = _make_request(
                    "/runs/search",
                    config,
                    method="POST",
                    json_data={
                        "experiment_ids": [exp_id],
                        "max_results": 5,
                        "order_by": ["start_time DESC"],
                    },
                )
                experiment["recent_runs"] = recent_runs.get("runs", [])

        logger.info(f"Retrieved experiment info for: {experiment.get('name')}")
        return experiment
    except Exception as e:
        logger.error(f"Error getting experiment info: {e}")
        raise


@mcp.tool()
def create_experiment(
    name: str,
    artifact_location: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Create a new MLflow experiment.

    Args:
        name: Name for the new experiment (must be unique)
        artifact_location: Optional custom location for storing artifacts
        tags: Optional dictionary of tags to attach to the experiment

    Returns:
        Dictionary with the new experiment_id
    """
    try:
        logger.info(f"Creating experiment: {name}")
        config = MLflowConfig.from_env()

        payload = {"name": name}

        if artifact_location:
            payload["artifact_location"] = artifact_location

        if tags:
            payload["tags"] = [{"key": k, "value": v} for k, v in tags.items()]

        response = _make_request("/experiments/create", config, method="POST", json_data=payload)
        logger.info(f"Created experiment with ID: {response.get('experiment_id')}")
        return response
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise


# =============================================================================
# RUN MANAGEMENT TOOLS
# =============================================================================


@mcp.tool()
def search_runs(
    experiment_ids: Optional[List[str]] = None,
    experiment_names: Optional[List[str]] = None,
    filter_string: Optional[str] = None,
    max_results: int = DEFAULT_LIMIT,
    order_by: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search for runs across experiments with filtering and sorting.

    Args:
        experiment_ids: List of experiment IDs to search in
        experiment_names: List of experiment names to search in (alternative to IDs)
        filter_string: MLflow search filter (e.g., "metrics.accuracy > 0.9 AND params.model = 'gpt-4'")
        max_results: Maximum number of runs to return (default: 100)
        order_by: List of order by clauses (e.g., ["start_time DESC", "metrics.accuracy DESC"])

    Returns:
        List of run dictionaries with info, data (metrics/params/tags), and metadata
    """
    try:
        logger.info("Searching MLflow runs...")
        config = MLflowConfig.from_env()

        # Resolve experiment names to IDs if provided
        if experiment_names:
            resolved_ids = [_resolve_experiment_id(name, config) for name in experiment_names]
            experiment_ids = (experiment_ids or []) + resolved_ids

        payload = {"max_results": min(max_results, MAX_RESULTS)}

        if experiment_ids:
            payload["experiment_ids"] = experiment_ids

        if filter_string:
            payload["filter"] = filter_string

        if order_by:
            payload["order_by"] = order_by
        else:
            payload["order_by"] = ["start_time DESC"]

        response = _make_request("/runs/search", config, method="POST", json_data=payload)
        runs = response.get("runs", [])

        logger.info(f"Found {len(runs)} runs")
        return runs
    except Exception as e:
        logger.error(f"Error searching runs: {e}")
        raise


@mcp.tool()
def get_run_details(
    run_id: str,
    include_children: bool = False,
) -> Dict[str, Any]:
    """
    Get complete details for a specific run.

    Args:
        run_id: UUID of the run
        include_children: If True, includes information about child runs

    Returns:
        Dictionary with complete run info, data (params/metrics/tags), and metadata
    """
    try:
        logger.info(f"Getting run details for: {run_id}")
        config = MLflowConfig.from_env()

        response = _make_request("/runs/get", config, params={"run_id": run_id})
        run = response.get("run", {})

        if include_children:
            # Search for child runs
            child_runs = _make_request(
                "/runs/search",
                config,
                method="POST",
                json_data={"filter": f"tags.mlflow.parentRunId = '{run_id}'", "max_results": 100},
            )
            run["child_runs"] = child_runs.get("runs", [])
            run["child_run_count"] = len(run["child_runs"])

        logger.info(f"Retrieved run details: {run.get('info', {}).get('run_name', run_id)}")
        return run
    except Exception as e:
        logger.error(f"Error getting run details: {e}")
        raise


@mcp.tool()
def get_latest_run(
    experiment_id_or_name: str,
    status_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get the most recent run from an experiment.

    Args:
        experiment_id_or_name: Experiment ID or name
        status_filter: Optional status filter (RUNNING, SCHEDULED, FINISHED, FAILED, KILLED)

    Returns:
        Dictionary with the latest run details or error message if no runs found
    """
    try:
        logger.info(f"Getting latest run from experiment: {experiment_id_or_name}")
        config = MLflowConfig.from_env()
        exp_id = _resolve_experiment_id(experiment_id_or_name, config)

        payload = {
            "experiment_ids": [exp_id],
            "max_results": 1,
            "order_by": ["start_time DESC"],
        }

        if status_filter:
            payload["filter"] = f"attributes.status = '{status_filter}'"

        response = _make_request("/runs/search", config, method="POST", json_data=payload)
        runs = response.get("runs", [])

        if not runs:
            logger.warning(f"No runs found in experiment {experiment_id_or_name}")
            return {"error": "No runs found matching criteria"}

        logger.info(f"Found latest run: {runs[0].get('info', {}).get('run_id')}")
        return runs[0]
    except Exception as e:
        logger.error(f"Error getting latest run: {e}")
        raise


@mcp.tool()
def get_latest_error_run(
    experiment_id_or_name: str,
    include_traceback: bool = True,
) -> Dict[str, Any]:
    """
    Get the most recent failed run from an experiment.

    Args:
        experiment_id_or_name: Experiment ID or name
        include_traceback: If True, includes full error traceback if available

    Returns:
        Dictionary with error run details including status and error information
    """
    try:
        logger.info(f"Getting latest error run from: {experiment_id_or_name}")
        config = MLflowConfig.from_env()
        exp_id = _resolve_experiment_id(experiment_id_or_name, config)

        payload = {
            "experiment_ids": [exp_id],
            "max_results": 1,
            "order_by": ["start_time DESC"],
            "filter": "attributes.status = 'FAILED'",
        }

        response = _make_request("/runs/search", config, method="POST", json_data=payload)
        runs = response.get("runs", [])

        if not runs:
            logger.info(f"No failed runs found in experiment {experiment_id_or_name}")
            return {"message": "No failed runs found"}

        error_run = runs[0]

        # Check for error tag or attribute
        tags = error_run.get("data", {}).get("tags", [])
        error_tag = next((t for t in tags if t.get("key") == "error"), None)

        if error_tag and not include_traceback:
            error_msg = error_tag.get("value", "")
            if len(error_msg) > 1000:
                error_tag["value"] = error_msg[:1000] + "..."

        logger.info(f"Found failed run: {error_run.get('info', {}).get('run_id')}")
        return error_run
    except Exception as e:
        logger.error(f"Error getting latest error run: {e}")
        raise


@mcp.tool()
def compare_runs(
    run_ids: List[str],
    metrics: Optional[List[str]] = None,
    params: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare multiple runs side-by-side.

    Args:
        run_ids: List of run IDs to compare (2-10 recommended)
        metrics: Specific metrics to compare (default: all metrics)
        params: Specific parameters to compare (default: all params)

    Returns:
        Dictionary with comparison data including runs, metrics, params, and summary statistics
    """
    try:
        logger.info(f"Comparing {len(run_ids)} runs...")
        config = MLflowConfig.from_env()

        if len(run_ids) < 2:
            raise ValueError("Need at least 2 runs to compare")

        if len(run_ids) > 20:
            logger.warning(f"Comparing {len(run_ids)} runs may be slow")

        # Fetch all runs
        runs = []
        for run_id in run_ids:
            try:
                response = _make_request("/runs/get", config, params={"run_id": run_id})
                runs.append(response.get("run", {}))
            except Exception as e:
                logger.warning(f"Failed to fetch run {run_id}: {e}")
                continue

        if len(runs) < 2:
            raise ValueError("Failed to fetch enough runs for comparison")

        # Build comparison data
        comparison = {
            "run_count": len(runs),
            "runs": [],
            "metrics_comparison": {},
            "params_comparison": {},
            "summary": {},
        }

        # Extract data from each run
        all_metrics = set()
        all_params = set()

        for run in runs:
            run_data = run.get("data", {})
            run_info = run.get("info", {})

            run_summary = {
                "run_id": run_info.get("run_id"),
                "run_name": run_info.get("run_name"),
                "status": run_info.get("status"),
                "start_time": run_info.get("start_time"),
                "end_time": run_info.get("end_time"),
                "metrics": {},
                "params": {},
            }

            # Extract metrics
            for metric in run_data.get("metrics", []):
                metric_key = metric.get("key")
                metric_value = metric.get("value")
                all_metrics.add(metric_key)
                run_summary["metrics"][metric_key] = metric_value

            # Extract params
            for param in run_data.get("params", []):
                param_key = param.get("key")
                param_value = param.get("value")
                all_params.add(param_key)
                run_summary["params"][param_key] = param_value

            comparison["runs"].append(run_summary)

        # Filter to requested metrics/params or use all
        metrics_to_compare = metrics if metrics else list(all_metrics)
        params_to_compare = params if params else list(all_params)

        # Build metrics comparison
        for metric_name in metrics_to_compare:
            values = []
            for run in comparison["runs"]:
                val = run["metrics"].get(metric_name)
                if val is not None:
                    values.append(val)

            if values:
                comparison["metrics_comparison"][metric_name] = {
                    "values": values,
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "range": max(values) - min(values),
                }

        # Build params comparison
        for param_name in params_to_compare:
            values = []
            for run in comparison["runs"]:
                val = run["params"].get(param_name)
                if val is not None:
                    values.append(val)

            if values:
                unique_values = set(values)
                comparison["params_comparison"][param_name] = {
                    "unique_values": list(unique_values),
                    "variation_count": len(unique_values),
                }

        # Summary statistics
        statuses = [r["status"] for r in comparison["runs"]]
        comparison["summary"]["status_distribution"] = {
            status: statuses.count(status) for status in set(statuses)
        }

        logger.info(f"Successfully compared {len(runs)} runs")
        return comparison
    except Exception as e:
        logger.error(f"Error comparing runs: {e}")
        raise


# =============================================================================
# METRICS AND PARAMETERS TOOLS
# =============================================================================


@mcp.tool()
def get_metric_history(
    run_id: str,
    metric_key: str,
    max_results: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Get the complete history of a metric across steps/time.

    Args:
        run_id: UUID of the run
        metric_key: Name of the metric (e.g., "accuracy", "loss")
        max_results: Maximum number of metric entries to return

    Returns:
        List of metric history entries with timestamp, step, and value
    """
    try:
        logger.info(f"Getting metric history for {metric_key} in run {run_id}")
        config = MLflowConfig.from_env()

        params = {
            "run_id": run_id,
            "metric_key": metric_key,
            "max_results": min(max_results, MAX_RESULTS),
        }

        response = _make_request("/metrics/get-history", config, params=params)
        metrics = response.get("metrics", [])

        logger.info(f"Retrieved {len(metrics)} metric history entries")
        return metrics
    except Exception as e:
        logger.error(f"Error getting metric history: {e}")
        raise


# =============================================================================
# ARTIFACTS TOOLS
# =============================================================================


@mcp.tool()
def list_artifacts(
    run_id: str,
    path: str = "",
) -> List[Dict[str, Any]]:
    """
    List artifacts logged for a run.

    Args:
        run_id: UUID of the run
        path: Optional path prefix to list artifacts under (default: root)

    Returns:
        List of artifact file info with path, size, and is_dir flag
    """
    try:
        logger.info(f"Listing artifacts for run {run_id} at path: {path or 'root'}")
        config = MLflowConfig.from_env()

        params = {"run_id": run_id}
        if path:
            params["path"] = path

        response = _make_request("/artifacts/list", config, params=params)
        files = response.get("files", [])

        logger.info(f"Found {len(files)} artifacts")
        return files
    except Exception as e:
        logger.error(f"Error listing artifacts: {e}")
        raise


# =============================================================================
# TRACE AND SPAN TOOLS (MLflow 3.5+ with tracing support)
# =============================================================================


@mcp.tool()
def get_child_runs(
    parent_run_id: str,
    max_results: int = DEFAULT_LIMIT,
) -> List[Dict[str, Any]]:
    """
    Get all child runs of a parent run (for nested/hierarchical execution traces).

    Args:
        parent_run_id: UUID of the parent run
        max_results: Maximum number of child runs to return

    Returns:
        List of child run summaries with id, name, status, and key metrics
    """
    try:
        logger.info(f"Getting child runs for parent: {parent_run_id}")
        config = MLflowConfig.from_env()

        # Search for runs with parent run ID tag
        payload = {
            "filter": f"tags.mlflow.parentRunId = '{parent_run_id}'",
            "max_results": min(max_results, MAX_RESULTS),
            "order_by": ["start_time ASC"],
        }

        response = _make_request("/runs/search", config, method="POST", json_data=payload)
        child_runs = response.get("runs", [])

        logger.info(f"Found {len(child_runs)} child runs")
        return child_runs
    except Exception as e:
        logger.error(f"Error getting child runs: {e}")
        raise


@mcp.tool()
def get_run_trace(
    run_id: str,
    max_depth: int = 5,
) -> Dict[str, Any]:
    """
    Get the complete execution trace hierarchy for a run (parent + all descendants).

    Args:
        run_id: UUID of the root run
        max_depth: Maximum depth to traverse in the hierarchy (default: 5)

    Returns:
        Dictionary with nested trace structure showing parent-child relationships
    """
    try:
        logger.info(f"Building execution trace for run: {run_id}")
        config = MLflowConfig.from_env()

        def _build_trace_recursive(
            current_run_id: str,
            current_depth: int,
        ) -> Dict[str, Any]:
            """Recursively build trace tree."""
            if current_depth > max_depth:
                return {"error": "Max depth reached"}

            # Get run details
            response = _make_request("/runs/get", config, params={"run_id": current_run_id})
            run = response.get("run", {})

            run_info = run.get("info", {})
            run_data = run.get("data", {})

            # Build lightweight summary
            trace_node = {
                "run_id": run_info.get("run_id"),
                "run_name": run_info.get("run_name"),
                "status": run_info.get("status"),
                "start_time": run_info.get("start_time"),
                "end_time": run_info.get("end_time"),
                "duration_ms": run_info.get("end_time", 0) - run_info.get("start_time", 0),
                "depth": current_depth,
                "children": [],
            }

            # Add key metrics/params
            metrics = {m["key"]: m["value"] for m in run_data.get("metrics", [])}
            params = {p["key"]: p["value"] for p in run_data.get("params", [])}

            if metrics:
                trace_node["metrics"] = metrics
            if params:
                trace_node["params"] = params

            # Check for error
            tags = {t["key"]: t["value"] for t in run_data.get("tags", [])}
            if run_info.get("status") == "FAILED":
                trace_node["error"] = tags.get("error", "Unknown error")

            # Get child runs
            child_response = _make_request(
                "/runs/search",
                config,
                method="POST",
                json_data={
                    "filter": f"tags.mlflow.parentRunId = '{current_run_id}'",
                    "max_results": 100,
                    "order_by": ["start_time ASC"],
                },
            )

            child_runs = child_response.get("runs", [])

            # Recursively build children
            for child in child_runs:
                child_id = child.get("info", {}).get("run_id")
                child_trace = _build_trace_recursive(child_id, current_depth + 1)
                trace_node["children"].append(child_trace)

            trace_node["child_count"] = len(trace_node["children"])

            return trace_node

        # Build the trace tree
        trace = _build_trace_recursive(run_id, 0)

        # Add summary statistics
        def _count_runs(node: Dict[str, Any]) -> int:
            """Count total runs in tree."""
            count = 1
            for child in node.get("children", []):
                count += _count_runs(child)
            return count

        trace["total_runs_in_trace"] = _count_runs(trace)

        logger.info(f"Built trace with {trace['total_runs_in_trace']} total runs")
        return trace
    except Exception as e:
        logger.error(f"Error building run trace: {e}")
        raise


# =============================================================================
# OBSERVABILITY AND ANALYTICS TOOLS
# =============================================================================


@mcp.tool()
def get_experiment_statistics(
    experiment_id_or_name: str,
) -> Dict[str, Any]:
    """
    Get comprehensive statistics for an experiment.

    Args:
        experiment_id_or_name: Experiment ID or name

    Returns:
        Dictionary with run counts, success rates, metric distributions, and trends
    """
    try:
        logger.info(f"Calculating statistics for experiment: {experiment_id_or_name}")
        config = MLflowConfig.from_env()
        exp_id = _resolve_experiment_id(experiment_id_or_name, config)

        # Get all runs
        response = _make_request(
            "/runs/search",
            config,
            method="POST",
            json_data={
                "experiment_ids": [exp_id],
                "max_results": MAX_RESULTS,
            },
        )

        runs = response.get("runs", [])
        total_runs = len(runs)

        if total_runs == 0:
            return {"message": "No runs found in experiment"}

        # Calculate statistics
        stats = {
            "experiment_id": exp_id,
            "total_runs": total_runs,
            "status_distribution": {},
            "metrics_statistics": {},
            "recent_activity": {},
        }

        # Status distribution
        statuses = [r.get("info", {}).get("status") for r in runs]
        for status in set(statuses):
            count = statuses.count(status)
            stats["status_distribution"][status] = {
                "count": count,
                "percentage": round(count / total_runs * 100, 2),
            }

        # Collect all metrics
        all_metrics = {}
        for run in runs:
            for metric in run.get("data", {}).get("metrics", []):
                metric_key = metric.get("key")
                metric_value = metric.get("value")

                if metric_key not in all_metrics:
                    all_metrics[metric_key] = []

                if metric_value is not None:
                    all_metrics[metric_key].append(metric_value)

        # Calculate metric statistics
        for metric_name, values in all_metrics.items():
            if values:
                stats["metrics_statistics"][metric_name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "median": sorted(values)[len(values) // 2],
                }

        # Recent activity (last 24 hours, last 7 days, etc.)
        current_time = int(datetime.now().timestamp() * 1000)
        day_ms = 24 * 60 * 60 * 1000
        week_ms = 7 * day_ms

        recent_24h = sum(
            1
            for r in runs
            if current_time - r.get("info", {}).get("start_time", 0) < day_ms
        )
        recent_7d = sum(
            1
            for r in runs
            if current_time - r.get("info", {}).get("start_time", 0) < week_ms
        )

        stats["recent_activity"] = {
            "last_24_hours": recent_24h,
            "last_7_days": recent_7d,
        }

        logger.info(f"Calculated statistics for {total_runs} runs")
        return stats
    except Exception as e:
        logger.error(f"Error calculating experiment statistics: {e}")
        raise


@mcp.tool()
def analyze_llm_traces(
    experiment_id_or_name: str,
    max_traces: int = 50,
) -> Dict[str, Any]:
    """
    Analyze LLM traces for insights on performance, errors, and patterns.

    Args:
        experiment_id_or_name: Experiment ID or name
        max_traces: Maximum number of traces to analyze (default: 50)

    Returns:
        Dictionary with trace analysis including latency patterns, error analysis, and token usage
    """
    try:
        logger.info(f"Analyzing LLM traces for experiment: {experiment_id_or_name}")
        config = MLflowConfig.from_env()
        exp_id = _resolve_experiment_id(experiment_id_or_name, config)

        # Get recent runs
        response = _make_request(
            "/runs/search",
            config,
            method="POST",
            json_data={
                "experiment_ids": [exp_id],
                "max_results": min(max_traces, MAX_RESULTS),
                "order_by": ["start_time DESC"],
            },
        )

        runs = response.get("runs", [])

        if not runs:
            return {"message": "No runs found to analyze"}

        analysis = {
            "total_traces_analyzed": len(runs),
            "latency_analysis": {},
            "error_analysis": {},
            "token_usage": {},
            "model_distribution": {},
        }

        latencies = []
        errors = []
        models = []
        total_tokens = []
        prompt_tokens = []
        completion_tokens = []

        for run in runs:
            run_info = run.get("info", {})
            run_data = run.get("data", {})

            # Extract latency
            start_time = run_info.get("start_time", 0)
            end_time = run_info.get("end_time", 0)
            if start_time and end_time:
                latency_ms = end_time - start_time
                latencies.append(latency_ms)

            # Check for errors
            if run_info.get("status") == "FAILED":
                tags = {t["key"]: t["value"] for t in run_data.get("tags", [])}
                error_msg = tags.get("error", "Unknown error")
                errors.append(error_msg[:200])

            # Extract model info from tags or params
            params = {p["key"]: p["value"] for p in run_data.get("params", [])}
            tags = {t["key"]: t["value"] for t in run_data.get("tags", [])}

            model = params.get("model") or tags.get("model_name") or tags.get("model")
            if model:
                models.append(model)

            # Extract token usage from metrics
            metrics = {m["key"]: m["value"] for m in run_data.get("metrics", [])}

            if "total_tokens" in metrics:
                total_tokens.append(metrics["total_tokens"])
            if "prompt_tokens" in metrics:
                prompt_tokens.append(metrics["prompt_tokens"])
            if "completion_tokens" in metrics:
                completion_tokens.append(metrics["completion_tokens"])

        # Latency analysis
        if latencies:
            analysis["latency_analysis"] = {
                "count": len(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "avg_ms": sum(latencies) / len(latencies),
                "median_ms": sorted(latencies)[len(latencies) // 2],
                "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 20 else None,
            }

        # Error analysis
        if errors:
            analysis["error_analysis"] = {
                "error_count": len(errors),
                "error_rate": round(len(errors) / len(runs) * 100, 2),
                "recent_errors": errors[:5],
            }
        else:
            analysis["error_analysis"] = {
                "error_count": 0,
                "error_rate": 0.0,
                "message": "No errors found",
            }

        # Token usage
        if total_tokens:
            analysis["token_usage"] = {
                "total_tokens": {
                    "sum": sum(total_tokens),
                    "avg": sum(total_tokens) / len(total_tokens),
                    "min": min(total_tokens),
                    "max": max(total_tokens),
                },
            }

            if prompt_tokens:
                analysis["token_usage"]["prompt_tokens"] = {
                    "sum": sum(prompt_tokens),
                    "avg": sum(prompt_tokens) / len(prompt_tokens),
                }

            if completion_tokens:
                analysis["token_usage"]["completion_tokens"] = {
                    "sum": sum(completion_tokens),
                    "avg": sum(completion_tokens) / len(completion_tokens),
                }

        # Model distribution
        if models:
            model_counts = {}
            for model in models:
                model_counts[model] = model_counts.get(model, 0) + 1

            analysis["model_distribution"] = model_counts

        logger.info(f"Analyzed {len(runs)} LLM traces")
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing LLM traces: {e}")
        raise


if __name__ == "__main__":
    mcp.run(transport="stdio")
