"""
MLflow MCP Server for comprehensive LLM observability and experiment tracking.

This MCP server exposes MLflow API tools for analyzing execution traces,
tracking experiments, monitoring metrics, managing artifacts, and providing
complete observability for LLM and agent workflows.
"""

import os
import json
import logging
from typing import Any, Optional, Dict, List
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_LIMIT: int = 100
MAX_RESULTS: int = 1000


def _get_mlflow_client() -> MlflowClient:
    """Get an initialized MLflow client."""
    # MLflow client will use environment variables automatically:
    # - MLFLOW_TRACKING_URI
    # - DATABRICKS_HOST and DATABRICKS_TOKEN (for Databricks)
    return MlflowClient()


def _resolve_experiment_id(
    experiment_identifier: str,
    client: MlflowClient,
) -> str:
    """Resolve experiment name or ID to a valid experiment ID."""
    # If it's numeric, assume it's already an ID
    if experiment_identifier.isdigit():
        logger.debug(f"Using experiment ID directly: {experiment_identifier}")
        return experiment_identifier

    # Try to find experiment by name
    logger.info(f"Resolving experiment name '{experiment_identifier}' to ID...")

    try:
        # Try to get experiment by name directly
        experiment = client.get_experiment_by_name(experiment_identifier)
        if experiment:
            exp_id = experiment.experiment_id
            logger.info(f"Resolved '{experiment_identifier}' to ID: {exp_id}")
            return exp_id

        # If not found, search all experiments for similar names
        experiments = client.search_experiments(view_type=ViewType.ALL)
        available_names = [e.name for e in experiments]

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
        client = _get_mlflow_client()

        # Map view_type string to ViewType enum
        view_type_map = {
            "ACTIVE_ONLY": ViewType.ACTIVE_ONLY,
            "DELETED_ONLY": ViewType.DELETED_ONLY,
            "ALL": ViewType.ALL,
        }
        view = view_type_map.get(view_type, ViewType.ACTIVE_ONLY)

        # Search experiments
        experiments = client.search_experiments(
            view_type=view,
            max_results=min(max_results, MAX_RESULTS),
        )

        # Convert to dict and filter by name if specified
        result = []
        for exp in experiments:
            if name_contains and name_contains.lower() not in exp.name.lower():
                continue

            result.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
                "tags": exp.tags,
            })

        logger.info(f"Found {len(result)} experiments")
        return result
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
        client = _get_mlflow_client()
        exp_id = _resolve_experiment_id(experiment_id_or_name, client)

        # Get experiment details
        experiment = client.get_experiment(exp_id)

        result = {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "artifact_location": experiment.artifact_location,
            "lifecycle_stage": experiment.lifecycle_stage,
            "tags": experiment.tags,
            "creation_time": experiment.creation_time,
            "last_update_time": experiment.last_update_time,
        }

        if include_run_stats:
            # Get run statistics
            runs = client.search_runs(
                experiment_ids=[exp_id],
                max_results=5,
                order_by=["attributes.start_time DESC"],
            )

            result["total_runs"] = len(runs)

            # Get recent runs summary
            if runs:
                result["recent_runs"] = [
                    {
                        "run_id": run.info.run_id,
                        "run_name": run.info.run_name,
                        "status": run.info.status,
                        "start_time": run.info.start_time,
                        "end_time": run.info.end_time,
                    }
                    for run in runs[:5]
                ]

        logger.info(f"Retrieved experiment info for: {result['name']}")
        return result
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
        client = _get_mlflow_client()

        experiment_id = client.create_experiment(
            name=name,
            artifact_location=artifact_location,
            tags=tags,
        )

        logger.info(f"Created experiment with ID: {experiment_id}")
        return {"experiment_id": experiment_id}
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
        order_by: List of order by clauses (e.g., ["attributes.start_time DESC", "metrics.accuracy DESC"])

    Returns:
        List of run dictionaries with info, data (metrics/params/tags), and metadata
    """
    try:
        logger.info("Searching MLflow runs...")
        client = _get_mlflow_client()

        # Resolve experiment names to IDs if provided
        if experiment_names:
            resolved_ids = [_resolve_experiment_id(name, client) for name in experiment_names]
            experiment_ids = (experiment_ids or []) + resolved_ids

        # Set default order_by if not specified
        if not order_by:
            order_by = ["attributes.start_time DESC"]

        # Search runs using MLflow client
        runs = client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string or "",
            max_results=min(max_results, MAX_RESULTS),
            order_by=order_by,
        )

        # Convert Run objects to dictionaries
        result = []
        for run in runs:
            result.append({
                "info": {
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    "experiment_id": run.info.experiment_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "artifact_uri": run.info.artifact_uri,
                    "lifecycle_stage": run.info.lifecycle_stage,
                },
                "data": {
                    "metrics": [{"key": k, "value": v} for k, v in run.data.metrics.items()],
                    "params": [{"key": k, "value": v} for k, v in run.data.params.items()],
                    "tags": [{"key": k, "value": v} for k, v in run.data.tags.items()],
                },
            })

        logger.info(f"Found {len(result)} runs")
        return result
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
        client = _get_mlflow_client()

        run = client.get_run(run_id)

        result = {
            "info": {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "artifact_uri": run.info.artifact_uri,
                "lifecycle_stage": run.info.lifecycle_stage,
            },
            "data": {
                "metrics": [{"key": k, "value": v} for k, v in run.data.metrics.items()],
                "params": [{"key": k, "value": v} for k, v in run.data.params.items()],
                "tags": [{"key": k, "value": v} for k, v in run.data.tags.items()],
            },
        }

        if include_children:
            # Search for child runs
            experiment_id = run.info.experiment_id
            child_runs = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=f"tags.`mlflow.parentRunId` = '{run_id}'",
                max_results=100,
            )

            result["child_runs"] = [
                {
                    "run_id": child.info.run_id,
                    "run_name": child.info.run_name,
                    "status": child.info.status,
                }
                for child in child_runs
            ]
            result["child_run_count"] = len(child_runs)

        logger.info(f"Retrieved run details: {result['info']['run_name']}")
        return result
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
        client = _get_mlflow_client()
        exp_id = _resolve_experiment_id(experiment_id_or_name, client)

        filter_str = f"attributes.status = '{status_filter}'" if status_filter else ""

        runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string=filter_str,
            max_results=1,
            order_by=["attributes.start_time DESC"],
        )

        if not runs:
            logger.warning(f"No runs found in experiment {experiment_id_or_name}")
            return {"error": "No runs found matching criteria"}

        run = runs[0]
        result = {
            "info": {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
            },
            "data": {
                "metrics": [{"key": k, "value": v} for k, v in run.data.metrics.items()],
                "params": [{"key": k, "value": v} for k, v in run.data.params.items()],
                "tags": [{"key": k, "value": v} for k, v in run.data.tags.items()],
            },
        }

        logger.info(f"Found latest run: {result['info']['run_id']}")
        return result
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
        client = _get_mlflow_client()
        exp_id = _resolve_experiment_id(experiment_id_or_name, client)

        runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string="attributes.status = 'FAILED'",
            max_results=1,
            order_by=["attributes.start_time DESC"],
        )

        if not runs:
            logger.info(f"No failed runs found in experiment {experiment_id_or_name}")
            return {"message": "No failed runs found"}

        run = runs[0]
        result = {
            "info": {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
            },
            "data": {
                "metrics": [{"key": k, "value": v} for k, v in run.data.metrics.items()],
                "params": [{"key": k, "value": v} for k, v in run.data.params.items()],
                "tags": [{"key": k, "value": v} for k, v in run.data.tags.items()],
            },
        }

        # Truncate error message if not including full traceback
        if not include_traceback:
            error_tags = [t for t in result["data"]["tags"] if t["key"] == "error"]
            for tag in error_tags:
                if len(tag["value"]) > 1000:
                    tag["value"] = tag["value"][:1000] + "..."

        logger.info(f"Found failed run: {result['info']['run_id']}")
        return result
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
        client = _get_mlflow_client()

        if len(run_ids) < 2:
            raise ValueError("Need at least 2 runs to compare")

        if len(run_ids) > 20:
            logger.warning(f"Comparing {len(run_ids)} runs may be slow")

        # Fetch all runs
        runs = []
        for run_id in run_ids:
            try:
                run = client.get_run(run_id)
                runs.append(run)
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
            run_summary = {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
            }

            # Collect all metric and param names
            all_metrics.update(run.data.metrics.keys())
            all_params.update(run.data.params.keys())

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
        client = _get_mlflow_client()

        metric_history = client.get_metric_history(run_id, metric_key)

        # Convert to dict format and limit results
        result = [
            {
                "timestamp": metric.timestamp,
                "step": metric.step,
                "value": metric.value,
            }
            for metric in metric_history[:min(max_results, MAX_RESULTS)]
        ]

        logger.info(f"Retrieved {len(result)} metric history entries")
        return result
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
        client = _get_mlflow_client()

        artifacts = client.list_artifacts(run_id, path)

        # Convert to dict format
        result = [
            {
                "path": artifact.path,
                "is_dir": artifact.is_dir,
                "file_size": artifact.file_size,
            }
            for artifact in artifacts
        ]

        logger.info(f"Found {len(result)} artifacts")
        return result
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
        client = _get_mlflow_client()

        # First get the parent run to find its experiment_id
        try:
            parent_run = client.get_run(parent_run_id)
            experiment_id = parent_run.info.experiment_id
        except Exception as e:
            logger.error(f"Failed to get parent run {parent_run_id}: {e}")
            return []

        # Search for runs with parent run ID tag
        child_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.`mlflow.parentRunId` = '{parent_run_id}'",
            max_results=min(max_results, MAX_RESULTS),
            order_by=["attributes.start_time ASC"],
        )

        # Convert to dict format
        result = []
        for run in child_runs:
            result.append({
                "info": {
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                },
                "data": {
                    "metrics": [{"key": k, "value": v} for k, v in run.data.metrics.items()],
                    "params": [{"key": k, "value": v} for k, v in run.data.params.items()],
                    "tags": [{"key": k, "value": v} for k, v in run.data.tags.items()],
                },
            })

        logger.info(f"Found {len(result)} child runs")
        return result
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
        client = _get_mlflow_client()

        def _build_trace_recursive(
            current_run_id: str,
            current_depth: int,
        ) -> Dict[str, Any]:
            """Recursively build trace tree."""
            if current_depth > max_depth:
                return {"error": "Max depth reached"}

            # Get run details
            try:
                run = client.get_run(current_run_id)
            except Exception as e:
                logger.error(f"Failed to get run {current_run_id}: {e}")
                return {"error": f"Failed to fetch run: {str(e)}"}

            # Build lightweight summary
            trace_node = {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "duration_ms": run.info.end_time - run.info.start_time if run.info.end_time else 0,
                "depth": current_depth,
                "children": [],
            }

            # Add key metrics/params
            if run.data.metrics:
                trace_node["metrics"] = run.data.metrics

            if run.data.params:
                trace_node["params"] = run.data.params

            # Check for error
            if run.info.status == "FAILED":
                trace_node["error"] = run.data.tags.get("error", "Unknown error")

            # Get experiment ID for the search
            experiment_id = run.info.experiment_id

            # Get child runs
            try:
                child_runs = client.search_runs(
                    experiment_ids=[experiment_id],
                    filter_string=f"tags.`mlflow.parentRunId` = '{current_run_id}'",
                    max_results=100,
                    order_by=["attributes.start_time ASC"],
                )
            except Exception as e:
                logger.warning(f"Failed to search for child runs of {current_run_id}: {e}")
                child_runs = []

            # Recursively build children
            for child in child_runs:
                child_trace = _build_trace_recursive(child.info.run_id, current_depth + 1)
                trace_node["children"].append(child_trace)

            trace_node["child_count"] = len(trace_node["children"])

            return trace_node

        # Build the trace tree
        trace = _build_trace_recursive(run_id, 0)

        # Add summary statistics
        def _count_runs(node: Dict[str, Any]) -> int:
            """Count total runs in tree."""
            if "error" in node and "run_id" not in node:
                return 0
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
        client = _get_mlflow_client()
        exp_id = _resolve_experiment_id(experiment_id_or_name, client)

        # Get all runs
        runs = client.search_runs(
            experiment_ids=[exp_id],
            max_results=MAX_RESULTS,
        )

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
        statuses = [r.info.status for r in runs]
        for status in set(statuses):
            count = statuses.count(status)
            stats["status_distribution"][status] = {
                "count": count,
                "percentage": round(count / total_runs * 100, 2),
            }

        # Collect all metrics
        all_metrics = {}
        for run in runs:
            for metric_key, metric_value in run.data.metrics.items():
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
            if current_time - r.info.start_time < day_ms
        )
        recent_7d = sum(
            1
            for r in runs
            if current_time - r.info.start_time < week_ms
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
        client = _get_mlflow_client()
        exp_id = _resolve_experiment_id(experiment_id_or_name, client)

        # Get recent runs
        runs = client.search_runs(
            experiment_ids=[exp_id],
            max_results=min(max_traces, MAX_RESULTS),
            order_by=["attributes.start_time DESC"],
        )

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
            # Extract latency
            if run.info.start_time and run.info.end_time:
                latency_ms = run.info.end_time - run.info.start_time
                latencies.append(latency_ms)

            # Check for errors
            if run.info.status == "FAILED":
                error_msg = run.data.tags.get("error", "Unknown error")
                errors.append(error_msg[:200])

            # Extract model info from tags or params
            model = run.data.params.get("model") or run.data.tags.get("model_name") or run.data.tags.get("model")
            if model:
                models.append(model)

            # Extract token usage from metrics
            metrics = run.data.metrics

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
