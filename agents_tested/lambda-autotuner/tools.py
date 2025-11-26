"""
Tools for Lambda Auto-Tuner Agent.

This module provides tools for monitoring Lambda function performance via CloudWatch
and proposing/applying configuration changes to optimize performance.
"""

# import the relevant libraries
import boto3
import logging
from datetime import (
    datetime,
    timedelta,
)
from typing import (
    Dict,
    List,
    Optional,
)
from langchain_core.tools import tool

# Import MLflow tracing utilities
from mlflow_integration.mlflow_tracer import create_trace_context, annotate_span

from constants import (
    DEFAULT_DURATION_BUDGET_PERCENT,
    DEFAULT_MEMORY_STEP_MB,
    DEFAULT_METRIC_PERIOD_MINUTES,
    DEFAULT_ERROR_RATE_THRESHOLD,
    MAX_MEMORY_MB,
    MIN_MEMORY_MB,
)

from pydantic import (
    # this is the core class that is used for data validation and settings management using python type annotations
    BaseModel,
    # this is used to provide additional metadata and validation rules for model fields
    Field,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

class MetricData(BaseModel):
    """
    CloudWatch metric data for Lambda function. This is a data model that is created to get a few
    relevant metrics from CloudWatch for a given lambda function
    """
    duration_p95_ms: Optional[float] = Field(
        None, description="95th percentile duration in milliseconds"
    )
    error_count: int = Field(default=0, description="Total number of errors")
    invocation_count: int = Field(default=0, description="Total number of invocations")
    throttle_count: int = Field(default=0, description="Total number of throttles")
    error_rate: float = Field(default=0.0, description="Error rate (errors/invocations)")

class AnalysisResult(BaseModel):
    """
    Result of analyzing Lambda metrics against thresholds.
    """

    is_healthy: bool = Field(description="Whether the function is performing within acceptable limits")
    duration_budget_used: Optional[float] = Field(
        None, description="Percentage of timeout budget used (0-1)"
    )
    error_rate: float = Field(description="Current error rate (0-1)")
    issues: List[str] = Field(default_factory=list, description="List of identified issues")
    recommendations: List[str] = Field(
        default_factory=list, description="List of recommendations"
    )

class Action(BaseModel):
    """Proposed action to take on Lambda function."""

    action_type: str = Field(
        description="Type of action: 'increase_memory', 'decrease_memory', 'no_change'"
    )
    current_memory_mb: int = Field(description="Current memory allocation in MB")
    proposed_memory_mb: int = Field(description="Proposed new memory allocation in MB")
    reason: str = Field(description="Reason for the proposed action")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in the action (0-1)"
    )

class ActionResult(BaseModel):
    """Result of applying an action to Lambda function."""

    success: bool = Field(description="Whether the action was applied successfully")
    action: Action = Field(description="The action that was applied")
    error_message: Optional[str] = Field(None, description="Error message if action failed")
    dry_run: bool = Field(default=False, description="Whether this was a dry run")


def _get_cloudwatch_client():
    """Get boto3 CloudWatch client."""
    return boto3.client("cloudwatch")


def _get_lambda_client():
    """Get boto3 Lambda client."""
    return boto3.client("lambda")


@tool
def list_lambda_functions(
    region_name: Optional[str] = None,
    max_items: int = 50,
) -> List[Dict]:
    """
    List all Lambda functions in the AWS account.

    Args:
        region_name: AWS region name (optional)
        max_items: Maximum number of functions to return (default: 50)

    Returns:
        List of dictionaries containing function information with keys:
        - FunctionName: Name of the function
        - FunctionArn: ARN of the function
        - Runtime: Runtime environment
        - MemorySize: Memory allocation in MB
        - Timeout: Function timeout in seconds
        - LastModified: Last modification timestamp

    Raises:
        Exception: If there's an error listing Lambda functions

    Example:
        >>> functions = list_lambda_functions(max_items=10)
        >>> for func in functions:
        >>>     print(f"{func['FunctionName']}: {func['MemorySize']}MB")
    """
    logger.info(f"Listing Lambda functions (max: {max_items})")

    try:
        if region_name:
            lambda_client = boto3.client("lambda", region_name=region_name)
        else:
            lambda_client = _get_lambda_client()

        functions = []
        paginator = lambda_client.get_paginator("list_functions")
        page_iterator = paginator.paginate(
            PaginationConfig={"MaxItems": max_items}
        )

        for page in page_iterator:
            for function in page.get("Functions", []):
                # Extract key information
                function_info = {
                    "FunctionName": function.get("FunctionName"),
                    "FunctionArn": function.get("FunctionArn"),
                    "Runtime": function.get("Runtime"),
                    "MemorySize": function.get("MemorySize"),
                    "Timeout": function.get("Timeout"),
                    "LastModified": function.get("LastModified"),
                    "Description": function.get("Description", ""),
                }
                functions.append(function_info)

        logger.info(f"Found {len(functions)} Lambda functions")
        return functions

    except Exception as e:
        logger.error(f"Error listing Lambda functions: {e}")
        raise


@tool
def fetch_lambda_metrics(
    function_name: str,
    period_minutes: int = DEFAULT_METRIC_PERIOD_MINUTES,
    region_name: Optional[str] = None,
) -> MetricData:
    """
    Fetch CloudWatch metrics for a Lambda function.

    Args:
        function_name: Name of the Lambda function
        period_minutes: Time period to fetch metrics for (default: 30 minutes)
        region_name: AWS region name (optional)

    Returns:
        MetricData object containing the fetched metrics

    Raises:
        Exception: If there's an error fetching metrics from CloudWatch

    Example:
        >>> metrics = fetch_lambda_metrics("my-function", period_minutes=60)
        >>> print(f"P95 Duration: {metrics.duration_p95_ms}ms")
    """
    logger.info(f"Fetching metrics for Lambda function: {function_name}")

    try:
        if region_name:
            cloudwatch = boto3.client("cloudwatch", region_name=region_name)
        else:
            cloudwatch = _get_cloudwatch_client()

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=period_minutes)

        # Define metrics to fetch
        metric_queries = [
            {
                "Id": "duration_p95",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/Lambda",
                        "MetricName": "Duration",
                        "Dimensions": [{"Name": "FunctionName", "Value": function_name}],
                    },
                    "Period": period_minutes * 60,
                    "Stat": "p95",
                },
            },
            {
                "Id": "errors",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/Lambda",
                        "MetricName": "Errors",
                        "Dimensions": [{"Name": "FunctionName", "Value": function_name}],
                    },
                    "Period": period_minutes * 60,
                    "Stat": "Sum",
                },
            },
            {
                "Id": "invocations",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/Lambda",
                        "MetricName": "Invocations",
                        "Dimensions": [{"Name": "FunctionName", "Value": function_name}],
                    },
                    "Period": period_minutes * 60,
                    "Stat": "Sum",
                },
            },
            {
                "Id": "throttles",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/Lambda",
                        "MetricName": "Throttles",
                        "Dimensions": [{"Name": "FunctionName", "Value": function_name}],
                    },
                    "Period": period_minutes * 60,
                    "Stat": "Sum",
                },
            },
        ]

        response = cloudwatch.get_metric_data(
            MetricDataQueries=metric_queries,
            StartTime=start_time,
            EndTime=end_time,
        )
        print(f"Response from the FETCH_LAMBDA_METRICS function: {response}")
        # Parse response
        metrics = MetricData()

        for result in response["MetricDataResults"]:
            if result["Values"]:
                value = result["Values"][0]
                if result["Id"] == "duration_p95":
                    metrics.duration_p95_ms = value
                elif result["Id"] == "errors":
                    metrics.error_count = int(value)
                elif result["Id"] == "invocations":
                    metrics.invocation_count = int(value)
                elif result["Id"] == "throttles":
                    metrics.throttle_count = int(value)

        # Calculate error rate
        if metrics.invocation_count > 0:
            metrics.error_rate = metrics.error_count / metrics.invocation_count
        else:
            metrics.error_rate = 0.0
        print(f"Successfully fetched metrics: {metrics.model_dump()}")
        return metrics
    except Exception as e:
        logger.error(f"Error fetching metrics for {function_name}: {e}")
        raise

@tool
def get_lambda_configuration(
    function_name: str,
    region_name: Optional[str] = None,
) -> Dict:
    """
    Get current Lambda function configuration.

    Args:
        function_name: Name of the Lambda function
        region_name: AWS region name (optional)

    Returns:
        Dictionary containing function configuration

    Raises:
        Exception: If there's an error fetching configuration

    Example:
        >>> config = get_lambda_configuration("my-function")
        >>> print(f"Memory: {config['MemorySize']}MB")
    """
    logger.info(f"Fetching configuration for Lambda function: {function_name}")

    try:
        if region_name:
            lambda_client = boto3.client("lambda", region_name=region_name)
        else:
            lambda_client = _get_lambda_client()
        print(f"In the GET_LAMBDA_CONFIGURATION function, the function name is: {function_name}. Getting the configuration...")
        response = lambda_client.get_function_configuration(FunctionName=function_name)
        print(f"Successfully fetched configuration: {response}")
        return response
    except Exception as e:
        logger.error(f"Error fetching configuration for {function_name}: {e}")
        raise

@tool
def analyze_metrics(
    metrics: MetricData,
    timeout_ms: int,
    duration_budget_percent: float = DEFAULT_DURATION_BUDGET_PERCENT,
    error_rate_threshold: float = DEFAULT_ERROR_RATE_THRESHOLD,
) -> AnalysisResult:
    """
    Analyze Lambda metrics against thresholds.

    Args:
        metrics: Metric data to analyze
        timeout_ms: Lambda function timeout in milliseconds
        duration_budget_percent: Acceptable percentage of timeout to use (default: 0.70)
        error_rate_threshold: Maximum acceptable error rate (default: 0.01)

    Returns:
        AnalysisResult containing health status and recommendations

    Example:
        >>> result = analyze_metrics(metrics, timeout_ms=3000)
        >>> if not result.is_healthy:
        >>>     print(f"Issues: {result.issues}")
    """
    print("Analyzing metrics against thresholds. From the ANALYZE_METRICS tool...")
    issues: List[str] = []
    recommendations: List[str] = []
    is_healthy = True
    # Calculate duration budget
    duration_budget_ms = timeout_ms * duration_budget_percent
    duration_budget_used: Optional[float] = None
    if metrics.duration_p95_ms is not None:
        duration_budget_used = metrics.duration_p95_ms / timeout_ms
        if metrics.duration_p95_ms > duration_budget_ms:
            is_healthy = False
            issues.append(
                f"P95 duration ({metrics.duration_p95_ms:.0f}ms) exceeds "
                f"{duration_budget_percent*100:.0f}% of timeout ({duration_budget_ms:.0f}ms)"
            )
            recommendations.append(
                "Consider increasing memory allocation to improve performance"
            )
        # Check if significantly under budget (opportunity to reduce cost)
        if metrics.duration_p95_ms < timeout_ms * 0.30:
            recommendations.append(
                f"Function is using only {duration_budget_used*100:.0f}% of timeout budget. "
                "Consider decreasing memory to reduce costs."
            )
    # Check error rate
    if metrics.error_rate > error_rate_threshold:
        is_healthy = False
        issues.append(
            f"Error rate ({metrics.error_rate*100:.2f}%) exceeds threshold "
            f"({error_rate_threshold*100:.2f}%)"
        )
        recommendations.append("Investigate errors and consider increasing memory")
    # Check throttles
    if metrics.throttle_count > 0:
        is_healthy = False
        issues.append(f"Function experienced {metrics.throttle_count} throttles")
        recommendations.append("Check concurrency limits and consider increasing reserved concurrency")
    # Check if there's insufficient data
    if metrics.invocation_count == 0:
        recommendations.append("No invocations in the period - unable to assess performance")
    result = AnalysisResult(
        is_healthy=is_healthy,
        duration_budget_used=duration_budget_used,
        error_rate=metrics.error_rate,
        issues=issues,
        recommendations=recommendations,
    )
    print(f"Analysis result: {result.model_dump()}")
    return result

@tool
def decide_action(
    analysis: AnalysisResult,
    current_memory_mb: int,
    memory_step_mb: int = DEFAULT_MEMORY_STEP_MB,
    max_memory_mb: int = MAX_MEMORY_MB,
    min_memory_mb: int = MIN_MEMORY_MB,
) -> Action:
    """
    Decide what action to take based on analysis results.

    Args:
        analysis: Analysis result from analyze_metrics
        current_memory_mb: Current memory allocation in MB
        memory_step_mb: Step size for memory changes (default: 128MB)
        max_memory_mb: Maximum allowed memory (default: 10240MB)
        min_memory_mb: Minimum allowed memory (default: 128MB)

    Returns:
        Action object describing the proposed action

    Example:
        >>> action = decide_action(analysis, current_memory_mb=512)
        >>> print(f"Action: {action.action_type}, New memory: {action.proposed_memory_mb}MB")
    """
    logger.info(f"Deciding action based on analysis (current memory: {current_memory_mb}MB)")

    # If healthy and no recommendations, no change needed
    if analysis.is_healthy and not analysis.recommendations:
        return Action(
            action_type="no_change",
            current_memory_mb=current_memory_mb,
            proposed_memory_mb=current_memory_mb,
            reason="Function is performing within acceptable limits",
            confidence=1.0,
        )

    # Check for performance issues requiring more memory
    performance_issues = any(
        "duration" in issue.lower() or "error" in issue.lower() for issue in analysis.issues
    )

    if performance_issues:
        if current_memory_mb >= max_memory_mb:
            return Action(
                action_type="no_change",
                current_memory_mb=current_memory_mb,
                proposed_memory_mb=current_memory_mb,
                reason=f"Function has performance issues but already at max memory ({max_memory_mb}MB)",
                confidence=0.5,
            )
        proposed_memory = min(current_memory_mb + memory_step_mb, max_memory_mb)
        return Action(
            action_type="increase_memory",
            current_memory_mb=current_memory_mb,
            proposed_memory_mb=proposed_memory,
            reason=f"Performance issues detected: {', '.join(analysis.issues)}",
            confidence=0.8,
        )
    # Check for opportunity to reduce memory (cost optimization)
    if analysis.duration_budget_used is not None and analysis.duration_budget_used < 0.30:
        if current_memory_mb <= min_memory_mb:
            return Action(
                action_type="no_change",
                current_memory_mb=current_memory_mb,
                proposed_memory_mb=current_memory_mb,
                reason=f"Function is underutilized but already at min memory ({min_memory_mb}MB)",
                confidence=0.5,
            )
        proposed_memory = max(current_memory_mb - memory_step_mb, min_memory_mb)
        return Action(
            action_type="decrease_memory",
            current_memory_mb=current_memory_mb,
            proposed_memory_mb=proposed_memory,
            reason=f"Function using only {analysis.duration_budget_used*100:.0f}% of timeout budget",
            confidence=0.6,
        )
    # Default to no change
    return Action(
        action_type="no_change",
        current_memory_mb=current_memory_mb,
        proposed_memory_mb=current_memory_mb,
        reason="No clear action needed based on current analysis",
        confidence=0.5)

@tool
def apply_action(
    function_name: str,
    action: Action,
    dry_run: bool = True,
    region_name: Optional[str] = None,
) -> ActionResult:
    """
    Apply the proposed action to the Lambda function.

    Args:
        function_name: Name of the Lambda function
        action: Action to apply
        dry_run: If True, only simulate the action (default: True)
        region_name: AWS region name (optional)

    Returns:
        ActionResult containing the result of applying the action

    Raises:
        Exception: If there's an error applying the action (when not dry_run)

    Example:
        >>> result = apply_action("my-function", action, dry_run=False)
        >>> if result.success:
        >>>     print("Action applied successfully")
    """
    logger.info(
        f"Applying action to {function_name}: {action.action_type} "
        f"(dry_run={dry_run})"
    )

    if action.action_type == "no_change":
        return ActionResult(
            success=True,
            action=action,
            dry_run=dry_run,
        )

    if dry_run:
        logger.info(
            f"DRY RUN: Would update {function_name} memory from "
            f"{action.current_memory_mb}MB to {action.proposed_memory_mb}MB"
        )
        return ActionResult(
            success=True,
            action=action,
            dry_run=True,
        )

    # Actually apply the action
    try:
        if region_name:
            lambda_client = boto3.client("lambda", region_name=region_name)
        else:
            lambda_client = _get_lambda_client()

        response = lambda_client.update_function_configuration(
            FunctionName=function_name,
            MemorySize=action.proposed_memory_mb,
        )

        logger.info(
            f"Successfully updated {function_name} memory to {action.proposed_memory_mb}MB"
        )

        return ActionResult(
            success=True,
            action=action,
            dry_run=False,
        )

    except Exception as e:
        error_msg = f"Error applying action to {function_name}: {e}"
        logger.error(error_msg)
        return ActionResult(
            success=False,
            action=action,
            error_message=error_msg,
            dry_run=False,
        )

"""
CloudWatch Monitoring Tools

Provides LangChain tools for AWS CloudWatch monitoring operations including
dashboards, logs, alarms, and cross-account access.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from langchain_core.tools import tool
from aws_helpers import _get_cross_account_client, _format_account_context

logger = logging.getLogger(__name__)


# Service to log group prefix mapping
SERVICE_LOG_GROUPS = {
    "lambda": ["/aws/lambda/"],
    "ec2": ["/aws/ec2/", "/var/log/"],
    "rds": ["/aws/rds/"],
    "eks": ["/aws/eks/"],
    "apigateway": ["/aws/apigateway/"],
    "bedrock": ["/aws/bedrock/"],
    "vpc": ["/aws/vpc/"],
    "iam": ["/aws/iam/"],
    "s3": ["/aws/s3/"],
    "cloudtrail": ["/aws/cloudtrail/"],
    "waf": ["/aws/waf/"],
}


@tool
def list_cloudwatch_dashboards(
    account_id: Optional[str] = None,
    role_name: Optional[str] = None,
) -> str:
    """
    List all CloudWatch dashboards in an AWS account.

    Use this tool to discover available CloudWatch dashboards for monitoring.
    Supports cross-account access when account_id and role_name are provided.

    Args:
        account_id: Target AWS account ID for cross-account access (optional)
        role_name: IAM role name to assume in target account (optional)

    Returns:
        Formatted string with list of dashboard names and descriptions
    """
    try:
        cloudwatch = _get_cross_account_client("cloudwatch", account_id, role_name)
        response = cloudwatch.list_dashboards()

        dashboards = response.get("DashboardEntries", [])
        account_context = _format_account_context(account_id)

        if not dashboards:
            return f"No CloudWatch dashboards found in {account_context}."

        result = [f"Found {len(dashboards)} CloudWatch dashboard(s) in {account_context}:\n"]

        for dashboard in dashboards:
            result.append(f"  - {dashboard['DashboardName']}")

        logger.info(f"Listed {len(dashboards)} dashboards from {account_context}")
        return "\n".join(result)

    except Exception as e:
        error_msg = f"Error listing CloudWatch dashboards: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def get_dashboard_summary(
    dashboard_name: str,
    account_id: Optional[str] = None,
    role_name: Optional[str] = None,
) -> str:
    """
    Get detailed summary of a specific CloudWatch dashboard.

    Use this tool to retrieve configuration details for a specific dashboard.

    Args:
        dashboard_name: Name of the CloudWatch dashboard
        account_id: Target AWS account ID for cross-account access (optional)
        role_name: IAM role name to assume in target account (optional)

    Returns:
        Formatted string with dashboard summary
    """
    try:
        cloudwatch = _get_cross_account_client("cloudwatch", account_id, role_name)
        response = cloudwatch.get_dashboard(DashboardName=dashboard_name)

        account_context = _format_account_context(account_id)
        dashboard_body = response.get("DashboardBody", "")

        result = [
            f"Dashboard: {dashboard_name}",
            f"Account: {account_context}",
            f"ARN: {response.get('DashboardArn', 'N/A')}",
            f"\nConfiguration retrieved successfully.",
        ]

        logger.info(f"Retrieved dashboard summary for {dashboard_name}")
        return "\n".join(result)

    except Exception as e:
        error_msg = f"Error getting dashboard summary for '{dashboard_name}': {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def list_log_groups(
    account_id: Optional[str] = None,
    role_name: Optional[str] = None,
    limit: int = 50,
) -> str:
    """
    List CloudWatch log groups in an AWS account.

    Use this tool to discover available log groups for analysis.

    Args:
        account_id: Target AWS account ID for cross-account access (optional)
        role_name: IAM role name to assume in target account (optional)
        limit: Maximum number of log groups to return (default: 50)

    Returns:
        Formatted string with list of log group names
    """
    try:
        logs_client = _get_cross_account_client("logs", account_id, role_name)
        account_context = _format_account_context(account_id)

        log_groups = []
        paginator = logs_client.get_paginator("describe_log_groups")

        for page in paginator.paginate():
            for log_group in page["logGroups"]:
                log_groups.append(log_group["logGroupName"])
                if len(log_groups) >= limit:
                    break
            if len(log_groups) >= limit:
                break

        if not log_groups:
            return f"No log groups found in {account_context}."

        result = [f"Found {len(log_groups)} log group(s) in {account_context}:\n"]
        for log_group in log_groups:
            result.append(f"  - {log_group}")

        logger.info(f"Listed {len(log_groups)} log groups from {account_context}")
        return "\n".join(result)

    except Exception as e:
        error_msg = f"Error listing log groups: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def fetch_cloudwatch_logs_for_service(
    service_name: str,
    hours: int = 1,
    account_id: Optional[str] = None,
    role_name: Optional[str] = None,
    max_events: int = 50,
) -> str:
    """
    Fetch recent CloudWatch logs for a specific AWS service.

    Use this tool to retrieve and analyze recent log entries from services like
    Lambda, EC2, RDS, EKS, API Gateway, Amazon Bedrock, etc.

    Args:
        service_name: AWS service name (e.g., 'lambda', 'ec2', 'bedrock')
        hours: Number of hours of logs to retrieve (default: 1)
        account_id: Target AWS account ID for cross-account access (optional)
        role_name: IAM role name to assume in target account (optional)
        max_events: Maximum number of log events to return (default: 50)

    Returns:
        Formatted string with log entries
    """
    try:
        logs_client = _get_cross_account_client("logs", account_id, role_name)
        account_context = _format_account_context(account_id)

        # Get log group prefixes for the service
        log_group_prefixes = SERVICE_LOG_GROUPS.get(
            service_name.lower(),
            [f"/aws/{service_name}/"],
        )

        start_time = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
        all_logs = []

        for prefix in log_group_prefixes:
            try:
                paginator = logs_client.get_paginator("describe_log_groups")
                for page in paginator.paginate(logGroupNamePrefix=prefix):
                    for log_group in page["logGroups"]:
                        try:
                            events = logs_client.filter_log_events(
                                logGroupName=log_group["logGroupName"],
                                startTime=start_time,
                                limit=max_events,
                            )

                            for event in events.get("events", []):
                                timestamp = datetime.fromtimestamp(
                                    event["timestamp"] / 1000
                                ).isoformat()
                                all_logs.append(
                                    {
                                        "timestamp": timestamp,
                                        "log_group": log_group["logGroupName"],
                                        "message": event["message"],
                                    }
                                )

                                if len(all_logs) >= max_events:
                                    break

                        except Exception as log_error:
                            logger.warning(
                                f"Error fetching logs from {log_group['logGroupName']}: {str(log_error)}"
                            )
                            continue

                        if len(all_logs) >= max_events:
                            break

                    if len(all_logs) >= max_events:
                        break

            except Exception as group_error:
                logger.warning(
                    f"Error listing log groups with prefix {prefix}: {str(group_error)}"
                )
                continue

        if not all_logs:
            return f"No logs found for service '{service_name}' in the last {hours} hour(s) in {account_context}."

        result = [
            f"Retrieved {len(all_logs)} log entries for service '{service_name}' from {account_context}:\n"
        ]

        for log in all_logs[:max_events]:
            result.append(f"[{log['timestamp']}] {log['log_group']}")
            result.append(f"  {log['message'][:200]}...\n")

        logger.info(
            f"Retrieved {len(all_logs)} log entries for service {service_name}"
        )
        return "\n".join(result)

    except Exception as e:
        error_msg = f"Error fetching logs for service '{service_name}': {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def analyze_log_group(
    log_group_name: str,
    hours: int = 1,
    account_id: Optional[str] = None,
    role_name: Optional[str] = None,
) -> str:
    """
    Analyze a specific CloudWatch log group for errors and patterns.

    Use this tool to get insights into log patterns, error rates, and anomalies
    in a specific log group.

    Args:
        log_group_name: Name of the CloudWatch log group to analyze
        hours: Number of hours of logs to analyze (default: 1)
        account_id: Target AWS account ID for cross-account access (optional)
        role_name: IAM role name to assume in target account (optional)

    Returns:
        Formatted string with analysis results
    """
    try:
        logs_client = _get_cross_account_client("logs", account_id, role_name)
        account_context = _format_account_context(account_id)

        start_time = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)

        events = logs_client.filter_log_events(
            logGroupName=log_group_name,
            startTime=start_time,
            limit=1000,
        )

        log_events = events.get("events", [])
        total_events = len(log_events)

        if total_events == 0:
            return f"No log events found in '{log_group_name}' for the last {hours} hour(s) in {account_context}."

        # Analyze for errors
        error_count = 0
        warning_count = 0
        error_keywords = ["error", "fail", "exception", "critical"]
        warning_keywords = ["warning", "warn"]

        for event in log_events:
            message_lower = event["message"].lower()
            if any(keyword in message_lower for keyword in error_keywords):
                error_count += 1
            elif any(keyword in message_lower for keyword in warning_keywords):
                warning_count += 1

        error_rate = (error_count / total_events * 100) if total_events > 0 else 0
        warning_rate = (warning_count / total_events * 100) if total_events > 0 else 0

        result = [
            f"Log Group Analysis: {log_group_name}",
            f"Account: {account_context}",
            f"Time Range: Last {hours} hour(s)",
            f"\nSummary:",
            f"  Total Events: {total_events}",
            f"  Errors: {error_count} ({error_rate:.1f}%)",
            f"  Warnings: {warning_count} ({warning_rate:.1f}%)",
        ]

        if error_count > 0:
            result.append(
                f"\n[!]  High error rate detected! Investigate immediately."
            )
        elif warning_count > total_events * 0.1:
            result.append(
                f"\n[!]  Elevated warning count. Review may be needed."
            )
        else:
            result.append(f"\n Log group appears healthy.")

        logger.info(
            f"Analyzed log group {log_group_name}: {total_events} events, {error_count} errors"
        )
        return "\n".join(result)

    except Exception as e:
        error_msg = f"Error analyzing log group '{log_group_name}': {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def get_cloudwatch_alarms_for_service(
    service_name: str,
    account_id: Optional[str] = None,
    role_name: Optional[str] = None,
) -> str:
    """
    Get CloudWatch alarms related to a specific AWS service.

    Use this tool to check alarm status and identify issues with AWS services.

    Args:
        service_name: AWS service name (e.g., 'lambda', 'ec2', 'bedrock')
        account_id: Target AWS account ID for cross-account access (optional)
        role_name: IAM role name to assume in target account (optional)

    Returns:
        Formatted string with alarm details
    """
    try:
        cloudwatch = _get_cross_account_client("cloudwatch", account_id, role_name)
        account_context = _format_account_context(account_id)

        response = cloudwatch.describe_alarms()
        all_alarms = response.get("MetricAlarms", [])

        # Filter alarms related to the service
        service_alarms = []
        for alarm in all_alarms:
            alarm_name = alarm.get("AlarmName", "").lower()
            namespace = alarm.get("Namespace", "").lower()

            if service_name.lower() in alarm_name or service_name.lower() in namespace:
                service_alarms.append(
                    {
                        "name": alarm["AlarmName"],
                        "state": alarm["StateValue"],
                        "reason": alarm.get("StateReason", "N/A"),
                        "namespace": alarm.get("Namespace", "N/A"),
                    }
                )

        if not service_alarms:
            return f"No CloudWatch alarms found for service '{service_name}' in {account_context}."

        # Group by state
        alarm_state = alarm_ok = in_alarm = insufficient_data = 0

        for alarm in service_alarms:
            if alarm["state"] == "OK":
                alarm_ok += 1
            elif alarm["state"] == "ALARM":
                in_alarm += 1
            else:
                insufficient_data += 1

        result = [
            f"CloudWatch Alarms for '{service_name}' in {account_context}:",
            f"\nSummary:",
            f"  Total Alarms: {len(service_alarms)}",
            f"  OK: {alarm_ok}",
            f"  ALARM: {in_alarm}",
            f"  INSUFFICIENT_DATA: {insufficient_data}",
            f"\nAlarm Details:",
        ]

        for alarm in service_alarms:
            state_icon = (
                "" if alarm["state"] == "OK" else "[!]" if alarm["state"] == "ALARM" else "?"
            )
            result.append(f"  {state_icon} {alarm['name']}: {alarm['state']}")
            if alarm["state"] == "ALARM":
                result.append(f"      Reason: {alarm['reason']}")

        logger.info(
            f"Found {len(service_alarms)} alarms for service {service_name} ({in_alarm} in ALARM state)"
        )
        return "\n".join(result)

    except Exception as e:
        error_msg = (
            f"Error getting CloudWatch alarms for service '{service_name}': {str(e)}"
        )
        logger.error(error_msg)
        return error_msg


@tool
def setup_cross_account_access(
    account_id: str,
    role_name: str,
) -> str:
    """
    Setup and verify cross-account access to CloudWatch and logs.

    Use this tool to test cross-account IAM role configuration before
    performing monitoring operations.

    Args:
        account_id: Target AWS account ID
        role_name: IAM role name to assume in target account

    Returns:
        Formatted string with verification results
    """
    try:
        # Test cross-account access
        test_client = _get_cross_account_client("sts", account_id, role_name)
        identity = test_client.get_caller_identity()

        assumed_account = identity["Account"]
        assumed_arn = identity["Arn"]

        result = [
            f" Cross-account access verified successfully!",
            f"\nTarget Account: {account_id}",
            f"Role Name: {role_name}",
            f"Assumed Account: {assumed_account}",
            f"Assumed Role ARN: {assumed_arn}",
            f"\nYou can now use this account configuration with other monitoring tools.",
        ]

        logger.info(
            f"Successfully verified cross-account access for account {account_id}"
        )
        return "\n".join(result)

    except Exception as e:
        error_msg = f"L Failed to setup cross-account access: {str(e)}"
        logger.error(error_msg)
        return error_msg