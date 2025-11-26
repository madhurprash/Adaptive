"""
MLflow tracing utilities for LLM observability.

This module provides utilities for:
1. Initializing MLflow with Databricks
2. Creating trace contexts for agent operations
3. Logging agent runs and evaluations
4. Tracking metrics and metadata
"""

import os
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass

import mlflow
from mlflow.entities import SpanType

logger = logging.getLogger(__name__)


@dataclass
class MLflowConfig:
    """Configuration for MLflow tracing."""

    tracking_uri: str = "databricks"
    experiment_name: str = "lambda-autotuner-agent"
    databricks_host: Optional[str] = None
    databricks_token: Optional[str] = None
    enable_tracing: bool = True


def _get_config_from_env() -> MLflowConfig:
    """
    Load MLflow configuration from environment variables.

    Returns:
        MLflowConfig with values from environment or defaults
    """
    return MLflowConfig(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "databricks"),
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "lambda-autotuner-agent"),
        databricks_host=os.getenv("DATABRICKS_HOST"),
        databricks_token=os.getenv("DATABRICKS_TOKEN"),
        enable_tracing=os.getenv("MLFLOW_ENABLE_TRACING", "true").lower() == "true",
    )


def initialize_mlflow(
    config: Optional[MLflowConfig] = None,
) -> bool:
    """
    Initialize MLflow tracing with Databricks.

    Args:
        config: Optional MLflowConfig. If None, loads from environment variables.

    Returns:
        True if initialization successful, False otherwise
    """
    if config is None:
        config = _get_config_from_env()

    if not config.enable_tracing:
        logger.info("MLflow tracing is disabled")
        return False

    try:
        # Set tracking URI
        mlflow.set_tracking_uri(config.tracking_uri)
        logger.info(f"MLflow tracking URI set to: {config.tracking_uri}")

        # Configure Databricks credentials if provided
        if config.databricks_host and config.databricks_token:
            os.environ["DATABRICKS_HOST"] = config.databricks_host
            os.environ["DATABRICKS_TOKEN"] = config.databricks_token
            logger.info(f"Databricks credentials configured for: {config.databricks_host}")

        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(config.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(config.experiment_name)
                logger.info(f"Created new MLflow experiment: {config.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {config.experiment_name}")

            mlflow.set_experiment(config.experiment_name)

        except Exception as e:
            logger.warning(f"Could not set experiment {config.experiment_name}: {e}")
            logger.info("MLflow will use default experiment")

        # Enable autologging for LangChain with proper configuration
        # This will automatically trace LangGraph agent executions
        try:
            mlflow.langchain.autolog(
                log_models=True,
                log_input_examples=True,
                log_model_signatures=True,
                extra_tags={"framework": "langchain", "agent_type": "langgraph"}
            )
            logger.info("MLflow LangChain autologging enabled with LangGraph support")
        except Exception as e:
            logger.warning(f"Could not enable LangChain autologging: {e}")

        logger.info("✅ MLflow tracing initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize MLflow tracing: {e}")
        logger.error(
            "Check DATABRICKS_HOST and DATABRICKS_TOKEN environment variables. "
            "Get your token from: https://docs.databricks.com/en/dev-tools/auth.html#personal-access-tokens"
        )
        return False


@contextmanager
def create_trace_context(
    name: str,
    span_type: str = "AGENT",
    inputs: Optional[Dict[str, Any]] = None,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Create a traced context for an operation.

    Usage:
        with create_trace_context(
            name="evaluate_question",
            span_type="AGENT",
            inputs={"question": "What is..."},
            attributes={"category": "metrics"}
        ) as span:
            # Your code here
            result = do_work()

    Args:
        name: Name of the operation
        span_type: Type of span (AGENT, CHAIN, TOOL, etc.)
        inputs: Input data for the operation
        attributes: Additional metadata to attach

    Yields:
        The active span context
    """
    try:
        # Map string span type to MLflow SpanType
        span_type_map = {
            "AGENT": SpanType.AGENT,
            "CHAIN": SpanType.CHAIN,
            "TOOL": SpanType.TOOL,
            "LLM": SpanType.LLM,
            "RETRIEVER": SpanType.RETRIEVER,
            "EMBEDDING": SpanType.EMBEDDING,
        }

        mlflow_span_type = span_type_map.get(span_type.upper(), SpanType.AGENT)

        # Start a trace with span
        with mlflow.start_span(name=name, span_type=mlflow_span_type) as span:
            # Set inputs as attributes instead of using log_inputs
            # log_inputs expects a specific dataset format, not a simple dict
            if inputs:
                for key, value in inputs.items():
                    try:
                        span.set_attribute(f"input.{key}", value)
                    except Exception as e:
                        logger.warning(f"Failed to set input attribute {key}: {e}")

            # Set attributes if provided
            if attributes:
                for key, value in attributes.items():
                    try:
                        span.set_attribute(key, value)
                    except Exception as e:
                        logger.warning(f"Failed to set attribute {key}: {e}")

            logger.debug(f"Started MLflow span: {name} (type: {span_type})")

            yield span

            logger.debug(f"Completed MLflow span: {name}")

    except Exception as e:
        logger.warning(f"Error in MLflow trace context {name}: {e}")
        # Yield a dummy context that does nothing
        yield None


def log_agent_run(
    question: str,
    response: str,
    execution_time: float,
    success: bool,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an agent run to MLflow.

    This function logs metrics and metadata to the current active span.
    It should be called within a traced context (e.g., create_trace_context).

    Args:
        question: The input question
        response: The agent's response
        execution_time: Time taken in seconds
        success: Whether the run was successful
        metadata: Additional metadata to log
    """
    try:
        # Get the active span to set attributes
        active_span = mlflow.active_span()

        if active_span is None:
            logger.warning("No active span to log agent run - skipping MLflow logging")
            return

        # Set success as attribute
        active_span.set_attribute("success", success)

        # Set metrics as attributes
        active_span.set_attribute("execution_time_seconds", execution_time)
        active_span.set_attribute("response_length", len(response))

        # Set metadata as attributes
        if metadata:
            for key, value in metadata.items():
                if value is not None:
                    active_span.set_attribute(key, str(value))

        logger.debug(f"Logged agent run to MLflow (success: {success})")

    except Exception as e:
        logger.warning(f"Failed to log agent run to MLflow: {e}")


def log_question_evaluation(
    question_id: str,
    category: str,
    difficulty: str,
    question: str,
    response: str,
    execution_time: float,
    success: bool,
    error: Optional[str] = None,
) -> None:
    """
    Log a synthetic question evaluation to MLflow.

    Args:
        question_id: Unique identifier for the question
        category: Question category
        difficulty: Difficulty level
        question: The question text
        response: The agent's response
        execution_time: Execution time in seconds
        success: Whether evaluation was successful
        error: Error message if any
    """
    try:
        with create_trace_context(
            name=f"question_{question_id}",
            span_type="AGENT",
            inputs={"question": question},
            attributes={
                "question_id": question_id,
                "category": category,
                "difficulty": difficulty,
            },
        ):
            # Log as a run
            log_agent_run(
                question=question,
                response=response,
                execution_time=execution_time,
                success=success,
                metadata={
                    "question.id": question_id,
                    "question.category": category,
                    "question.difficulty": difficulty,
                    "error": error if error else None,
                },
            )

        logger.debug(f"Logged question evaluation {question_id} to MLflow")

    except Exception as e:
        logger.warning(f"Failed to log question evaluation {question_id}: {e}")


def annotate_span(
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Annotate the current active span with metadata, tags, and metrics.

    This function adds additional information to the currently active MLflow span.
    It should be called within a traced context (e.g., inside a function decorated
    with @track_agent_execution or within a create_trace_context block).

    Args:
        metadata: Dictionary of metadata to attach as attributes
        tags: Dictionary of tags to set
        metrics: Dictionary of metrics to log

    Example:
        @track_agent_execution(name="my_agent")
        def run_agent(message: str):
            annotate_span(
                metadata={"session_id": "123", "user_id": "456"},
                tags={"environment": "production"},
                metrics={"response_time": 1.5}
            )
            return process(message)
    """
    try:
        # Get the active span
        active_span = mlflow.active_span()

        if active_span is None:
            logger.debug("No active span to annotate")
            return

        # Set metadata as attributes
        if metadata:
            for key, value in metadata.items():
                try:
                    active_span.set_attribute(key, value)
                except Exception as e:
                    logger.warning(f"Failed to set attribute {key}: {e}")

        # Set tags as attributes (prefixed with 'tag.')
        if tags:
            for key, value in tags.items():
                try:
                    active_span.set_attribute(f"tag.{key}", str(value))
                except Exception as e:
                    logger.warning(f"Failed to set tag {key}: {e}")

        # Set metrics as attributes (prefixed with 'metric.')
        if metrics:
            for key, value in metrics.items():
                try:
                    active_span.set_attribute(f"metric.{key}", float(value))
                except Exception as e:
                    logger.warning(f"Failed to log metric {key}: {e}")

        logger.debug(
            f"Annotated span with "
            f"{len(metadata or {})} metadata items, "
            f"{len(tags or {})} tags, "
            f"{len(metrics or {})} metrics"
        )

    except Exception as e:
        logger.warning(f"Error annotating span: {e}")


def track_agent_execution(
    name: Optional[str] = None,
    model_name: Optional[str] = None,
    model_provider: Optional[str] = None,
) -> Callable:
    """
    Decorator to track agent execution with MLflow tracing.

    Usage:
        @track_agent_execution(
            name="lambda_autotuner",
            model_name="claude-sonnet-4",
            model_provider="bedrock"
        )
        def run_agent(question: str) -> dict:
            # Agent logic here
            return result

    Args:
        name: Name of the agent/workflow (defaults to function name)
        model_name: Name of the model being used
        model_provider: Provider of the model (e.g., "bedrock", "openai")

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            span_name = name or func.__name__

            # Extract question if available
            question = None
            if args and len(args) > 0:
                if isinstance(args[0], str):
                    question = args[0]
                elif isinstance(args[0], dict) and "question" in args[0]:
                    question = args[0]["question"]

            if not question and "user_message" in kwargs:
                question = kwargs["user_message"]

            # Build attributes
            attributes = {}
            if model_name:
                attributes["model_name"] = model_name
            if model_provider:
                attributes["model_provider"] = model_provider

            # Create trace context
            with create_trace_context(
                name=span_name,
                span_type="AGENT",
                inputs={"question": question} if question else None,
                attributes=attributes if attributes else None,
            ):
                # Execute the function
                result = func(*args, **kwargs)

                # Log output as span attributes
                try:
                    active_span = mlflow.active_span()
                    if active_span and isinstance(result, dict):
                        active_span.set_attribute("result_keys", ",".join(result.keys()))
                except Exception as e:
                    logger.warning(f"Failed to log result metadata: {e}")

                return result

        return wrapper

    return decorator
