"""
This is the self questioning module that is implemented based
on the research paper: https://arxiv.org/pdf/2511.10395v1

The insights agent current does the following:

┌────────────────────────────────────────────────────────────┐
│                    Existing System                          │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐         ┌──────────────┐                │
│  │  Insights    │────────▶│  AgentCore   │                │
│  │  Agent       │  store  │  Memory      │                │
│  │  (MCP Tools) │◀────────│  (Traces)    │  retrieve      │
│  └──────────────┘         └──────────────┘                │
│         │                        │                          │
│         │ insights               │ execution history        │
│         ▼                        ▼                          │
└─────────────────────────────────────────────────────────────┘
         │                         │
         └──────────┬──────────────┘
                    ▼
┌────────────────────────────────────────────────────────────┐
│              Self-Questioning Module (NEW)                  │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  1. Capability Gap Analyzer                          │ │
│  │     - Parse insights for failure patterns            │ │
│  │     - Identify underperforming areas                 │ │
│  │     - Score capability gaps by severity              │ │
│  └──────────────────────────────────────────────────────┘ │
│                         │                                   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  2. Task Generator (LLM-based)                       │ │
│  │     - Generate tasks targeting gaps                  │ │
│  │     - Control task difficulty (0.0-1.0)              │ │
│  │     - Specify expected tools and success criteria    │ │
│  └──────────────────────────────────────────────────────┘ │
│                         │                                   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  3. Diversity Enforcer                               │ │
│  │     - Compute task embeddings                        │ │
│  │     - Check similarity to existing tasks             │ │
│  │     - Reject tasks below diversity threshold         │ │
│  └──────────────────────────────────────────────────────┘ │
│                         │                                   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  4. Task Store                                       │ │
│  │     - Store generated tasks in AgentCore Memory      │ │
│  │     - Namespace: "synthetic_tasks"                   │ │
│  │     - Metadata: difficulty, type, generation_time    │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
         ┌──────────────────────────────┐
         │  Evaluation Framework        │
         │  (uses generated tasks)      │
         └──────────────────────────────┘
"""

# import libraries
import json
import uuid
import logging
import numpy as np
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from langchain.agents import create_agent
from typing import List, Dict, Any, Optional
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
# these are the file system tools that the self questioning
# agent will have access to to analyze the current repository and then
# generate high quality synthetic data that can then be used to analyze where
# the agent is going wrong and where it can go wrong, and how to fix it
from agent_tools.file_tools import (
            read_file,
            write_file,
            list_directory,
            search_files
        )
# Import config and prompt loading utilities
from utils import load_config, load_system_prompt

# set a logger
logger = logging.getLogger(__name__)

class SyntheticTask(BaseModel):
    """
    This is the generated task for agent evaluation
    
    This model represents a single synthetic task created by the self-questioning
    module. Each task includes the metadata about its difficulty, type, expected behavior, 
    and the success criteria
    """
    # this is the task id
    task_id: str = Field(
        description="Unique task identifier (UUID)"
    )
    # this is the description of the task
    task_description: str = Field(
        description="Human-readable task description per agent",
        min_length=10
    )
    # this is the type of the task
    task_type: str = Field(
        description="Task category",
        pattern="^(exploration|edge_case|optimization|regression)$",
    )
    # this is the difficulty level of the synthetic task
    difficulty_level: float = Field(
        ge=0.0,
        le=1.0, 
        description="Estimated difficulty (0=easy, 1=hard)"
    )
    # these are the expected tools that the agent should use to complete
    # the task
    expected_tools: List[str] = Field(
        default_factory=list, 
        description="Tools the agent should use to complete the task"
    )
    # this is the sunccess criteria to evaluate the task success
    success_criteria: Dict[str, Any] = Field(
        description="Criteria to evaluate the task success", 
        examples=[{
            "max_execution_time_ms": 5000,
            "error_rate_threshold": 0.05,
            "required_outputs": ["metrics_analyzed", "config_updated"],
        }]
    )
    # generation metadata: this is the context about how the task was generated
    generation_metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Context about how the task was generated"
    )
    embedding: Optional[np.ndarray] = Field(
        default=None, 
        description = "Task embedding for diversity checking (not serialized)", 
        exclude=True
    )
    class Config:
        """
        Pydantic configuration
        """
        arbitrary_types_allowed = True # This is for the numpy arrays

class CapabilityGap(BaseModel):
    """
    Identified weakness in the agent architecture capabilities.
    
    This represents an area where the agent struggles or lacks coverage, used to target
    task generation
    """
    gap_id: str
    gap_description: str
    severity: float = Field(ge=0.0, le=1.0)
    failure_count: int = Field(ge=0)
    example_failures: List[str] = Field(default_factory=list)
    suggested_task_types: List[str] = Field(default_factory=list)

class SelfQuestioningModule:
    """
    This is the autonomous task generation for the agent training process

    This module implements the self questioning mechanism, enabling agents to generate
    diverse synthetic tasks by analyzing their own execution history and identifying capability
    gaps.

    Key features:
    - Capability gap analysis from insights
    - LLM-Based task generation
    - Embedding-based diversity enforcement
    - Difficulty calliberation
    - Task metadata tracking

    Integration:
    - Consumes insights from the insights_agent (All Observability frameworks)
    - Queries AgentCore memory for execution history
    - Generates tasks for evaluate_adaptive/run_evaluation.py
    """

    def __init__(
        self,
        llm: ChatBedrockConverse,
        embedding_model: Any,
        agentcore_memory: Any,
        config_dict: Dict[str, Any],
    ):
        """
        Initialize the Self-Questioning Module.

        Args:
            llm: Language model for task generation
            embedding_model: Model for computing task embeddings
            agentcore_memory: AgentCore memory interface for storing/retrieving tasks
            config_dict: Configuration dictionary from main config.yaml
        """
        # Use provided configuration dictionary
        logger.info("Initializing SelfQuestioningModule with provided configuration")
        self.config = config_dict

        if not self.config:
            raise ValueError("Configuration dictionary is empty")

        # Get sub-configurations
        task_gen_config = self.config.get('task_generator', {})
        diversity_config = self.config.get('diversity_enforcer', {})
        gap_config = self.config.get('capability_gap_analyzer', {})

        # Set LLM and other services
        self.llm = llm
        self.embedding_model = embedding_model
        self.agentcore_memory = agentcore_memory

        # Load prompt templates from the self_questioning module directory
        logger.info("Loading prompt templates...")
        prompt_base_path = "evolution/self_questioning/prompt_templates"
        self.gap_analysis_prompt_template = load_system_prompt(
            f"{prompt_base_path}/capability_gap_analysis_prompt.txt"
        )
        self.task_generation_prompt_template = load_system_prompt(
            f"{prompt_base_path}/task_generation_prompt.txt"
        )
        self.exploration_prompt_template = load_system_prompt(
            f"{prompt_base_path}/exploration_prompt.txt"
        )

        # Set parameters from main config
        self.diversity_threshold = diversity_config.get('min_similarity_threshold', 0.85)
        # Use task difficulty distribution from config
        difficulty_dist = task_gen_config.get('difficulty_distribution', {})
        self.min_difficulty = min(difficulty_dist.get('easy', 0.3), difficulty_dist.get('medium', 0.5))
        self.max_difficulty = max(difficulty_dist.get('hard', 0.2), difficulty_dist.get('medium', 0.5))
        self.tasks_per_gap = task_gen_config.get('max_tasks_per_session', 5)

        # Exploration configuration (enable by default if not specified)
        self.exploration_enabled = True
        self.num_exploration_tasks = 3
        self.default_agent_codebase_path = "agents_tested/"

        # Initialize file system tools for exploration agent
        self.file_tools = self._create_file_tools()

        # Create exploration agent with file system access
        self.exploration_agent = self._create_exploration_agent()

        # Store task embeddings for diversity checking
        self.task_embeddings_store: List[np.ndarray] = []
        self.task_metadata_store: List[Dict[str, Any]] = []

        logger.info(
            f"✅ SelfQuestioningModule initialized with:\n"
            f"   - Diversity threshold: {self.diversity_threshold}\n"
            f"   - Difficulty range: [{self.min_difficulty}, {self.max_difficulty}]\n"
            f"   - Tasks per gap: {self.tasks_per_gap}\n"
            f"   - Exploration enabled: {self.exploration_enabled}\n"
            f"   - File tools: {len(self.file_tools)}\n"
        )

    def _create_file_tools(self) -> List:
        """
        Create file system tools for codebase exploration.

        Returns:
            List of file operation tools
        """
        try:
            logger.info("Creating file system tools for task generation agent...")
            file_tools = [read_file, write_file, list_directory, search_files]
            logger.info(
                f"Created {len(file_tools)} file system tools: "
                f"{', '.join([tool.name for tool in file_tools])}"
            )
            return file_tools
        except Exception as e:
            logger.error(f"An error occurred while instantiating the file system tools: {e}")
            raise

    def _create_exploration_agent(self):
        """
        Create an exploration agent for analyzing codebases and generating exploration tasks.

        This agent uses file system tools to explore codebases and identify
        unexplored areas that need testing.

        Returns:
            Configured exploration agent with file system tools
        """
        # Use the exploration prompt template as the system prompt
        # Note: We'll format it with actual values when invoking the agent
        system_prompt = self.exploration_prompt_template
        logger.info("Building exploration agent with file system tools...")
        agent = create_agent(
            model=self.llm,
            tools=self.file_tools,
            system_prompt=system_prompt,
        )
        logger.info("✅ Exploration agent created successfully")
        return agent

    async def generate_tasks(
        self,
        environment_context: Dict[str, Any],
        agent_execution_history: List[Dict],
        max_tasks: Optional[int] = None,
    ) -> List[SyntheticTask]:
        """
        Generate diverse synthetic tasks based on environment exploration and capability gaps.

        Args:
            environment_context: Information about environment (agent_name, codebase_path, etc.)
            agent_execution_history: Past agent executions with success/failure info
            max_tasks: Maximum number of tasks to generate (uses config default if None)

        Returns:
            List of diverse synthetic tasks with varying difficulty

        Raises:
            ValueError: If environment_context is missing required fields
        """
        logger.info("🔄 Starting task generation process...")

        if max_tasks is None:
            max_tasks = self.tasks_per_gap * 10

        generated_tasks: List[SyntheticTask] = []

        try:
            logger.info("Step 1: Analyzing capability gaps...")
            capability_gaps = self._analyze_capability_gaps(agent_execution_history)
            logger.info(f"Identified {len(capability_gaps)} capability gaps")

            logger.info("Step 2: Generating tasks for capability gaps...")
            for gap in capability_gaps:
                gap_tasks = self._generate_tasks_for_gap(gap)
                for task in gap_tasks:
                    if self._ensure_task_diversity(task):
                        generated_tasks.append(task)
                        logger.debug(f"Added gap-targeted task: {task.task_id}")

                        if len(generated_tasks) >= max_tasks:
                            break

                if len(generated_tasks) >= max_tasks:
                    break

            if self.exploration_enabled and len(generated_tasks) < max_tasks:
                logger.info("Step 3: Generating exploration tasks...")
                codebase_path = environment_context.get('codebase_path', self.default_agent_codebase_path)
                exploration_tasks = self._generate_exploration_tasks(codebase_path)

                for task in exploration_tasks:
                    if self._ensure_task_diversity(task):
                        generated_tasks.append(task)
                        logger.debug(f"Added exploration task: {task.task_id}")

                        if len(generated_tasks) >= max_tasks:
                            break

            logger.info("Step 4: Storing generated tasks...")
            for task in generated_tasks:
                task_id = self._store_task_in_memory(task)
                logger.debug(f"Stored task {task_id} in memory")

            logger.info(
                f"✅ Task generation complete: {len(generated_tasks)} tasks generated\n"
                f"   - Task types: {self._count_task_types(generated_tasks)}\n"
                f"   - Difficulty range: [{self._min_difficulty(generated_tasks):.2f}, "
                f"{self._max_difficulty(generated_tasks):.2f}]"
            )

            return generated_tasks

        except Exception as e:
            logger.error(f"Error during task generation: {e}", exc_info=True)
            raise

    def _analyze_capability_gaps(
        self,
        execution_history: List[Dict]
    ) -> List[CapabilityGap]:
        """
        Identify areas where agent struggles by analyzing execution history.

        Args:
            execution_history: Past agent executions with success/failure metadata

        Returns:
            List of identified capability gaps sorted by severity
        """
        logger.info("Analyzing capability gaps from execution history...")

        if not execution_history:
            logger.warning("No execution history provided, returning empty gaps list")
            return []

        gaps: List[CapabilityGap] = []

        try:
            failure_patterns = self._aggregate_failure_patterns(execution_history)

            gap_analysis_prompt = self.gap_analysis_prompt_template.format(
                execution_history=json.dumps(failure_patterns, indent=2, default=str),
                num_gaps=5
            )

            logger.debug("Invoking LLM for gap analysis...")
            response = self.llm.invoke([HumanMessage(content=gap_analysis_prompt)])

            gaps = self._parse_gap_analysis_response(response.content, failure_patterns)
            gaps.sort(key=lambda g: g.severity, reverse=True)

            logger.info(f"Identified {len(gaps)} capability gaps")
            for gap in gaps:
                logger.debug(
                    f"  - {gap.gap_description} "
                    f"(severity: {gap.severity:.2f}, failures: {gap.failure_count})"
                )

            return gaps

        except Exception as e:
            logger.error(f"Error analyzing capability gaps: {e}", exc_info=True)
            return []

    def _aggregate_failure_patterns(
        self,
        execution_history: List[Dict]
    ) -> Dict[str, Any]:
        """Aggregate failure patterns from execution history."""
        patterns = {
            "total_executions": len(execution_history),
            "failures": [],
            "common_errors": {},
            "tool_failures": {},
            "timeout_count": 0,
        }

        for execution in execution_history:
            if not execution.get("success", False):
                failure_info = {
                    "task": execution.get("task", "unknown"),
                    "error": execution.get("error", ""),
                    "tools_used": execution.get("tools_used", []),
                    "duration_ms": execution.get("duration_ms", 0),
                }
                patterns["failures"].append(failure_info)

                error_type = self._classify_error(execution.get("error", ""))
                patterns["common_errors"][error_type] = patterns["common_errors"].get(error_type, 0) + 1

                for tool in execution.get("failed_tools", []):
                    patterns["tool_failures"][tool] = patterns["tool_failures"].get(tool, 0) + 1

                if "timeout" in execution.get("error", "").lower():
                    patterns["timeout_count"] += 1

        return patterns

    def _classify_error(
        self,
        error_message: str
    ) -> str:
        """Classify error message into category."""
        error_lower = error_message.lower()

        if "timeout" in error_lower:
            return "timeout"
        elif "permission" in error_lower or "access" in error_lower:
            return "permission_error"
        elif "not found" in error_lower or "missing" in error_lower:
            return "resource_not_found"
        elif "invalid" in error_lower or "malformed" in error_lower:
            return "invalid_input"
        elif "rate limit" in error_lower or "throttle" in error_lower:
            return "rate_limit"
        else:
            return "unknown_error"

    def _parse_gap_analysis_response(
        self,
        llm_response: str,
        failure_patterns: Dict[str, Any]
    ) -> List[CapabilityGap]:
        """Parse LLM response into CapabilityGap objects."""
        gaps: List[CapabilityGap] = []

        try:
            if llm_response.strip().startswith("{") or llm_response.strip().startswith("["):
                gap_data = json.loads(llm_response)
                if isinstance(gap_data, dict):
                    gap_data = gap_data.get("gaps", [])

                for item in gap_data:
                    gap = CapabilityGap(
                        gap_id=item.get("gap_id", str(uuid.uuid4())),
                        gap_description=item.get("description", ""),
                        severity=float(item.get("severity", 0.5)),
                        failure_count=int(item.get("failure_count", 0)),
                        example_failures=item.get("example_failures", []),
                        suggested_task_types=item.get("suggested_task_types", [])
                    )
                    gaps.append(gap)
            else:
                logger.warning("LLM response not in JSON format, using fallback parser")
                gaps = self._parse_gaps_from_text(llm_response, failure_patterns)

        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response, using fallback")
            gaps = self._parse_gaps_from_text(llm_response, failure_patterns)

        return gaps

    def _parse_gaps_from_text(
        self,
        text: str,
        failure_patterns: Dict[str, Any]
    ) -> List[CapabilityGap]:
        """Fallback parser for non-JSON LLM responses."""
        gaps: List[CapabilityGap] = []
        lines = text.split("\n")
        current_gap_text = []

        for line in lines:
            line_stripped = line.strip()
            if line_stripped and (
                line_stripped[0].isdigit() or
                line_stripped.startswith("-") or
                line_stripped.startswith("•")
            ):
                if current_gap_text:
                    gap_desc = " ".join(current_gap_text)
                    gaps.append(CapabilityGap(
                        gap_id=str(uuid.uuid4()),
                        gap_description=gap_desc,
                        severity=0.7,
                        failure_count=len(failure_patterns.get("failures", [])),
                        example_failures=[],
                        suggested_task_types=["edge_case", "optimization"]
                    ))
                    current_gap_text = []

                current_gap_text.append(line_stripped.lstrip("0123456789.-• "))
            elif current_gap_text and line_stripped:
                current_gap_text.append(line_stripped)

        if current_gap_text:
            gap_desc = " ".join(current_gap_text)
            gaps.append(CapabilityGap(
                gap_id=str(uuid.uuid4()),
                gap_description=gap_desc,
                severity=0.7,
                failure_count=len(failure_patterns.get("failures", [])),
                example_failures=[],
                suggested_task_types=["edge_case", "optimization"]
            ))

        return gaps

    def _generate_tasks_for_gap(
        self,
        gap: CapabilityGap
    ) -> List[SyntheticTask]:
        """Generate synthetic tasks targeting a specific capability gap."""
        logger.debug(f"Generating tasks for gap: {gap.gap_description}")

        tasks: List[SyntheticTask] = []

        try:
            task_prompt = self.task_generation_prompt_template.format(
                gap_description=gap.gap_description,
                gap_severity=gap.severity,
                example_failures=json.dumps(gap.example_failures[:3], indent=2),
                num_tasks=self.tasks_per_gap,
                min_difficulty=self.min_difficulty,
                max_difficulty=self.max_difficulty
            )

            logger.debug("Invoking LLM for task generation...")
            response = self.llm.invoke([HumanMessage(content=task_prompt)])

            tasks = self._parse_task_generation_response(response.content, gap)

            logger.debug(f"Generated {len(tasks)} tasks for gap")

        except Exception as e:
            logger.error(f"Error generating tasks for gap: {e}", exc_info=True)

        return tasks

    def _parse_task_generation_response(
        self,
        llm_response: str,
        gap: CapabilityGap
    ) -> List[SyntheticTask]:
        """Parse LLM task generation response into SyntheticTask objects."""
        tasks: List[SyntheticTask] = []

        try:
            if llm_response.strip().startswith("{") or llm_response.strip().startswith("["):
                task_data = json.loads(llm_response)
                if isinstance(task_data, dict):
                    task_data = task_data.get("tasks", [])

                for item in task_data:
                    task = SyntheticTask(
                        task_id=str(uuid.uuid4()),
                        task_description=item.get("description", ""),
                        task_type=item.get("type", "edge_case"),
                        difficulty_level=float(item.get("difficulty", gap.severity)),
                        expected_tools=item.get("expected_tools", []),
                        success_criteria=item.get("success_criteria", {}),
                        generation_metadata={
                            "gap_id": gap.gap_id,
                            "gap_description": gap.gap_description,
                            "generation_time": datetime.now(timezone.utc).isoformat(),
                            "llm_model": str(self.llm.model_id) if hasattr(self.llm, 'model_id') else "unknown"
                        }
                    )
                    tasks.append(task)
            else:
                logger.warning("Task generation response not in JSON, using fallback")
                lines = [l.strip() for l in llm_response.split("\n") if l.strip()]
                for line in lines[:self.tasks_per_gap]:
                    if len(line) > 10:
                        task = SyntheticTask(
                            task_id=str(uuid.uuid4()),
                            task_description=line.lstrip("0123456789.-• "),
                            task_type="edge_case",
                            difficulty_level=gap.severity,
                            expected_tools=[],
                            success_criteria={},
                            generation_metadata={
                                "gap_id": gap.gap_id,
                                "generation_time": datetime.now(timezone.utc).isoformat()
                            }
                        )
                        tasks.append(task)

        except json.JSONDecodeError:
            logger.error("Failed to parse task generation response as JSON")

        return tasks

    def _generate_exploration_tasks(
        self,
        agent_codebase_path: str
    ) -> List[SyntheticTask]:
        """Use exploration agent to analyze codebase and generate tasks."""
        logger.info(f"Generating exploration tasks for codebase: {agent_codebase_path}")

        tasks: List[SyntheticTask] = []

        try:
            exploration_query = (
                f"Analyze the codebase at {agent_codebase_path} and identify "
                f"{self.num_exploration_tasks} areas that need testing. "
                f"For each area, suggest a specific test scenario or edge case. "
                f"Focus on: untested code paths, complex logic, error handling, "
                f"and boundary conditions."
            )

            logger.debug("Invoking exploration agent...")
            result = self.exploration_agent.invoke({"input": exploration_query})

            exploration_findings = result.get("output", "")
            tasks = self._parse_exploration_findings(exploration_findings)

            logger.info(f"Generated {len(tasks)} exploration tasks")

        except Exception as e:
            logger.error(f"Error generating exploration tasks: {e}", exc_info=True)

        return tasks

    def _parse_exploration_findings(
        self,
        findings: str
    ) -> List[SyntheticTask]:
        """Parse exploration agent findings into SyntheticTask objects."""
        tasks: List[SyntheticTask] = []
        lines = [l.strip() for l in findings.split("\n") if l.strip()]

        for line in lines:
            if len(line) < 20 or line.endswith(":"):
                continue

            task = SyntheticTask(
                task_id=str(uuid.uuid4()),
                task_description=line.lstrip("0123456789.-• "),
                task_type="exploration",
                difficulty_level=(self.min_difficulty + self.max_difficulty) / 2,
                expected_tools=[],
                success_criteria={"exploration_based": True},
                generation_metadata={
                    "source": "exploration_agent",
                    "generation_time": datetime.now(timezone.utc).isoformat()
                }
            )
            tasks.append(task)

            if len(tasks) >= self.num_exploration_tasks:
                break

        return tasks

    def _ensure_task_diversity(
        self,
        new_task: SyntheticTask
    ) -> bool:
        """Check if new task is sufficiently different from existing tasks."""
        try:
            task_embedding = self._embed_task(new_task.task_description)

            for stored_embedding in self.task_embeddings_store:
                similarity = self._cosine_similarity(task_embedding, stored_embedding)

                if similarity > self.diversity_threshold:
                    logger.debug(
                        f"Task rejected due to similarity {similarity:.3f} > "
                        f"threshold {self.diversity_threshold}"
                    )
                    return False

            self.task_embeddings_store.append(task_embedding)
            new_task.embedding = task_embedding

            return True

        except Exception as e:
            logger.error(f"Error checking task diversity: {e}")
            return True

    def _embed_task(
        self,
        task_description: str
    ) -> np.ndarray:
        """Generate embedding for task description."""
        try:
            embedding_response = self.embedding_model.embed_query(task_description)

            if isinstance(embedding_response, list):
                return np.array(embedding_response)
            elif isinstance(embedding_response, np.ndarray):
                return embedding_response
            else:
                return np.array(embedding_response)

        except Exception as e:
            logger.error(f"Error generating task embedding: {e}")
            return np.zeros(1024)

    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two vectors."""
        try:
            vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
            vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)

            similarity = np.dot(vec1_norm, vec2_norm)

            return float(similarity)

        except Exception as e:
            logger.error(f"Error computing cosine similarity: {e}")
            return 0.0

    def _store_task_in_memory(
        self,
        task: SyntheticTask
    ) -> str:
        """Store generated task in AgentCore Memory."""
        try:
            self.agentcore_memory.add_memory(
                content=task.task_description,
                metadata={
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "difficulty_level": task.difficulty_level,
                    "generation_time": task.generation_metadata.get(
                        "generation_time",
                        datetime.now(timezone.utc).isoformat()
                    ),
                    "expected_tools": json.dumps(task.expected_tools),
                    "success_criteria": json.dumps(task.success_criteria),
                },
                namespace="synthetic_tasks"
            )

            logger.debug(f"Stored task {task.task_id} in AgentCore Memory")
            return task.task_id

        except Exception as e:
            logger.error(f"Error storing task in memory: {e}")
            return task.task_id

    def _count_task_types(
        self,
        tasks: List[SyntheticTask]
    ) -> Dict[str, int]:
        """Count tasks by type."""
        counts: Dict[str, int] = {}
        for task in tasks:
            counts[task.task_type] = counts.get(task.task_type, 0) + 1
        return counts

    def _min_difficulty(
        self,
        tasks: List[SyntheticTask]
    ) -> float:
        """Get minimum difficulty from tasks."""
        if not tasks:
            return 0.0
        return min(task.difficulty_level for task in tasks)

    def _max_difficulty(
        self,
        tasks: List[SyntheticTask]
    ) -> float:
        """Get maximum difficulty from tasks."""
        if not tasks:
            return 0.0
        return max(task.difficulty_level for task in tasks)
