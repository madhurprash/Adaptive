"""Run evaluation of the Self-Healing Agent through all synthetic questions.

This script loads synthetic questions and runs the agent on each one,
logging results locally for analysis.
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from io import StringIO

# Add parent directory to path to import agent
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import run_agent


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


class _LogCapture:
    """Context manager to capture all logging output for a specific execution."""

    def __init__(self, output_file: Path):
        """Initialize log capture.

        Args:
            output_file: Path to save captured logs
        """
        self.output_file = output_file
        self.log_stream = StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setLevel(logging.DEBUG)
        self.handler.setFormatter(
            logging.Formatter(
                "%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s"
            )
        )

    def __enter__(self):
        """Start capturing logs."""
        # Add handler to root logger to capture all logs
        logging.getLogger().addHandler(self.handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop capturing and save logs."""
        # Remove handler
        logging.getLogger().removeHandler(self.handler)

        # Save captured logs to file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as f:
            f.write(self.log_stream.getvalue())

        # Close stream
        self.log_stream.close()


def _load_questions(
    file_path: Path,
) -> List[Dict[str, Any]]:
    """Load questions from JSON file and flatten into list.

    Args:
        file_path: Path to questions JSON file

    Returns:
        List of question dictionaries with metadata
    """
    logger.info(f"Loading questions from {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    questions = []
    question_id = 1

    # Process each category
    for category_name, category_data in data.items():
        logger.info(f"Processing category: {category_name}")

        if isinstance(category_data, list):
            # Handle list of subcategories
            for subcategory in category_data:
                subcategory_name = subcategory.get("category", "unknown")
                subcategory_questions = subcategory.get("questions", [])

                for question_text in subcategory_questions:
                    questions.append({
                        "id": f"q{question_id:03d}",
                        "category": category_name,
                        "subcategory": subcategory_name,
                        "question": question_text,
                        "metadata": subcategory.get("metadata", {}),
                    })
                    question_id += 1
        else:
            logger.warning(f"Unexpected format for category {category_name}")

    logger.info(f"Loaded {len(questions)} total questions")
    return questions


def _run_single_question(
    question_data: Dict[str, Any],
    thread_id: str,
    logs_dir: Path,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single question through the agent.

    Args:
        question_data: Question dictionary with metadata
        thread_id: Thread ID for this evaluation run
        logs_dir: Directory to save individual question logs
        session_id: Optional LangSmith session ID

    Returns:
        Result dictionary with response and metrics
    """
    question_id = question_data["id"]
    question = question_data["question"]
    category = question_data["category"]
    subcategory = question_data["subcategory"]

    # Create log file path for this question
    log_file = logs_dir / f"{question_id}_log.txt"

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Question ID: {question_id}")
    logger.info(f"Category: {category} / {subcategory}")
    logger.info(f"Question: {question}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"{'=' * 80}\n")

    start_time = time.time()

    try:
        # Capture all logs for this question execution
        with _LogCapture(log_file) as log_capture:
            # Run agent
            result = run_agent(
                user_question=question,
                session_id=session_id,
                thread_id=thread_id,
            )

            execution_time = time.time() - start_time

            # Log the final response and execution time BEFORE exiting the context
            logger.info(f"\n{'=' * 80}")
            logger.info("FINAL AGENT RESPONSE")
            logger.info(f"{'=' * 80}")
            logger.info(f"\nInsights:\n{result.get('insights', '')}")
            if result.get('research_results'):
                logger.info(f"\nResearch Results:\n{result.get('research_results', '')}")
            if result.get('output_file_path'):
                logger.info(f"\nOutput File Path: {result.get('output_file_path', '')}")
            logger.info(f"\nExecution time: {execution_time:.2f}s")
            logger.info(f"{'=' * 80}\n")

        # Log to console (outside capture context)
        logger.info(f"\nAgent Response (Insights):\n{result.get('insights', '')[:500]}...\n")
        logger.info(f"Execution time: {execution_time:.2f}s\n")
        logger.info(f"Logs saved to: {log_file}\n")

        return {
            "question_id": question_id,
            "category": category,
            "subcategory": subcategory,
            "question": question,
            "insights": result.get("insights", ""),
            "research_results": result.get("research_results", ""),
            "output_file_path": result.get("output_file_path", ""),
            "log_file_path": str(log_file),
            "execution_time_seconds": execution_time,
            "success": True,
            "error": None,
        }

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error running question {question_id}: {e}\n")

        return {
            "question_id": question_id,
            "category": category,
            "subcategory": subcategory,
            "question": question,
            "insights": "",
            "research_results": "",
            "output_file_path": "",
            "log_file_path": str(log_file),
            "execution_time_seconds": execution_time,
            "success": False,
            "error": str(e),
        }


def _save_results(
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save evaluation results to output directory.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save results as JSON
    results_file = output_dir / f"results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved results to {results_file}")

    # Save human-readable report
    report_file = output_dir / f"report_{timestamp}.txt"
    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Self-Healing Agent - Evaluation Results\n")
        f.write(f"Run timestamp: {timestamp}\n")
        f.write("=" * 80 + "\n\n")

        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        f.write(f"Total Questions: {len(results)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Success Rate: {successful / len(results) * 100:.1f}%\n\n")

        total_time = sum(r["execution_time_seconds"] for r in results)
        avg_time = total_time / len(results) if results else 0

        f.write(f"Total Execution Time: {total_time:.2f}s\n")
        f.write(f"Average Execution Time: {avg_time:.2f}s\n\n")

        # Category breakdown
        f.write("=" * 80 + "\n")
        f.write("Results by Category\n")
        f.write("=" * 80 + "\n\n")

        categories = {}
        for result in results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "successful": 0, "avg_time": 0}
            categories[cat]["total"] += 1
            if result["success"]:
                categories[cat]["successful"] += 1

        for cat, stats in categories.items():
            cat_results = [r for r in results if r["category"] == cat]
            avg_time = sum(r["execution_time_seconds"] for r in cat_results) / len(cat_results)
            success_rate = stats["successful"] / stats["total"] * 100

            f.write(f"{cat}:\n")
            f.write(f"  Questions: {stats['total']}\n")
            f.write(f"  Success Rate: {success_rate:.1f}%\n")
            f.write(f"  Avg Time: {avg_time:.2f}s\n\n")

        f.write("=" * 80 + "\n")
        f.write("Individual Question Results\n")
        f.write("=" * 80 + "\n\n")

        for result in results:
            f.write(f"Question ID: {result['question_id']}\n")
            f.write(f"Category: {result['category']} / {result['subcategory']}\n")
            f.write(f"Status: {'✓ Success' if result['success'] else '✗ Failed'}\n")
            f.write(f"Execution Time: {result['execution_time_seconds']:.2f}s\n")
            f.write(f"\nQuestion:\n{result['question']}\n")
            f.write(f"\nInsights:\n{result['insights']}\n")

            if result["research_results"]:
                f.write(f"\nResearch Results:\n{result['research_results'][:500]}...\n")

            if result["output_file_path"]:
                f.write(f"\nOutput File: {result['output_file_path']}\n")

            if result.get("log_file_path"):
                f.write(f"\nLog File: {result['log_file_path']}\n")

            if result["error"]:
                f.write(f"\nError: {result['error']}\n")

            f.write("\n" + "-" * 80 + "\n\n")

    logger.info(f"Saved report to {report_file}")

    # Also save latest results as "latest" for easy access
    latest_results = output_dir / "results_latest.json"
    with open(latest_results, "w") as f:
        json.dump(results, f, indent=2, default=str)

    latest_report = output_dir / "report_latest.txt"
    with open(latest_report, "w") as f:
        with open(report_file, "r") as src:
            f.write(src.read())


def main() -> None:
    """Main function to run evaluation."""
    logger.info("Starting Self-Healing Agent evaluation")

    # Load questions
    questions_file = Path(__file__).parent / "test_data" / "langsmith_synthetic_questions.json"
    questions = _load_questions(questions_file)

    logger.info(f"Running evaluation on {len(questions)} questions")
    logger.warning("Processing full dataset. This may take a long time.")

    # Generate unique thread_id for this evaluation run
    thread_id = f"eval-{int(time.time())}"
    logger.info(f"Using thread_id: {thread_id}")

    # Create logs directory for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "evaluation_results"
    logs_dir = output_dir / f"logs_{timestamp}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Individual question logs will be saved to: {logs_dir}")

    # Get session_id from environment if available
    import os
    session_id = os.getenv("LANGSMITH_SESSION_ID")
    if session_id:
        logger.info(f"Using session_id from environment: {session_id}")
    else:
        logger.warning("No LANGSMITH_SESSION_ID set. Agent may have limited data to analyze.")

    # Run all questions
    start_time = time.time()
    results = []

    for i, question_data in enumerate(questions, 1):
        logger.info(f"\n{'#' * 80}")
        logger.info(f"Question {i}/{len(questions)}")
        logger.info(f"{'#' * 80}")

        result = _run_single_question(
            question_data,
            thread_id=thread_id,
            logs_dir=logs_dir,
            session_id=session_id,
        )
        results.append(result)

    total_time = time.time() - start_time

    # Display summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    logger.info(f"\n{'=' * 80}")
    logger.info("Evaluation Complete")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total Questions: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {successful / len(results) * 100:.1f}%")

    minutes = int(total_time // 60)
    seconds = total_time % 60
    if minutes > 0:
        logger.info(f"Total time: {minutes} minutes and {seconds:.1f} seconds")
    else:
        logger.info(f"Total time: {seconds:.1f} seconds")

    # Save results
    _save_results(results, output_dir)

    logger.info("\nEvaluation complete!")
    logger.info(f"Results saved to: {output_dir}/")
    logger.info(f"Individual question logs saved to: {logs_dir}/")


if __name__ == "__main__":
    main()
