"""Script to run Lambda Autotuner Agent through all synthetic questions.

This script loads questions from the synthetic questions file and runs
the agent for each one, saving the responses.
"""

import json
import logging
import time
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage

from adaptive import graph


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _load_config() -> dict:
    """Load configuration from config.yaml.

    Returns:
        Configuration dictionary
    """
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_questions(file_path: Path) -> list:
    """Load questions from JSON file.

    Args:
        file_path: Path to questions JSON file

    Returns:
        List of question dictionaries
    """
    logger.info(f"Loading questions from {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    logger.info(f"Loaded {len(questions)} questions")

    return questions


def _run_question(
    question_data: dict,
    session_id: str,
) -> dict:
    """Run a single question through the agent.

    Args:
        question_data: Question dictionary
        session_id: Session ID for this question

    Returns:
        Result dictionary with response and metadata
    """
    question_id = question_data["id"]
    question = question_data["question"]
    category = question_data["category"]
    difficulty = question_data["difficulty"]

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Question ID: {question_id}")
    logger.info(f"Category: {category}")
    logger.info(f"Difficulty: {difficulty}")
    logger.info(f"Question: {question}")
    logger.info(f"{'=' * 80}\n")

    start_time = time.time()

    try:
        # Configure agent
        config = {"configurable": {"thread_id": session_id}}

        # Run agent
        result = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )

        # Extract response
        messages = result.get("messages", [])
        response = ""

        for message in reversed(messages):
            if hasattr(message, "content") and message.content:
                response = message.content
                break

        execution_time = time.time() - start_time

        logger.info(f"\nAgent Response:\n{response}\n")
        logger.info(f"Execution time: {execution_time:.2f}s\n")

        return {
            "question_id": question_id,
            "category": category,
            "difficulty": difficulty,
            "question": question,
            "response": response,
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
            "difficulty": difficulty,
            "question": question,
            "response": "",
            "execution_time_seconds": execution_time,
            "success": False,
            "error": str(e),
        }


def _save_results(
    results: list,
    output_dir: Path,
) -> None:
    """Save results to output directory.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results as JSON
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved results to {results_file}")

    # Save human-readable report
    report_file = output_dir / "report.txt"
    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Lambda Autotuner Agent - Synthetic Questions Run\n")
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

        f.write("=" * 80 + "\n")
        f.write("Individual Question Results\n")
        f.write("=" * 80 + "\n\n")

        for result in results:
            f.write(f"Question ID: {result['question_id']}\n")
            f.write(f"Category: {result['category']}\n")
            f.write(f"Difficulty: {result['difficulty']}\n")
            f.write(f"Status: {'✓ Success' if result['success'] else '✗ Failed'}\n")
            f.write(f"Execution Time: {result['execution_time_seconds']:.2f}s\n")
            f.write(f"\nQuestion:\n{result['question']}\n")
            f.write(f"\nResponse:\n{result['response']}\n")

            if result["error"]:
                f.write(f"\nError: {result['error']}\n")

            f.write("\n" + "-" * 80 + "\n\n")

    logger.info(f"Saved report to {report_file}")


def main() -> None:
    """Main function to run all synthetic questions."""
    logger.info("Starting Lambda Autotuner Agent synthetic questions run")

    # Load configuration
    config = _load_config()
    eval_config = config.get("evaluation", {})

    # Configure logging level
    if eval_config.get("debug_logging", False):
        logging.getLogger().setLevel(logging.DEBUG)

    # Load questions
    questions_file = Path(__file__).parent / eval_config.get(
        "synthetic_questions_fpath",
        "synthetic_questions/lambda_autotuner_questions.json",
    )
    questions = _load_questions(questions_file)

    # Get sample size
    sample_size = eval_config.get("sample_size", 0)
    if sample_size and 0 < sample_size < len(questions):
        import random

        # This is not for security/cryptographic purposes - nosec B311
        questions = random.sample(questions, sample_size)  # nosec B311
        logger.info(f"Sampled {sample_size} questions")

    logger.info(f"Running {len(questions)} questions")
    if len(questions) > 10:
        logger.warning("Processing many questions. This may take a long time.")

    # Run all questions
    start_time = time.time()
    results = []

    for i, question_data in enumerate(questions, 1):
        logger.info(f"\n{'#' * 80}")
        logger.info(f"Question {i}/{len(questions)}")
        logger.info(f"{'#' * 80}")

        session_id = f"synthetic-{question_data['id']}-{int(time.time())}"
        result = _run_question(question_data, session_id)
        results.append(result)

    total_time = time.time() - start_time

    # Display summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    logger.info(f"\n{'=' * 80}")
    logger.info("Run Complete")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total Questions: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")

    minutes = int(total_time // 60)
    seconds = total_time % 60
    if minutes > 0:
        logger.info(f"Total time: {minutes} minutes and {seconds:.1f} seconds")
    else:
        logger.info(f"Total time: {seconds:.1f} seconds")

    # Save results
    output_dir = Path(__file__).parent / eval_config.get(
        "results_output_dir",
        "evaluation_results",
    )
    _save_results(results, output_dir)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
