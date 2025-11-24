# Agent Evaluation Framework

This directory contains the evaluation framework for the Self-Healing Agent (Log Curator).

## Overview

The evaluation script runs the agent through a comprehensive set of synthetic questions to assess its performance across different categories and question types.

## Prerequisites

- Python 3.11+
- All agent dependencies installed
- AWS credentials configured for Amazon Bedrock
- LangSmith API key configured
- (Optional) TAVILY_API_KEY for internet search functionality
- (Optional) LANGSMITH_SESSION_ID for analyzing specific traces

## Dataset

The evaluation uses synthetic questions from `test_data/langsmith_synthetic_questions.json`, covering:

1. Project Overview Questions - Health, performance, and activity metrics
2. Project Detail Questions - Specific project analysis
3. Trace Detail Questions - Individual trace inspection
4. Comparative Questions - Cross-project comparisons
5. Actionable Questions - Recommendations and next steps
6. Complex Multi-Step Questions - Multi-faceted analysis
7. Metadata Questions - Configuration and settings
8. Natural Language Variations - Different phrasings

## Usage

### Basic Evaluation

Run the full evaluation suite:

```bash
cd evaluate_adaptive
python run_evaluation.py
```

### With Specific Session

To analyze a specific LangSmith session:

```bash
export LANGSMITH_SESSION_ID="your-session-id"
python run_evaluation.py
```

### With All Environment Variables

```bash
export LANGSMITH_API_KEY="your-api-key"
export LANGSMITH_SESSION_ID="your-session-id"
export TAVILY_API_KEY="your-tavily-key"
python run_evaluation.py
```

## Output

The script generates timestamped results in the `evaluation_results/` directory:

- `results_TIMESTAMP.json` - Machine-readable detailed results
- `report_TIMESTAMP.txt` - Human-readable summary and analysis
- `results_latest.json` - Latest results (for easy access)
- `report_latest.txt` - Latest report (for easy access)

### Results Include

- Success/failure rate
- Execution time statistics
- Category-wise breakdown
- Individual question results with full responses
- Error details for failed questions

## Performance Notes

Running the full evaluation suite may take a long time depending on:

- Number of questions (currently 100+ questions)
- Agent complexity and tool usage
- Network latency for API calls
- Whether deep research is triggered

Expect 10-30 seconds per question on average, potentially longer for complex questions.

## Interpreting Results

### Success Metrics

- Success Rate: Percentage of questions that completed without errors
- Execution Time: Time taken per question (indicates performance)
- Category Performance: Success rates by question category

### Common Failure Modes

- API timeouts or rate limits
- Missing environment variables
- Network connectivity issues
- Invalid session IDs

## Development Workflow

Before committing changes:

```bash
# Format and lint
uv run ruff check --fix . && uv run ruff format .

# Run evaluation
python run_evaluation.py
```

## Configuration

Edit `eval.yaml` to configure evaluation parameters (currently minimal, can be extended).
