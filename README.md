# Self-Healing Agent

A multi-agent system for healing and evolving agentic applications over time. This tool analyzes observability traces from LangSmith and Langfuse to provide insights and automatically optimize your agent prompts.

## Features

- Analyze agent execution traces from multiple observability platforms (LangSmith, Langfuse)
- Generate insights about agent performance and behavior
- Automatically optimize system prompts based on observed patterns
- Interactive conversation mode with memory
- Human-in-the-loop (HITL) approval for prompt modifications
- Evolution engine for continuous agent improvement

## Prerequisites

- Python 3.12+ (recommended)
- AWS credentials configured (for Amazon Bedrock)
- Amazon Bedrock Guardrail with sensitive information filters
- Access to LangSmith or Langfuse for observability traces

## Installation

There are multiple ways to install the Self-Healing Agent:

### Method 1: Quick Install (Recommended)

Use the installation script (similar to Claude Code):

```bash
curl -fsSL https://raw.githubusercontent.com/yourusername/self-healing-agent/main/scripts/install.sh | bash
```

Or download and run locally:

```bash
git clone https://github.com/yourusername/self-healing-agent.git
cd self-healing-agent
bash scripts/install.sh
```

### Method 2: Using pip with uv

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/self-healing-agent.git
cd self-healing-agent

# Install in editable mode
uv pip install -e .
```

### Method 3: Using pip (traditional)

```bash
# Clone the repository
git clone https://github.com/yourusername/self-healing-agent.git
cd self-healing-agent

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

### Method 4: Development Install with uv

```bash
# Clone the repository
git clone https://github.com/yourusername/self-healing-agent.git
cd self-healing-agent

# Install with development dependencies
uv sync
```

After installation, you should be able to run either command:
- `evolve` - Main CLI command
- `self-healing-agent` - Alternative command name

## Quick Start

### 1. Configure Environment

Create a `.env` file in the project directory:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_PROFILE=default  # Optional

# LangSmith Configuration (if using LangSmith)
LANGSMITH_API_KEY=your_api_key_here
LANGSMITH_PROJECT=your_project_name

# Langfuse Configuration (if using Langfuse)
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

# AgentCore Memory (optional)
AGENTCORE_MEMORY_ROLE_ARN=arn:aws:iam::123456789:role/AgentCoreMemoryRole
```

### 2. Run the Agent

**Interactive Mode (Default):**

```bash
# Run with interactive mode
evolve run

# Or with session ID
evolve run --session-id "my-session-123"

# Or using environment variable
export LANGSMITH_SESSION_ID="my-session-123"
evolve run
```

**With Debug Logging:**

```bash
evolve run --debug
```

**Run as Background Daemon:**

```bash
# Check every hour (default)
evolve daemon

# Custom interval (every 30 minutes)
evolve daemon --interval 1800
```

### 3. Example Usage Flow

When you run `evolve run`, you'll be guided through:

1. **Platform Selection**: Choose between LangSmith or Langfuse
2. **Ask Questions**: Query your agent traces
3. **View Insights**: Get AI-generated insights about agent performance
4. **Evolution**: Optionally trigger prompt optimization
5. **Approve Changes**: Review and approve prompt modifications (HITL)

Example session:

```bash
$ evolve run

Self-Healing Agent - Unified Multi-Agent Workflow (Interactive Mode)
================================================================================
Workflow: Insights Agent -> Evolution Agent
  1. Insights Agent: Analyzes observability traces and generates insights
  2. Evolution Agent: Optimizes system prompts based on agent performance

= [PLATFORM SELECTION] Please select your observability platform:
   1. LangSmith
   2. Langfuse

Enter your choice (1 or 2): 1
 Selected: LangSmith

You: What are the main errors in my agent traces?

[Agent analyzes traces and provides insights...]

You: Can you optimize the prompts based on these insights?

[Evolution agent suggests changes and shows patch for approval...]
```

## Commands

### Main Commands

```bash
# Run agent once (interactive mode)
evolve run

# Run as background daemon
evolve daemon

# Show version
evolve version

# Show current configuration
evolve config
```

### Run Options

```bash
evolve run [OPTIONS]

Options:
  --config PATH    Path to configuration file
  --debug          Enable debug logging
  --help           Show help message
```

### Daemon Options

```bash
evolve daemon [OPTIONS]

Options:
  --config PATH       Path to configuration file
  --debug             Enable debug logging
  --interval SECONDS  Check interval in seconds (default: 3600)
  --help              Show help message
```

## Configuration

The agent uses a YAML configuration file located at `configs/config.yaml`. You can customize:

- Model IDs for different agents
- Inference parameters (temperature, max_tokens, etc.)
- Routing logic configuration
- Platform-specific settings
- AgentCore Memory settings

Example configuration structure:

```yaml
routing_configuration:
  router_model_id: "us.anthropic.claude-3-5-haiku-20241022-v1:0"
  inference_parameters:
    temperature: 0.1
    max_tokens: 500
    top_p: 0.92

agentcore_memory:
  enabled: true
  memory_name: "SelfHealingAgentMemory"
  region_name: "us-west-2"
  context_retrieval:
    top_k_relevant: 3
    keep_recent_messages: 3
```

## Architecture

The system consists of multiple specialized agents:

1. **Insights Agent**: Analyzes observability traces and generates insights
   - Platform-specific (LangSmith/Langfuse)
   - Uses MCP tools for data access
   - Provides conversational analysis

2. **Evolution Agent**: Optimizes system prompts
   - Analyzes agent performance patterns
   - Generates prompt improvements
   - HITL approval for changes
   - File system access for prompt updates

3. **Routing Agent**: Intelligent workflow routing
   - Determines when evolution is needed
   - User intent detection
   - Conditional workflow execution

## Advanced Features

### Conversation Memory

The agent maintains conversation history using AgentCore Memory for:
- Semantic search of previous interactions
- Context-aware responses
- Long-term learning

### Human-in-the-Loop (HITL)

Before making any prompt modifications, the agent:
1. Shows a detailed diff/patch
2. Displays change statistics
3. Optionally opens in VS Code for review
4. Waits for explicit approval

### Multi-Platform Support

Works with multiple observability platforms:
- **LangSmith**: Full integration with LangSmith MCP tools
- **Langfuse**: Native Langfuse API integration

## Troubleshooting

### Command not found

If `evolve` command is not found after installation:

```bash
# Option 1: Restart your terminal

# Option 2: Source your shell config
source ~/.bashrc  # or ~/.zshrc

# Option 3: Add to PATH manually
export PATH="$HOME/.local/bin:$PATH"
```

### Python version issues

Ensure Python 3.12+ is installed:

```bash
python3 --version
```

### AWS Credentials

Ensure AWS credentials are configured:

```bash
aws configure
# or
export AWS_PROFILE=your_profile
```

### Observability Platform Issues

**LangSmith:**
- Verify `LANGSMITH_API_KEY` is set
- Check project/session ID exists

**Langfuse:**
- Verify `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set
- Check `LANGFUSE_HOST` is correct

## Development

### Running Tests

```bash
# Install development dependencies
uv sync

# Run tests (when available)
uv run pytest
```

### Pre-commit Checks

```bash
# Format and lint
uv run ruff check --fix . && uv run ruff format .

# Security scanning
uv run bandit -r src/

# Type checking
uv run mypy src/
```

## Documentation

- [GitHub Repository](https://github.com/yourusername/self-healing-agent)
- [Configuration Guide](docs/configuration.md) (coming soon)
- [API Documentation](docs/api.md) (coming soon)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run pre-commit checks
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/self-healing-agent/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/self-healing-agent/discussions)

## Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Amazon Bedrock](https://aws.amazon.com/bedrock/)
- [LangSmith](https://www.langchain.com/langsmith)
- [Langfuse](https://langfuse.com/)
