# Quick Start Guide

Get started with the Self-Healing Agent in 5 minutes!

## Installation

Choose your preferred installation method:

### Option 1: One-Line Install (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/yourusername/self-healing-agent/main/scripts/install.sh | bash
```

### Option 2: Manual Install with uv

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/yourusername/self-healing-agent.git
cd self-healing-agent
uv pip install -e .
```

### Option 3: pip install

```bash
git clone https://github.com/yourusername/self-healing-agent.git
cd self-healing-agent
pip install -e .
```

## Setup Environment

Create a `.env` file:

```bash
# Copy the example
cp .env.example .env

# Edit with your credentials
nano .env  # or use your favorite editor
```

Minimum required configuration:

```bash
# AWS
AWS_REGION=us-east-1

# LangSmith (if using LangSmith)
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=your_project

# OR Langfuse (if using Langfuse)
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

## First Run

```bash
# Start the agent
evolve run

# Follow the prompts:
# 1. Select your observability platform (LangSmith or Langfuse)
# 2. Ask questions about your agent traces
# 3. Review and approve any suggested improvements
```

## Example Session

```bash
$ evolve run

Self-Healing Agent - Interactive Mode
================================================================================

🔍 [PLATFORM SELECTION] Please select your observability platform:
   1. LangSmith
   2. Langfuse

Enter your choice (1 or 2): 1
✅ Selected: LangSmith

You: What errors occurred in my last agent run?

[Agent analyzes traces...]

Agent: I found 3 main error patterns:
1. Timeout errors (42% of failures)
2. JSON parsing errors (31% of failures)
3. API rate limits (27% of failures)

The timeout errors primarily occur in the retrieval step...

You: Can you help optimize the prompts to handle timeouts better?

[Evolution engine analyzes and suggests improvements...]
[Shows diff of proposed changes...]

Apply this change? [y/n]: y

✅ Changes applied successfully!
```

## Common Commands

```bash
# Interactive mode
evolve run

# With debug logging
evolve run --debug

# Run as daemon (checks hourly)
evolve daemon

# Custom check interval (30 min)
evolve daemon --interval 1800

# Show version
evolve version

# Show config
evolve config
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Configure advanced settings in `configs/config.yaml`
- Set up AgentCore Memory for conversation history
- Customize prompt templates in `prompt_templates/`

## Troubleshooting

### Command not found

```bash
# Add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Or restart terminal
```

### AWS credentials

```bash
# Configure AWS
aws configure

# Or set profile
export AWS_PROFILE=your_profile
```

### Platform connection issues

```bash
# Test LangSmith connection
export LANGSMITH_API_KEY=your_key
python -c "from langsmith import Client; print(Client().list_projects())"

# Test Langfuse connection
export LANGFUSE_PUBLIC_KEY=your_key
export LANGFUSE_SECRET_KEY=your_secret
python -c "from langfuse import Langfuse; print(Langfuse().auth_check())"
```

## Getting Help

- GitHub Issues: [Report a problem](https://github.com/yourusername/self-healing-agent/issues)
- Documentation: [Full README](README.md)
- Examples: [Check examples/](examples/)

## What's Next?

Once you're comfortable with the basics:

1. Explore the evolution engine for automatic prompt optimization
2. Set up continuous monitoring with daemon mode
3. Integrate with your CI/CD pipeline
4. Customize the routing logic for your use case
5. Contribute to the project!

Happy evolving! 🚀
