# Quick Start Guide

Get started with the Self-Healing Agent in 5 minutes!

## Installation

Choose your preferred installation method:

### One-Line Install (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/madhurprash/evolve.ai/main/scripts/install.sh | bash
```

## Usage

After installation, you can run the agent using the `evolve` command:

### Basic Usage

```bash
# Run agent in interactive mode (auto-generates session ID)
evolve
evolve run

# Run agent with a specific session ID
evolve run --session-id your-session-id

# Run with debug logging
evolve run --debug

# Run with session ID and debug
evolve run --session-id madhur2039 --debug
```

### Alternative: Direct Python Execution

You can also run the agent directly using Python:

```bash
# Navigate to the installation directory
cd ~/.self-healing-agent

# Run with Python
python evolve.py --session-id your-session-id
```

### Other Commands

```bash
# Show version
evolve version

# Show current configuration
evolve config

# Run as background daemon (checks every hour)
evolve daemon

# Run daemon with custom interval (every 30 minutes)
evolve daemon --interval 1800
```