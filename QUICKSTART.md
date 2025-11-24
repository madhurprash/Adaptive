# Quick Start Guide

Get started with Adaptive in 5 minutes!

## Installation

Choose your preferred installation method:

### One-Line Install (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/madhurprash/adaptive/main/scripts/install.sh | bash
```

## Usage

After installation, you can run the agent using the `adaptive` command:

### Basic Usage

```bash
# Run agent in interactive mode (auto-generates session ID)
adaptive
adaptive run

# Run agent with a specific session ID
adaptive run --session-id your-session-id

# Run with debug logging
adaptive run --debug

# Run with session ID and debug
adaptive run --session-id madhur2039 --debug
```

### Alternative: Direct Python Execution

You can also run the agent directly using Python:

```bash
# Navigate to the installation directory
cd ~/.adaptive

# Run with Python
python adaptive.py --session-id your-session-id
```

### Other Commands

```bash
# Show version
adaptive version

# Show current configuration
adaptive config

# Run as background daemon (checks every hour)
adaptive daemon

# Run daemon with custom interval (every 30 minutes)
adaptive daemon --interval 1800
```
