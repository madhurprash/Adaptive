# Adaptive

Continuous optimization for AI agents through intelligent observability and automated code evolution.

## Installation

```bash
curl -fsSL https://raw.githubusercontent.com/madhurprash/adaptive/main/install.sh | bash
```

After installation, restart your terminal or run:
```bash
source ~/.bashrc  # or ~/.zshrc
```

## Quick Start

```bash
adaptive run
```

That's it! On first run, you'll be prompted to:
1. Authenticate with Google (opens browser)
2. Configure your settings
3. Start optimizing your agents

## What Happens on First Run

```
$ adaptive run

╔══════════════════════════════════════════════════════════════════╗
║                    WELCOME TO ADAPTIVE                           ║
╚══════════════════════════════════════════════════════════════════╝

Adaptive helps optimize your AI agents through intelligent
observability and automated code evolution.

╔══════════════════════════════════════════════════════════════════╗
║               AUTHENTICATION REQUIRED                            ║
╚══════════════════════════════════════════════════════════════════╝

To get started, you need to authenticate with Google.
This will:
  • Create your Adaptive account (if new)
  • Or log you into your existing account
  • Takes less than 30 seconds

This is a one-time setup.

Authenticate with Google now? (Y/n): y

🔄 Starting authentication flow...

🔐 Opening browser for Google authentication...

[Browser opens - sign in with Google or create account]

╔══════════════════════════════════════════════════════════════════╗
║            ✅ AUTHENTICATION SUCCESSFUL                          ║
╚══════════════════════════════════════════════════════════════════╝

Logged in as: John Doe (john@example.com)

You can now use Adaptive!

✅ You're all set! Continuing to Adaptive...
```

## Configuration

After authentication, configure Adaptive:

```bash
# Set your model
adaptive config set model us.anthropic.claude-sonnet-4-20250514-v1:0

# Set observability platform  
adaptive config set platform langsmith

# Store API keys
adaptive config set-key langsmith
adaptive config set-key aws
```

## Usage

```bash
# Run Adaptive
adaptive run

# Run with debug logging
adaptive run --debug

# View configuration
adaptive config show

# Check auth status
adaptive auth status
```

## Features

- **Multi-Platform**: LangSmith, Langfuse, MLflow
- **AI-Powered**: Identifies patterns and optimization opportunities
- **Automated**: Evolves prompts and code based on real data
- **Interactive**: Chat about your agent's behavior
- **Safe**: Human-in-the-loop approval for all changes
- **Flexible**: Use any Amazon Bedrock model

## Prerequisites

- **Python 3.12+**
- **AWS Account** (for Bedrock models)
- **Observability Platform** (LangSmith, Langfuse, or MLflow)

### Environment Setup

Set up AWS credentials:
```bash
aws configure
```

Set up your observability platform:

**LangSmith:**
```bash
export LANGSMITH_API_KEY=lsv2_pt_...
export LANGSMITH_PROJECT=my-project
```

**Langfuse:**
```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
```

## Commands

### Authentication
```bash
adaptive auth login      # Log in with Google
adaptive auth logout     # Log out
adaptive auth status     # Show auth status
```

### Configuration
```bash
adaptive config show                    # View config
adaptive config set KEY VALUE           # Set config
adaptive config set-key PLATFORM        # Store API key
adaptive config list-keys               # List API keys
adaptive config delete-key PLATFORM     # Delete API key
```

### Running
```bash
adaptive run                 # Run agent
adaptive run --debug         # Debug mode
adaptive version             # Show version
```

## Google OAuth Setup

To enable Google authentication, you need to set up OAuth credentials:

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project
3. Enable Google+ API
4. Create OAuth 2.0 credentials (Desktop app)
5. Update `src/adaptive/auth.py`:
   ```python
   GOOGLE_CLIENT_ID = "your-client-id.apps.googleusercontent.com"
   GOOGLE_CLIENT_SECRET = "your-client-secret"
   ```

## Configuration Files

- **Auth**: `~/.adaptive/auth.json` (created on first login)
- **Config**: `~/.adaptive/config.json` (API keys, settings)

Both files are automatically secured with user-only permissions.

## Troubleshooting

### Re-authenticate
```bash
adaptive auth logout
adaptive auth login
```

### Reset configuration
```bash
rm ~/.adaptive/config.json
adaptive config set model us.anthropic.claude-sonnet-4-20250514-v1:0
```

### Check installation
```bash
which adaptive
adaptive version
```

## Contributing

Contributions welcome! Please submit pull requests to:
https://github.com/madhurprash/adaptive

## License

MIT License

## Support

- Issues: https://github.com/madhurprash/adaptive/issues
- Discussions: https://github.com/madhurprash/adaptive/discussions
