# Adaptive Onboarding Guide

This guide explains the simplified onboarding flow for Adaptive.

## Quick Start

Just run `adaptive` and follow the prompts:

```bash
adaptive
```

That's it! The CLI will guide you through:
1. Google Authentication
2. Platform Selection (LangSmith, LangFuse, or MLflow)
3. API Key Configuration

## Detailed Flow

### Step 1: Authentication

When you first run `adaptive`, you'll be prompted to authenticate with Google OAuth.

```
WELCOME TO ADAPTIVE
======================================================================

Adaptive helps optimize your AI agents through intelligent
observability and automated code evolution.

Let's get you set up! This will take about 2 minutes.

──────────────────────────────────────────────────────────────────────
STEP 1: AUTHENTICATION
──────────────────────────────────────────────────────────────────────

Adaptive uses Google OAuth for secure authentication.

This will:
  • Create your Adaptive account (if new)
  • Or log you into your existing account
  • Store credentials securely on your machine

Authenticate with Google now? (Y/n):
```

This will:
- Open your browser for Google login
- Create your Adaptive account (or log into existing)
- Store credentials securely in `~/.adaptive/auth.json`

### Step 2: Platform Selection

Next, choose your observability platform:

```
──────────────────────────────────────────────────────────────────────
STEP 2: PLATFORM SELECTION
──────────────────────────────────────────────────────────────────────

Choose your observability platform:

  1. Langsmith
  2. Langfuse
  3. Mlflow

Enter choice (1-3):
```

This sets which platform Adaptive will use for tracking and insights.

### Step 3: API Key Configuration

Finally, provide your API key:

```
──────────────────────────────────────────────────────────────────────
STEP 3: API KEY CONFIGURATION
──────────────────────────────────────────────────────────────────────

Enter your Langsmith API key.
(Your input will be hidden for security)

Langsmith API Key:
```

**Environment Variables**: If you have the API key in an environment variable, Adaptive will detect it:

- LangSmith: `LANGSMITH_API_KEY`
- LangFuse: `LANGFUSE_API_KEY`
- MLflow: `MLFLOW_TRACKING_URI`

### Completion

Once setup is complete, you'll see:

```
======================================================================
                          SETUP COMPLETE
======================================================================

✅ You're all set! You can now use Adaptive.

Next steps:
  • Run 'adaptive run' to start the agent
  • Run 'adaptive config show' to view your configuration
  • Run 'adaptive --help' for more options
```

## Google OAuth Setup (For Developers)

To enable Google OAuth, you need to set up OAuth credentials:

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select existing
3. Enable Google OAuth 2.0
4. Create OAuth 2.0 credentials (Desktop app)
5. Set environment variables:

```bash
export ADAPTIVE_GOOGLE_CLIENT_ID="your-client-id.apps.googleusercontent.com"
export ADAPTIVE_GOOGLE_CLIENT_SECRET="your-client-secret"
```

## Manual Configuration (Optional)

You can also configure Adaptive manually:

```bash
# Set platform
adaptive config set platform langsmith

# Set API key
adaptive config set-key langsmith

# Set model (optional)
adaptive config set model us.anthropic.claude-sonnet-4-20250514-v1:0

# View configuration
adaptive config show

# View authentication status
adaptive auth status
```

## Returning Users

For returning users, just run:

```bash
adaptive
```

Adaptive will remember your authentication and configuration, and start immediately.

## Managing Authentication

```bash
# View status
adaptive auth status

# Log out
adaptive auth logout

# Log in again
adaptive auth login
```

## Managing API Keys

```bash
# Add or update API key
adaptive config set-key langsmith

# List stored API keys
adaptive config list-keys

# Delete API key
adaptive config delete-key langsmith
```

## Configuration Files

Adaptive stores configuration in your home directory:

- `~/.adaptive/auth.json` - Authentication credentials (secure, user-only)
- `~/.adaptive/config.json` - Configuration and API keys (secure, user-only)

Both files have `chmod 600` permissions for security.

## Troubleshooting

### Import Error: google_auth_oauthlib

If you see `ModuleNotFoundError: No module named 'google_auth_oauthlib'`, reinstall:

```bash
uv pip install google-auth-oauthlib
```

### Authentication Browser Not Opening

If the browser doesn't open automatically, copy the URL from the terminal and paste it into your browser.

### API Key Not Working

1. Check if the API key is correct:
   ```bash
   adaptive config show
   ```

2. Try setting it again:
   ```bash
   adaptive config set-key <platform>
   ```

3. Or use environment variables:
   ```bash
   export LANGSMITH_API_KEY="your-key"
   adaptive
   ```

## Environment Variables

### Required for Google OAuth

- `ADAPTIVE_GOOGLE_CLIENT_ID` - Google OAuth Client ID
- `ADAPTIVE_GOOGLE_CLIENT_SECRET` - Google OAuth Client Secret

### Optional (for API keys)

- `LANGSMITH_API_KEY` - LangSmith API key
- `LANGFUSE_API_KEY` - LangFuse API key
- `MLFLOW_TRACKING_URI` - MLflow tracking URI

## Security

- All credentials are stored locally with `chmod 600` (user-only access)
- API keys are never logged or printed
- Google OAuth tokens are stored securely
- No data is sent to external servers except Google (for auth) and your chosen platform
