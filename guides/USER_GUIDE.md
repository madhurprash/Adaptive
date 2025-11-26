# Adaptive User Guide

Welcome to Adaptive! This guide will help you get started.

## What is Adaptive?

Adaptive is a CLI tool that helps optimize your AI agents through intelligent observability and automated code evolution.

## Installation

```bash
# Using pip
pip install adaptive

# Or using uv (faster)
uv pip install adaptive
```

## First Time Setup

### Step 1: Create Account

```bash
adaptive auth signup
```

You'll be prompted for:
- **Email**: Your email address
- **Name**: Your name (optional)
- **Password**: Must contain:
  - At least 8 characters
  - One uppercase letter
  - One lowercase letter
  - One number
  - One special character

Example:
```
$ adaptive auth signup

==================================================================
CREATE ADAPTIVE ACCOUNT
==================================================================

Creating account for: you@example.com

Password requirements:
  • At least 8 characters
  • One uppercase letter
  • One lowercase letter
  • One number
  • One special character

Password: ********
Confirm password: ********

✅ Account created successfully!

⚠️  Please check your email for a verification code.
```

### Step 2: Verify Email

Check your email for a verification code, then:

```bash
adaptive auth verify
```

Enter your email and the verification code.

```
$ adaptive auth verify

==================================================================
VERIFY EMAIL
==================================================================

Email: you@example.com
Verification code: 123456

✅ Email verified successfully!
You can now log in with: adaptive auth login
```

### Step 3: Login

```bash
adaptive auth login
```

Enter your email and password.

```
$ adaptive auth login

==================================================================
ADAPTIVE - LOGIN
==================================================================

Logging in as: you@example.com
Password: ********

==================================================================
✅ LOGIN SUCCESSFUL
==================================================================

Logged in as: you@example.com
Name: Your Name

You can now use Adaptive!
```

## Using Adaptive

### Run the Agent

Once authenticated, just run:

```bash
adaptive
```

Or explicitly:

```bash
adaptive run
```

### Configure Settings

#### Set Observability Platform

```bash
adaptive config set platform langsmith
# or langfuse, mlflow
```

#### Set API Key

```bash
adaptive config set-key langsmith
# Will prompt for API key securely
```

#### View Current Config

```bash
adaptive config show
```

#### Set Model

```bash
adaptive config set model us.anthropic.claude-sonnet-4-20250514-v1:0
```

## Authentication Commands

### Check Login Status

```bash
adaptive auth status
```

### Logout

```bash
adaptive auth logout
```

### Login Again

```bash
adaptive auth login
```

## Configuration Management

### Set Configuration

```bash
# Set model
adaptive config set model <model-id>

# Set platform
adaptive config set platform <platform>  # langsmith, langfuse, mlflow

# Set temperature
adaptive config set temperature 0.7

# Set max tokens
adaptive config set max_tokens 1000
```

### Manage API Keys

```bash
# Store API key
adaptive config set-key <platform>

# List stored keys
adaptive config list-keys

# Delete API key
adaptive config delete-key <platform>
```

## Troubleshooting

### "Not authenticated" Error

**Solution**: Run `adaptive auth login`

### "Email not verified" Error

**Solution**:
1. Check your email for verification code
2. Run `adaptive auth verify`
3. Enter email and code

### Forgot Password

Currently, password reset must be done through AWS Cognito. Contact your admin.

### Can't Receive Verification Email

**Check**:
1. Spam folder
2. Email address is correct
3. Contact admin to resend code

### Configuration Not Saving

**Check**:
1. File permissions on `~/.adaptive/`
2. Disk space
3. Try running with elevated permissions (if on restricted system)

## File Locations

Adaptive stores data in:

```
~/.adaptive/
├── auth.json          # Authentication tokens (DO NOT SHARE)
├── config.json        # Configuration settings
└── keyring/          # API keys (encrypted)
```

### Security Note

**NEVER** share or commit `~/.adaptive/auth.json` - it contains your authentication tokens!

## Common Workflows

### First Time User

```bash
# 1. Install
pip install adaptive

# 2. Create account
adaptive auth signup

# 3. Verify email
adaptive auth verify

# 4. Configure platform
adaptive config set platform langsmith
adaptive config set-key langsmith

# 5. Run!
adaptive
```

### Returning User

```bash
# Just run it
adaptive

# Or if session expired
adaptive auth login
adaptive
```

### Switching Platforms

```bash
# Change platform
adaptive config set platform langfuse

# Set new API key
adaptive config set-key langfuse

# Run with new platform
adaptive
```

## Command Reference

### Auth Commands

| Command | Description |
|---------|-------------|
| `adaptive auth signup` | Create new account |
| `adaptive auth login` | Login with email/password |
| `adaptive auth verify` | Verify email with code |
| `adaptive auth logout` | Logout |
| `adaptive auth status` | Show auth status |

### Config Commands

| Command | Description |
|---------|-------------|
| `adaptive config show` | Show current config |
| `adaptive config set <key> <value>` | Set config value |
| `adaptive config set-key <platform>` | Store API key |
| `adaptive config list-keys` | List stored API keys |
| `adaptive config delete-key <platform>` | Delete API key |

### Run Commands

| Command | Description |
|---------|-------------|
| `adaptive` | Run agent (default) |
| `adaptive run` | Run agent explicitly |
| `adaptive run --debug` | Run with debug logging |
| `adaptive version` | Show version |

## Getting Help

- Documentation: https://github.com/madhurprash/adaptive
- Issues: https://github.com/madhurprash/adaptive/issues

## Privacy & Security

- Authentication tokens stored locally in `~/.adaptive/auth.json`
- File permissions set to `0600` (user read/write only)
- API keys encrypted in keyring
- Never commit `.adaptive/` directory to version control
- Passwords must meet strength requirements
- Email verification required for all accounts

## What's Next?

After setup, Adaptive will:
1. Monitor your AI agent's performance
2. Collect observability data
3. Suggest optimizations
4. Evolve your prompts automatically
5. Track improvements over time

Enjoy using Adaptive!
