# Adaptive CLI Architecture

## Simple Multi-User CLI Tool with Cognito Auth

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     END USERS                               │
│                                                             │
│  1. Install: pip install adaptive                          │
│  2. Sign up: adaptive auth signup                          │
│  3. Verify email (code sent to their inbox)                │
│  4. Login: adaptive auth login                             │
│  5. Use: adaptive                                           │
│                                                             │
│  Credentials stored locally in ~/.adaptive/auth.json       │
│  No AWS knowledge needed                                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ Direct Cognito API calls
                            │ (boto3 + warrant-lite)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              AWS COGNITO (YOUR USER POOL)                   │
│                                                             │
│  • ONE user pool for ALL users                             │
│  • Email/password authentication                           │
│  • Email verification required                             │
│  • Self-service sign-up enabled                            │
│  • JWT tokens issued on login                              │
│  • Refresh tokens for sessions                             │
│                                                             │
│  Free tier: 50,000 MAU                                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  YOU (ADMIN)                                │
│                                                             │
│  Manage via AWS Console:                                   │
│  • View all users                                          │
│  • Disable/enable accounts                                 │
│  • Delete users                                            │
│  • View metrics                                            │
│  • Monitor sign-ups                                        │
│                                                             │
│  Or via AWS CLI for automation                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. CLI Tool (`adaptive`)

**Location**: `src/adaptive/cli.py`

**Commands**:
- `adaptive auth signup` - Create account
- `adaptive auth login` - Login
- `adaptive auth verify` - Verify email
- `adaptive auth logout` - Logout
- `adaptive auth status` - Show status
- `adaptive run` - Run the agent

### 2. Cognito Auth Manager

**Location**: `src/adaptive/cognito_auth.py`

**Features**:
- User registration (signup)
- Email verification
- Login with email/password
- Token management (access, refresh, ID tokens)
- Automatic token refresh
- Local credential storage

**Dependencies**:
- `boto3` - AWS SDK
- `warrant-lite` - Cognito authentication helper

### 3. Local Storage

**Location**: `~/.adaptive/`

**Files**:
```
~/.adaptive/
├── auth.json          # User authentication tokens
├── config.json        # Configuration settings
└── keyring/          # API keys (encrypted)
```

**Security**:
- `auth.json` permissions: `0600` (user only)
- Directory permissions: `0700` (user only)
- Never committed to version control

## How It Works

### User Sign-Up Flow

```
1. User runs: adaptive auth signup
   ↓
2. CLI prompts for: email, name, password
   ↓
3. CLI calls Cognito API (boto3)
   ↓
4. Cognito creates user account
   ↓
5. Cognito sends verification email
   ↓
6. User receives email with code
   ↓
7. User runs: adaptive auth verify
   ↓
8. CLI calls Cognito API with code
   ↓
9. Account verified ✅
```

### User Login Flow

```
1. User runs: adaptive auth login
   ↓
2. CLI prompts for: email, password
   ↓
3. CLI calls Cognito API
   ↓
4. Cognito validates credentials
   ↓
5. Cognito returns JWT tokens:
   - Access token (for API calls)
   - ID token (user info)
   - Refresh token (get new access token)
   ↓
6. CLI stores tokens in ~/.adaptive/auth.json
   ↓
7. User authenticated ✅
```

### Running the Agent

```
1. User runs: adaptive
   ↓
2. CLI checks ~/.adaptive/auth.json
   ↓
3. If authenticated → continue
   ↓
4. If not → prompt for login
   ↓
5. Run agent with user context
```

## Admin Setup (One-Time)

### Step 1: Create Cognito User Pool

AWS Console → Cognito → Create User Pool:
- Sign-in: Email
- Self-service sign-up: ENABLED
- Email verification: Required
- App client: No secret
- Auth flows: ALLOW_USER_PASSWORD_AUTH, ALLOW_REFRESH_TOKEN_AUTH

### Step 2: Configure Environment

Set these environment variables:

```bash
export ADAPTIVE_COGNITO_USER_POOL_ID="us-east-1_XXXXXXXXX"
export ADAPTIVE_COGNITO_CLIENT_ID="XXXXXXXXXXXXXXXXXXXXXXXXXX"
export ADAPTIVE_COGNITO_REGION="us-east-1"
```

### Step 3: Distribute

Package the CLI tool with these environment variables either:
- **Option A**: Baked into the package during build
- **Option B**: Users set them in their shell config
- **Option C**: Read from a config file

## Security Model

### What's Protected

1. **User passwords** - Never stored, only in Cognito
2. **JWT tokens** - Stored locally, short-lived
3. **API keys** - Encrypted in local keyring
4. **Refresh tokens** - Long-lived, securely stored

### Token Lifecycle

```
Access Token:  1 hour (configurable)
ID Token:      1 hour (configurable)
Refresh Token: 30 days (configurable)
```

### Password Requirements

- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number
- At least one special character

## Scaling

### Current Capacity

- **Free tier**: 50,000 monthly active users (MAU)
- **Performance**: Direct API calls, no latency
- **Storage**: Local only, no database needed

### Beyond Free Tier

- **50,001 - 100,000 MAU**: ~$275/month
- **100,001 - 1,000,000 MAU**: ~$5,225/month

### Cost Optimization

- Users only count as MAU if they authenticate that month
- Inactive users don't cost anything
- Long-lived refresh tokens reduce re-authentication

## Admin Management

### Via AWS Console

**View Users**:
AWS Console → Cognito → Your Pool → Users

**Disable User**:
Select user → Disable

**Delete User**:
Select user → Delete

**View Metrics**:
Your Pool → Metrics tab

### Via AWS CLI

```bash
# List all users
aws cognito-idp list-users \
  --user-pool-id us-east-1_XXXXXXXXX

# Disable user
aws cognito-idp admin-disable-user \
  --user-pool-id us-east-1_XXXXXXXXX \
  --username user@example.com

# Delete user
aws cognito-idp admin-delete-user \
  --user-pool-id us-east-1_XXXXXXXXX \
  --username user@example.com

# Get user info
aws cognito-idp admin-get-user \
  --user-pool-id us-east-1_XXXXXXXXX \
  --username user@example.com
```

## What's NOT Included

❌ Backend API server - not needed for CLI tool
❌ Database - Cognito stores all user data
❌ Admin CLI commands - use AWS Console or CLI
❌ Web interface - pure CLI tool
❌ User roles/permissions - all users equal
❌ Multi-tenancy - single pool for all users

## Deployment Options

### Option 1: PyPI Package

```bash
# Build
python -m build

# Publish
twine upload dist/*

# Users install
pip install adaptive
```

### Option 2: Direct Install Script

```bash
#!/bin/bash
# install.sh

# Set environment variables
export ADAPTIVE_COGNITO_USER_POOL_ID="your-pool-id"
export ADAPTIVE_COGNITO_CLIENT_ID="your-client-id"
export ADAPTIVE_COGNITO_REGION="us-east-1"

# Install package
pip install adaptive

# Add env vars to shell config
echo 'export ADAPTIVE_COGNITO_USER_POOL_ID="your-pool-id"' >> ~/.bashrc
echo 'export ADAPTIVE_COGNITO_CLIENT_ID="your-client-id"' >> ~/.bashrc
echo 'export ADAPTIVE_COGNITO_REGION="us-east-1"' >> ~/.bashrc

# Done
echo "Adaptive installed! Run: source ~/.bashrc && adaptive"
```

### Option 3: Private Distribution

- Share wheel file directly
- Include `.env` file with variables
- Users install locally

## Monitoring

### CloudWatch Logs

Cognito automatically logs:
- Sign-up events
- Authentication attempts
- Failed logins
- Token refreshes

**View**: AWS Console → CloudWatch → Logs → `/aws/cognito/userpools/`

### Metrics

- Monthly active users (MAU)
- Sign-ups per day
- Sign-ins per day
- Failed authentication attempts

**View**: AWS Console → Cognito → Your Pool → Metrics

## Troubleshooting

### Common Issues

**Issue**: "Cognito not configured"
**Fix**: Set environment variables

**Issue**: "Email not verified"
**Fix**: User must run `adaptive auth verify`

**Issue**: "Token expired"
**Fix**: CLI automatically refreshes or prompts re-login

**Issue**: "User not found"
**Fix**: User must sign up first

## Future Enhancements (Optional)

1. **Google OAuth**: Add Google sign-in
2. **Admin CLI**: CLI commands for user management
3. **User roles**: Admin vs regular users
4. **MFA**: Two-factor authentication
5. **Password reset**: Self-service password recovery
6. **SSO**: Enterprise single sign-on

## Summary

This is a **simple, scalable, production-ready** architecture for a CLI tool with user authentication:

✅ **Simple**: Direct Cognito integration, no backend needed
✅ **Secure**: Industry-standard JWT auth, encrypted storage
✅ **Scalable**: 50K free users, auto-scales beyond that
✅ **Maintainable**: Cognito handles everything, you just manage users
✅ **Cost-effective**: Free for most use cases

Perfect for a CLI tool where users need accounts!
