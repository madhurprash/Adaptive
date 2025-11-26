# Admin Setup Guide for Adaptive

## Overview

This guide is for **YOU** (the admin) to set up Adaptive so that end users can create accounts and use the CLI tool.

## Simple Architecture

```
┌─────────────────────────────────────────────┐
│         END USERS (CLI)                     │
│  • Run: adaptive auth signup                │
│  • Creates account in YOUR Cognito pool     │
│  • Credentials stored locally               │
│  • No AWS knowledge needed                  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│      AWS COGNITO (YOUR POOL)                │
│  • ONE user pool for all users              │
│  • You set pool ID via environment vars     │
│  • Users automatically go to your pool      │
│  • You manage via AWS Console               │
└─────────────────────────────────────────────┘
```

## Step 1: Create Cognito User Pool (ONE TIME)

### Using AWS Console

1. Go to [AWS Cognito Console](https://console.aws.amazon.com/cognito/)
2. Click "Create user pool"
3. **Sign-in options**: Select "Email"
4. **Password policy**: Use defaults or customize
5. **MFA**: Optional (recommended: OFF for easier onboarding)
6. **Self-service sign-up**: **ENABLE** (important!)
7. **Email verification**: Select "Email"
8. **Email provider**: Use "Send email with Cognito" (free tier)
9. **User pool name**: `adaptive-users`
10. **App client name**: `adaptive-cli`
11. **Client secret**: Choose "Don't generate a client secret"
12. **Auth flows**: Enable:
    - ✅ ALLOW_USER_PASSWORD_AUTH
    - ✅ ALLOW_REFRESH_TOKEN_AUTH

13. **Create user pool**

### Save These Values

After creation, copy:
- **User Pool ID**: `us-east-1_XXXXXXXXX`
- **App Client ID**: `XXXXXXXXXXXXXXXXXXXXXXXXXX`
- **Region**: `us-east-1` (or your chosen region)

## Step 2: Set Environment Variables (YOUR MACHINE)

Add these to your `~/.bashrc` or `~/.zshrc`:

```bash
# Adaptive Cognito Configuration
export ADAPTIVE_COGNITO_USER_POOL_ID="us-east-1_XXXXXXXXX"
export ADAPTIVE_COGNITO_CLIENT_ID="XXXXXXXXXXXXXXXXXXXXXXXXXX"
export ADAPTIVE_COGNITO_REGION="us-east-1"
```

Apply changes:
```bash
source ~/.bashrc  # or ~/.zshrc
```

## Step 3: Package and Distribute

### Option A: Distribute as Python Package

Users install via pip/uv:
```bash
pip install adaptive
# or
uv pip install adaptive
```

### Option B: Share Environment Variables

Create a `.env.template` file for users:
```bash
# Add these to your shell configuration (~/.bashrc or ~/.zshrc)
export ADAPTIVE_COGNITO_USER_POOL_ID="us-east-1_XXXXXXXXX"
export ADAPTIVE_COGNITO_CLIENT_ID="XXXXXXXXXXXXXXXXXXXXXXXXXX"
export ADAPTIVE_COGNITO_REGION="us-east-1"
```

Users add this to their shell config and reload.

## Step 4: User Flow (What End Users Do)

### First Time Setup

```bash
# 1. Install adaptive
pip install adaptive

# 2. Create account
adaptive auth signup
# Prompts for: email, name (optional), password

# 3. Verify email
adaptive auth verify
# Enters verification code from email

# 4. Start using adaptive
adaptive
```

### Subsequent Uses

```bash
# Just login
adaptive auth login

# Or run directly (will prompt for login if needed)
adaptive
```

## Admin Tasks

### View All Users

Go to AWS Console → Cognito → Your User Pool → Users tab

You can see:
- All registered users
- Email verification status
- Account status (enabled/disabled)
- Sign-up date

### Manage Users

In AWS Console, you can:

**Disable a user:**
1. Select user
2. Click "Disable user"

**Delete a user:**
1. Select user
2. Click "Delete user"

**Resend verification code:**
1. Select user
2. Click "Resend code"

### View Sign-ups and Activity

AWS Console → Cognito → Your Pool → "Metrics" tab

Shows:
- Sign-ups per day
- Sign-ins per day
- Failed authentication attempts

## Security Best Practices

### 1. Password Policy

Default requirements (recommended):
- Minimum 8 characters
- Uppercase letter
- Lowercase letter
- Number
- Special character

### 2. User Data

- User credentials stored locally in `~/.adaptive/auth.json`
- File permissions: `0600` (user read/write only)
- Users should never share this file

### 3. Environment Variables

- Store in shell configuration
- Never commit to version control
- For production: Use AWS Secrets Manager or Parameter Store

## Troubleshooting

### User Can't Sign Up

**Error**: "Cognito not configured"

**Solution**: User needs to set environment variables (or you need to package them)

### User Not Receiving Verification Email

**Check**:
1. AWS Console → Cognito → Your Pool → Messaging
2. Ensure email delivery is configured
3. Check spam folder
4. Resend verification code via Console

### Too Many Users (>50,000 MAU)

Cognito free tier: 50,000 monthly active users

Beyond that: $0.0055 per MAU

**Solution**: You'll start getting charged automatically (AWS bill)

## Cost Summary

| Users | Monthly Cost |
|-------|--------------|
| 0 - 50,000 | **FREE** |
| 50,001 - 100,000 | ~$275 |
| 100,001 - 1,000,000 | ~$5,225 |

For most use cases, you'll stay in the free tier.

## Admin CLI Commands (Future)

Currently, admin tasks are done via AWS Console. You can manage users there.

If you want CLI admin commands, you would use AWS CLI:

```bash
# List users
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
```

## Architecture Summary

**What You Control:**
- ✅ Cognito User Pool (one pool for all users)
- ✅ User management via AWS Console
- ✅ Environment variables (pool ID, client ID)

**What Users Do:**
- ✅ Run `adaptive auth signup` (creates account in YOUR pool)
- ✅ Run `adaptive auth login` (authenticates against YOUR pool)
- ✅ Use the CLI tool

**What's NOT Needed:**
- ❌ No backend API to deploy
- ❌ No database to manage
- ❌ No server to maintain
- ❌ Just Cognito + CLI tool

## Next Steps

1. ✅ Create Cognito User Pool (Step 1)
2. ✅ Set your environment variables (Step 2)
3. ✅ Test account creation: `adaptive auth signup`
4. ✅ Check AWS Console to see your test user
5. ✅ Delete test user if needed
6. ✅ Ready for production!

## Support

For issues:
- Check AWS Console → Cognito → Logs
- CloudWatch logs for authentication events
- User reports email verification issues → Check spam/resend code
