# AWS Cognito Authentication Setup

This guide will walk you through setting up AWS Cognito for Adaptive authentication, including email/password authentication and Google OAuth.

## Prerequisites

- AWS Account with appropriate permissions
- AWS CLI configured (optional but recommended)
- Basic understanding of AWS Cognito

## Overview

Adaptive uses AWS Cognito for secure, scalable authentication with support for:
- **Email/Password Authentication** - Traditional account creation
- **Google OAuth** - Sign in with Google (via Cognito Federated Identities)
- **Token Management** - Automatic token refresh and secure storage

## Step 1: Create a Cognito User Pool

### Using AWS Console

1. **Navigate to Amazon Cognito**
   - Go to the [AWS Console](https://console.aws.amazon.com/)
   - Search for "Cognito" and click on "Amazon Cognito"

2. **Create User Pool**
   - Click "Create user pool"
   - Choose "User name" and "Email" as sign-in options
   - Select "Email" as an attribute
   - Click "Next"

3. **Configure Security Requirements**
   - **Password Policy**: Keep defaults or customize:
     - Minimum length: 8 characters
     - Require uppercase letters
     - Require lowercase letters
     - Require numbers
     - Require special characters
   - **MFA**: Optional (recommended for production)
   - Click "Next"

4. **Configure Sign-up Experience**
   - **Self-service sign-up**: Enable
   - **Attribute verification**: Select "Email"
   - **Required attributes**: Add "email" and optionally "name"
   - Click "Next"

5. **Configure Message Delivery**
   - Choose "Send email with Amazon SES" (recommended) or "Send email with Cognito"
   - If using Cognito email, no additional configuration needed
   - For SES, ensure your domain/email is verified
   - Click "Next"

6. **Integrate Your App**
   - **User pool name**: `adaptive-user-pool` (or your preferred name)
   - **App client name**: `adaptive-cli`
   - **Client authentication**: Choose "Don't generate a client secret"
   - **Authentication flows**: Enable these:
     - ✅ ALLOW_USER_PASSWORD_AUTH
     - ✅ ALLOW_REFRESH_TOKEN_AUTH
   - Click "Next"

7. **Review and Create**
   - Review your settings
   - Click "Create user pool"

8. **Save Important Values**
   After creation, note these values:
   - **User Pool ID**: `us-east-1_xxxxxxxxx`
   - **App Client ID**: `xxxxxxxxxxxxxxxxxxxxxxxxxx`
   - **Region**: `us-east-1` (or your chosen region)

### Using AWS CLI

```bash
# Create user pool
aws cognito-idp create-user-pool \
  --pool-name adaptive-user-pool \
  --policies "PasswordPolicy={MinimumLength=8,RequireUppercase=true,RequireLowercase=true,RequireNumbers=true,RequireSymbols=true}" \
  --auto-verified-attributes email \
  --username-attributes email \
  --schema '[{"Name":"email","Required":true,"Mutable":true},{"Name":"name","Required":false,"Mutable":true}]' \
  --email-configuration EmailSendingAccount=COGNITO_DEFAULT

# Note the User Pool ID from the output

# Create app client
aws cognito-idp create-user-pool-client \
  --user-pool-id us-east-1_xxxxxxxxx \
  --client-name adaptive-cli \
  --no-generate-secret \
  --explicit-auth-flows ALLOW_USER_PASSWORD_AUTH ALLOW_REFRESH_TOKEN_AUTH

# Note the Client ID from the output
```

## Step 2: Configure Environment Variables

Add these environment variables to your shell configuration (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
# Required for email/password authentication
export ADAPTIVE_COGNITO_USER_POOL_ID="us-east-1_xxxxxxxxx"
export ADAPTIVE_COGNITO_CLIENT_ID="xxxxxxxxxxxxxxxxxxxxxxxxxx"
export ADAPTIVE_COGNITO_REGION="us-east-1"
```

Apply the changes:
```bash
source ~/.bashrc  # or ~/.zshrc
```

## Step 3: Test Email/Password Authentication

Now you can use Adaptive with email/password authentication:

```bash
# Create a new account
adaptive auth signup

# Or with command-line arguments
adaptive auth signup --email user@example.com --name "Your Name"

# Verify your email (check your inbox for code)
adaptive auth verify

# Login
adaptive auth login

# Check authentication status
adaptive auth status
```

## Step 4: Configure Google OAuth (Optional)

To enable "Sign in with Google", you need to:

### 4.1: Create Google OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable the Google+ API
4. Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client ID"
5. Configure OAuth consent screen if prompted
6. Choose "Web application"
7. Add authorized redirect URIs (we'll add Cognito domain later)
8. Save the **Client ID** and **Client Secret**

### 4.2: Add Google as Identity Provider in Cognito

1. **In AWS Console, go to your User Pool**
2. **Navigate to "Sign-in experience" → "Federated identity provider sign-in"**
3. **Click "Add identity provider"**
4. **Select "Google"**
5. **Enter your Google credentials**:
   - **Google app ID**: Your Google OAuth Client ID
   - **Google app secret**: Your Google OAuth Client Secret
   - **Authorized scopes**: `profile email openid`
6. **Click "Save changes"**

### 4.3: Configure App Integration

1. **In your User Pool, go to "App integration" tab**
2. **Click "Create app client"** (or edit existing)
3. **Under "Hosted UI settings"**:
   - Choose a domain prefix (e.g., `adaptive-prod`)
   - Full domain will be: `adaptive-prod.auth.us-east-1.amazoncognito.com`
4. **Configure OAuth 2.0 settings**:
   - **Callback URLs**: `http://localhost:8080/callback`
   - **Sign-out URLs**: `http://localhost:8080/logout`
   - **OAuth Grant Types**: Select "Authorization code grant"
   - **OAuth Scopes**: Select `email`, `openid`, `profile`
5. **Save changes**

### 4.4: Update Google OAuth Redirect URIs

1. Go back to Google Cloud Console
2. Edit your OAuth 2.0 Client
3. Add this authorized redirect URI:
   ```
   https://adaptive-prod.auth.us-east-1.amazoncognito.com/oauth2/idpresponse
   ```
   (Replace `adaptive-prod` with your domain prefix and region)
4. Save changes

### 4.5: Update Environment Variables

Add the Cognito domain to your environment:

```bash
export ADAPTIVE_COGNITO_DOMAIN="adaptive-prod"
```

### 4.6: Test Google OAuth

```bash
# Login with Google
adaptive auth google

# This will open your browser for Google authentication
```

## Environment Variable Reference

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `ADAPTIVE_COGNITO_USER_POOL_ID` | Yes | Cognito User Pool ID | `us-east-1_ABC123XYZ` |
| `ADAPTIVE_COGNITO_CLIENT_ID` | Yes | App Client ID | `1a2b3c4d5e6f7g8h9i0j` |
| `ADAPTIVE_COGNITO_REGION` | No | AWS Region | `us-east-1` (default) |
| `ADAPTIVE_COGNITO_DOMAIN` | No* | Cognito domain prefix | `adaptive-prod` |

\* Required only for Google OAuth

## Usage Examples

### Creating an Account

```bash
# Interactive mode
adaptive auth signup

# With arguments
adaptive auth signup --email john@example.com --name "John Doe"
```

### Logging In

```bash
# Interactive mode
adaptive auth login

# With arguments
adaptive auth login --email john@example.com --password MySecurePass123!
```

### Email Verification

After signup, you'll receive an email with a verification code:

```bash
# Interactive mode
adaptive auth verify

# With arguments
adaptive auth verify --email john@example.com --code 123456
```

### Google OAuth

```bash
adaptive auth google
```

### Check Status

```bash
adaptive auth status
```

### Logout

```bash
adaptive auth logout
```

## Security Best Practices

### 1. Password Requirements
The default policy requires:
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number
- At least one special character

### 2. Credentials Storage
- Credentials are stored locally in `~/.adaptive/auth.json`
- File permissions are set to `0600` (user read/write only)
- Never commit this file to version control

### 3. Token Management
- Access tokens are short-lived
- Refresh tokens are used to obtain new access tokens
- Tokens are automatically refreshed when needed

### 4. Environment Variables
- Store environment variables in your shell configuration
- Never hardcode credentials in code
- Use AWS Secrets Manager for production deployments

## Troubleshooting

### "OAuth client was not found" Error

This error occurs when:
- Google OAuth credentials are not properly configured
- The Client ID is incorrect
- You haven't set up the environment variables

**Solution**: Follow Step 4 to configure Google OAuth properly

### Email Not Verified

If you see "UserNotConfirmedException":
1. Check your email for verification code
2. Run: `adaptive auth verify`
3. Enter your email and verification code

### Invalid Password

If signup fails due to invalid password:
- Ensure password meets all requirements
- Use at least 8 characters
- Include uppercase, lowercase, number, and special character

### AWS Credentials Not Found

Ensure you have:
- AWS CLI configured (optional for end-users)
- The boto3 library installed
- Valid AWS credentials if calling from a Lambda function

## Production Deployment

For production use:

1. **Use AWS Secrets Manager** for storing Cognito configuration:
   ```bash
   aws secretsmanager create-secret \
     --name adaptive/cognito-config \
     --secret-string '{"user_pool_id":"...","client_id":"...","region":"..."}'
   ```

2. **Enable MFA** for additional security

3. **Use Custom Email Sender** with Amazon SES:
   - Verify your domain
   - Configure DKIM
   - Set up email templates

4. **Set Up Monitoring**:
   - Enable CloudWatch Logs
   - Set up alarms for authentication failures
   - Monitor user sign-up trends

5. **Configure Password Policy** for compliance requirements

## Next Steps

After setting up authentication:

1. Configure your observability platform:
   ```bash
   adaptive config set platform langsmith
   adaptive config set-key langsmith
   ```

2. Start using Adaptive:
   ```bash
   adaptive
   ```

## Support

For issues or questions:
- GitHub Issues: https://github.com/madhurprash/adaptive/issues
- Documentation: https://github.com/madhurprash/adaptive

## Architecture Diagram

```
┌─────────────────┐
│  Adaptive CLI   │
└────────┬────────┘
         │
         ├─── Email/Password Auth
         │         │
         │         v
         │    ┌─────────────────┐
         │    │  AWS Cognito    │
         │    │   User Pool     │
         │    └─────────────────┘
         │
         └─── Google OAuth
                   │
                   v
              ┌─────────────────┐
              │  Cognito        │
              │  Hosted UI      │
              └────────┬────────┘
                       │
                       v
              ┌─────────────────┐
              │  Google OAuth   │
              └─────────────────┘
```

## Cost Considerations

AWS Cognito pricing (as of 2024):
- First 50,000 MAUs: Free
- Beyond 50,000: $0.0055 per MAU

For most individual users and small teams, Cognito is free under the 50,000 monthly active users threshold.
