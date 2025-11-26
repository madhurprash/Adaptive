# Quick Start Guide

## For You (Admin) - First Time Setup

### 1. Create Cognito User Pool

Run the automated setup script:

```bash
./scripts/setup_cognito.sh
```

This will:
- ✅ Create a Cognito User Pool
- ✅ Create an App Client
- ✅ Save configuration to `~/.adaptive/cognito_config.sh`

### 2. Add to Your Shell Config

Add this line to your `~/.bashrc` or `~/.zshrc`:

```bash
source ~/.adaptive/cognito_config.sh
```

Reload your shell:

```bash
source ~/.bashrc  # or ~/.zshrc
```

### 3. Test It

```bash
# Create a test account
adaptive auth signup

# Check AWS Console to see your user:
# https://console.aws.amazon.com/cognito/
```

**That's it! You're done.** 🎉

---

## For End Users

### 1. Install

```bash
pip install adaptive
```

### 2. Create Account

```bash
adaptive auth signup
```

Follow the prompts for email, name, and password.

### 3. Verify Email

Check your email for a verification code, then:

```bash
adaptive auth verify
```

### 4. Login & Use

```bash
adaptive auth login
adaptive
```

**Done!** 🚀

---

## Admin Management

### View All Users

AWS Console → Cognito → Your User Pool → Users

### Manage a User

Select user → Actions:
- Disable user
- Delete user
- Resend verification code

### View Metrics

Your User Pool → Metrics tab

---

## Common Commands

### Authentication
```bash
adaptive auth signup    # Create account
adaptive auth login     # Login
adaptive auth verify    # Verify email
adaptive auth status    # Check status
adaptive auth logout    # Logout
```

### Configuration
```bash
adaptive config show              # View config
adaptive config set platform X    # Set platform
adaptive config set-key langsmith # Store API key
```

### Running
```bash
adaptive           # Run agent
adaptive run       # Run explicitly
adaptive --help    # Show help
```

---

## Support

- **Setup issues**: See `ADMIN_SETUP.md`
- **User guide**: See `USER_GUIDE.md`
- **Architecture**: See `ARCHITECTURE.md`
- **Full docs**: See `IMPLEMENTATION_SUMMARY.md`

---

## That's It!

Simple multi-user CLI tool with Cognito authentication.

No backend server needed. No database. Just Cognito + CLI.

**Free for up to 50,000 users per month.**
