#!/bin/bash

# Setup script for Adaptive Cognito User Pool
# Run this ONCE to create your Cognito user pool

set -e

echo "======================================================================="
echo "ADAPTIVE COGNITO SETUP"
echo "======================================================================="
echo ""
echo "This script will create a Cognito User Pool for Adaptive users."
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed"
    echo "Install it from: https://aws.amazon.com/cli/"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured"
    echo "Run: aws configure"
    exit 1
fi

echo "✓ AWS CLI configured"
echo ""

# Prompt for region
read -p "AWS Region (default: us-east-1): " AWS_REGION
AWS_REGION=${AWS_REGION:-us-east-1}

# Prompt for user pool name
read -p "User Pool Name (default: adaptive-users): " POOL_NAME
POOL_NAME=${POOL_NAME:-adaptive-users}

echo ""
echo "Creating Cognito User Pool..."
echo "  Name: $POOL_NAME"
echo "  Region: $AWS_REGION"
echo ""

# Create user pool
POOL_OUTPUT=$(aws cognito-idp create-user-pool \
  --pool-name "$POOL_NAME" \
  --region "$AWS_REGION" \
  --policies '{
    "PasswordPolicy": {
      "MinimumLength": 8,
      "RequireUppercase": true,
      "RequireLowercase": true,
      "RequireNumbers": true,
      "RequireSymbols": true
    }
  }' \
  --auto-verified-attributes email \
  --username-attributes email \
  --schema '[
    {
      "Name": "email",
      "AttributeDataType": "String",
      "Required": true,
      "Mutable": true
    },
    {
      "Name": "name",
      "AttributeDataType": "String",
      "Required": false,
      "Mutable": true
    }
  ]' \
  --email-configuration EmailSendingAccount=COGNITO_DEFAULT \
  --user-pool-add-ons '{"AdvancedSecurityMode": "OFF"}' \
  --output json)

# Extract user pool ID
USER_POOL_ID=$(echo "$POOL_OUTPUT" | jq -r '.UserPool.Id')

if [ -z "$USER_POOL_ID" ] || [ "$USER_POOL_ID" == "null" ]; then
    echo "❌ Failed to create user pool"
    exit 1
fi

echo "✓ User Pool created: $USER_POOL_ID"
echo ""

# Create app client
echo "Creating App Client..."

CLIENT_OUTPUT=$(aws cognito-idp create-user-pool-client \
  --user-pool-id "$USER_POOL_ID" \
  --region "$AWS_REGION" \
  --client-name "adaptive-cli" \
  --no-generate-secret \
  --explicit-auth-flows ALLOW_USER_PASSWORD_AUTH ALLOW_REFRESH_TOKEN_AUTH ALLOW_USER_SRP_AUTH \
  --output json)

# Extract client ID
CLIENT_ID=$(echo "$CLIENT_OUTPUT" | jq -r '.UserPoolClient.ClientId')

if [ -z "$CLIENT_ID" ] || [ "$CLIENT_ID" == "null" ]; then
    echo "❌ Failed to create app client"
    exit 1
fi

echo "✓ App Client created: $CLIENT_ID"
echo ""

# Save configuration
CONFIG_FILE="$HOME/.adaptive/cognito_config.sh"
mkdir -p "$HOME/.adaptive"

cat > "$CONFIG_FILE" << EOF
# Adaptive Cognito Configuration
# Generated on $(date)

export ADAPTIVE_COGNITO_USER_POOL_ID="$USER_POOL_ID"
export ADAPTIVE_COGNITO_CLIENT_ID="$CLIENT_ID"
export ADAPTIVE_COGNITO_REGION="$AWS_REGION"
EOF

chmod 600 "$CONFIG_FILE"

echo "✓ Configuration saved to: $CONFIG_FILE"
echo ""

# Display summary
echo "======================================================================="
echo "✅ SETUP COMPLETE"
echo "======================================================================="
echo ""
echo "Configuration:"
echo "  User Pool ID: $USER_POOL_ID"
echo "  Client ID:    $CLIENT_ID"
echo "  Region:       $AWS_REGION"
echo ""
echo "Next steps:"
echo ""
echo "1. Add these to your shell configuration (~/.bashrc or ~/.zshrc):"
echo ""
echo "   source $CONFIG_FILE"
echo ""
echo "2. Apply changes:"
echo ""
echo "   source ~/.bashrc  # or ~/.zshrc"
echo ""
echo "3. Test account creation:"
echo ""
echo "   adaptive auth signup"
echo ""
echo "4. View users in AWS Console:"
echo ""
echo "   https://console.aws.amazon.com/cognito/v2/idp/user-pools/$USER_POOL_ID/users"
echo ""
echo "======================================================================="
echo ""
