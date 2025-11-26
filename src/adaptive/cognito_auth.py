"""
AWS Cognito authentication for Adaptive CLI.
Supports both email/password and Google OAuth through Cognito.
"""

import json
import logging
import os
import webbrowser
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from warrant_lite import WarrantLite


logger = logging.getLogger(__name__)


# Configuration - read from environment variables
COGNITO_USER_POOL_ID = os.getenv("ADAPTIVE_COGNITO_USER_POOL_ID", "")
COGNITO_CLIENT_ID = os.getenv("ADAPTIVE_COGNITO_CLIENT_ID", "")
COGNITO_REGION = os.getenv("ADAPTIVE_COGNITO_REGION", "us-east-1")
COGNITO_DOMAIN = os.getenv("ADAPTIVE_COGNITO_DOMAIN", "")

# Authentication directory
AUTH_DIR = Path.home() / ".adaptive"
AUTH_FILE = AUTH_DIR / "auth.json"


class CognitoAuthManager:
    """Manages AWS Cognito authentication with email/password and Google OAuth."""

    def __init__(self):
        """Initialize Cognito authentication manager."""
        self._ensure_auth_dir()
        self._auth_data = self._load_auth()
        self._cognito_client = boto3.client("cognito-idp", region_name=COGNITO_REGION)

    def _ensure_auth_dir(self) -> None:
        """Ensure authentication directory exists."""
        AUTH_DIR.mkdir(parents=True, exist_ok=True)
        AUTH_DIR.chmod(0o700)

    def _load_auth(self) -> dict:
        """Load authentication data from file."""
        if not AUTH_FILE.exists():
            return {}

        try:
            with open(AUTH_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load auth data: {e}")
            return {}

    def _save_auth(self) -> None:
        """Save authentication data to file."""
        try:
            with open(AUTH_FILE, "w") as f:
                json.dump(self._auth_data, f, indent=2)
            AUTH_FILE.chmod(0o600)
            logger.info("Authentication data saved")
        except Exception as e:
            logger.error(f"Failed to save auth data: {e}")
            raise

    def _check_config(self) -> bool:
        """
        Check if Cognito is properly configured.

        Returns:
            True if configured
        """
        if not COGNITO_USER_POOL_ID or not COGNITO_CLIENT_ID:
            print("\n❌ AWS Cognito is not configured")
            print("\nPlease set the following environment variables:")
            print("  - ADAPTIVE_COGNITO_USER_POOL_ID")
            print("  - ADAPTIVE_COGNITO_CLIENT_ID")
            print("  - ADAPTIVE_COGNITO_REGION (optional, defaults to us-east-1)")
            print("  - ADAPTIVE_COGNITO_DOMAIN (optional, for Google OAuth)")
            print("\nSee documentation for setup instructions.")
            return False
        return True

    def is_authenticated(self) -> bool:
        """
        Check if user is authenticated.

        Returns:
            True if authenticated
        """
        return (
            "email" in self._auth_data
            and "access_token" in self._auth_data
            and "refresh_token" in self._auth_data
        )

    def get_user_email(self) -> Optional[str]:
        """
        Get authenticated user's email.

        Returns:
            User email or None
        """
        return self._auth_data.get("email")

    def get_user_name(self) -> Optional[str]:
        """
        Get authenticated user's name.

        Returns:
            User name or None
        """
        return self._auth_data.get("name")

    def get_access_token(self) -> Optional[str]:
        """
        Get access token for API calls.

        Returns:
            Access token or None
        """
        return self._auth_data.get("access_token")

    def signup_with_email(
        self,
        email: str,
        password: str,
        name: Optional[str] = None,
    ) -> bool:
        """
        Create new account with email and password.

        Args:
            email: User email
            password: User password
            name: User name (optional)

        Returns:
            True if successful

        Raises:
            Exception: If signup fails
        """
        if not self._check_config():
            return False

        print("\n" + "=" * 70)
        print("CREATE ADAPTIVE ACCOUNT")
        print("=" * 70)
        print(f"\nCreating account for: {email}")

        try:
            wl = WarrantLite(
                username=email,
                password=password,
                pool_id=COGNITO_USER_POOL_ID,
                client_id=COGNITO_CLIENT_ID,
                client_secret=None,
            )

            # Prepare user attributes
            user_attributes = {"email": email}
            if name:
                user_attributes["name"] = name

            # Register user
            wl.register(
                email,
                password,
                user_attributes=user_attributes,
            )

            print("\n✅ Account created successfully!")
            print("\n⚠️  Please check your email for a verification code.")
            print("You'll need to verify your email before logging in.\n")

            # Prompt for verification
            verify = input("Have you received the verification code? (Y/n): ").strip().lower()
            if verify in ["", "y", "yes"]:
                code = input("Enter verification code: ").strip()
                if self._verify_email(email, code):
                    print("\n✅ Email verified! You can now log in.\n")
                    # Auto-login after verification
                    return self.login_with_email(email, password)
            else:
                print("\nYou can verify later with: adaptive auth verify\n")

            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "UsernameExistsException":
                print("\n❌ An account with this email already exists")
                print("Try logging in with: adaptive auth login\n")
            elif error_code == "InvalidPasswordException":
                print("\n❌ Password does not meet requirements")
                print("Password must:")
                print("  - Be at least 8 characters long")
                print("  - Contain at least one uppercase letter")
                print("  - Contain at least one lowercase letter")
                print("  - Contain at least one number")
                print("  - Contain at least one special character\n")
            else:
                print(f"\n❌ Signup failed: {e.response['Error']['Message']}\n")
            logger.error(f"Cognito signup failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Signup failed: {e}")
            print(f"\n❌ Signup failed: {e}\n")
            return False

    def _verify_email(
        self,
        email: str,
        code: str,
    ) -> bool:
        """
        Verify email with confirmation code.

        Args:
            email: User email
            code: Verification code

        Returns:
            True if successful
        """
        try:
            self._cognito_client.confirm_sign_up(
                ClientId=COGNITO_CLIENT_ID,
                Username=email,
                ConfirmationCode=code,
            )
            return True
        except ClientError as e:
            logger.error(f"Email verification failed: {e}")
            print(f"\n❌ Verification failed: {e.response['Error']['Message']}\n")
            return False

    def verify_email_standalone(
        self,
        email: str,
        code: str,
    ) -> bool:
        """
        Verify email as standalone action.

        Args:
            email: User email
            code: Verification code

        Returns:
            True if successful
        """
        print("\n" + "=" * 70)
        print("VERIFY EMAIL")
        print("=" * 70)

        if self._verify_email(email, code):
            print("\n✅ Email verified successfully!")
            print("You can now log in with: adaptive auth login\n")
            return True
        return False

    def login_with_email(
        self,
        email: str,
        password: str,
    ) -> bool:
        """
        Login with email and password.

        Args:
            email: User email
            password: User password

        Returns:
            True if successful

        Raises:
            Exception: If login fails
        """
        if not self._check_config():
            return False

        print("\n" + "=" * 70)
        print("ADAPTIVE - EMAIL LOGIN")
        print("=" * 70)
        print(f"\nLogging in as: {email}")

        try:
            wl = WarrantLite(
                username=email,
                password=password,
                pool_id=COGNITO_USER_POOL_ID,
                client_id=COGNITO_CLIENT_ID,
                client_secret=None,
            )

            # Authenticate
            tokens = wl.authenticate(email, password)

            # Get user attributes
            user_info = self._cognito_client.get_user(
                AccessToken=tokens["access_token"]
            )

            # Extract user attributes
            attributes = {
                attr["Name"]: attr["Value"] for attr in user_info["UserAttributes"]
            }

            # Store authentication data
            self._auth_data = {
                "email": attributes.get("email", email),
                "name": attributes.get("name", ""),
                "access_token": tokens["access_token"],
                "id_token": tokens["id_token"],
                "refresh_token": tokens["refresh_token"],
                "auth_method": "email",
            }

            self._save_auth()

            print("\n" + "=" * 70)
            print("✅ LOGIN SUCCESSFUL")
            print("=" * 70)
            print(f"\nLogged in as: {self._auth_data['email']}")
            if self._auth_data.get("name"):
                print(f"Name: {self._auth_data['name']}")
            print("\nYou can now use Adaptive!\n")

            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NotAuthorizedException":
                print("\n❌ Incorrect email or password")
            elif error_code == "UserNotConfirmedException":
                print("\n❌ Email not verified")
                print("Please check your email for the verification code")
                print("Or run: adaptive auth verify\n")
            else:
                print(f"\n❌ Login failed: {e.response['Error']['Message']}\n")
            logger.error(f"Cognito login failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Login failed: {e}")
            print(f"\n❌ Login failed: {e}\n")
            return False

    def login_with_google(self) -> bool:
        """
        Login with Google OAuth through Cognito Hosted UI.

        Returns:
            True if successful
        """
        if not self._check_config():
            return False

        if not COGNITO_DOMAIN:
            print("\n❌ Google OAuth is not configured")
            print("Please set ADAPTIVE_COGNITO_DOMAIN environment variable")
            print("See documentation for setup instructions.\n")
            return False

        print("\n" + "=" * 70)
        print("ADAPTIVE - GOOGLE OAUTH")
        print("=" * 70)
        print("\n🔐 Opening browser for Google authentication...")
        print("If the browser doesn't open, please visit the URL shown below.\n")

        # Construct Cognito Hosted UI URL
        oauth_url = (
            f"https://{COGNITO_DOMAIN}.auth.{COGNITO_REGION}.amazoncognito.com/oauth2/authorize"
            f"?client_id={COGNITO_CLIENT_ID}"
            f"&response_type=code"
            f"&scope=email+openid+profile"
            f"&redirect_uri=http://localhost:8080/callback"
            f"&identity_provider=Google"
        )

        print(f"Please visit: {oauth_url}\n")

        try:
            webbrowser.open(oauth_url)

            print("\n⚠️  Note: This requires setting up a local callback server.")
            print("For now, please use email/password authentication.")
            print("Google OAuth will be fully supported in a future release.\n")

            return False

        except Exception as e:
            logger.error(f"Google OAuth failed: {e}")
            print(f"\n❌ Google OAuth failed: {e}\n")
            return False

    def refresh_tokens(self) -> bool:
        """
        Refresh access token using refresh token.

        Returns:
            True if successful
        """
        if not self.is_authenticated():
            return False

        refresh_token = self._auth_data.get("refresh_token")
        if not refresh_token:
            return False

        try:
            response = self._cognito_client.initiate_auth(
                ClientId=COGNITO_CLIENT_ID,
                AuthFlow="REFRESH_TOKEN_AUTH",
                AuthParameters={"REFRESH_TOKEN": refresh_token},
            )

            # Update tokens
            self._auth_data["access_token"] = response["AuthenticationResult"][
                "AccessToken"
            ]
            self._auth_data["id_token"] = response["AuthenticationResult"]["IdToken"]
            self._save_auth()

            logger.info("Tokens refreshed successfully")
            return True

        except ClientError as e:
            logger.error(f"Token refresh failed: {e}")
            return False

    def logout(self) -> bool:
        """
        Logout and remove credentials.

        Returns:
            True if successful
        """
        if AUTH_FILE.exists():
            AUTH_FILE.unlink()
            self._auth_data = {}
            print("\n✅ Logged out successfully\n")
            return True
        else:
            print("\n❌ No active session found\n")
            return False

    def show_status(self) -> None:
        """Display authentication status."""
        print("\n" + "=" * 70)
        print("AUTHENTICATION STATUS")
        print("=" * 70)

        if self.is_authenticated():
            print("\n✅ Logged in")
            print(f"Email: {self._auth_data.get('email')}")
            if self._auth_data.get("name"):
                print(f"Name:  {self._auth_data.get('name')}")
            print(f"Method: {self._auth_data.get('auth_method', 'unknown')}")
        else:
            print("\n❌ Not logged in")
            print("\nAuthentication options:")
            print("  • adaptive auth signup   - Create account with email")
            print("  • adaptive auth login    - Login with email/password")
            print("  • adaptive auth google   - Login with Google (if configured)")

        print("=" * 70 + "\n")

    def ensure_authenticated(self) -> bool:
        """
        Ensure user is authenticated, prompt if not.

        Returns:
            True if authenticated

        Raises:
            SystemExit: If authentication fails
        """
        if self.is_authenticated():
            user_name = self.get_user_name() or self.get_user_email()
            print(f"\n✓ Authenticated as: {user_name}")
            return True

        # First-time user - show welcome
        print("\n" + "=" * 70)
        print("WELCOME TO ADAPTIVE")
        print("=" * 70)
        print("\nAdaptive helps optimize your AI agents through intelligent")
        print("observability and automated code evolution.")
        print("\n" + "=" * 70)
        print("AUTHENTICATION REQUIRED")
        print("=" * 70)
        print("\nTo get started, you need to authenticate.")
        print("\nOptions:")
        print("  1. Create new account with email")
        print("  2. Login with existing account")
        print("  3. Login with Google (if configured)")
        print("\nThis is a one-time setup.\n")

        response = input("Choose option (1/2/3) or 'q' to quit: ").strip().lower()

        if response == "1":
            email = input("Email: ").strip()
            name = input("Name (optional): ").strip()
            import getpass

            password = getpass.getpass("Password: ")

            if self.signup_with_email(email, password, name or None):
                print("\n✅ You're all set! Continuing to Adaptive...\n")
                return True
        elif response == "2":
            email = input("Email: ").strip()
            import getpass

            password = getpass.getpass("Password: ")

            if self.login_with_email(email, password):
                print("\n✅ You're all set! Continuing to Adaptive...\n")
                return True
        elif response == "3":
            if self.login_with_google():
                print("\n✅ You're all set! Continuing to Adaptive...\n")
                return True
        elif response == "q":
            print("\n❌ Authentication is required to use Adaptive\n")
            raise SystemExit(1)

        print("\n❌ Authentication failed. Please try again with: adaptive auth login\n")
        raise SystemExit(1)
