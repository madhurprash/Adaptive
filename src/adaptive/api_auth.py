"""
API-based authentication for Adaptive CLI.
Users interact with the backend API instead of directly with AWS Cognito.
No AWS credentials or configuration required for end users.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import requests


logger = logging.getLogger(__name__)


# Backend API configuration
API_BASE_URL: str = os.getenv(
    "ADAPTIVE_API_URL",
    "http://127.0.0.1:8000",  # Default to local development
)

# Authentication directory
AUTH_DIR = Path.home() / ".adaptive"
AUTH_FILE = AUTH_DIR / "auth.json"


class APIAuthManager:
    """
    API-based authentication manager.
    Communicates with Adaptive backend API for all auth operations.
    """

    def __init__(self):
        """Initialize API authentication manager."""
        self._ensure_auth_dir()
        self._auth_data = self._load_auth()

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
        Create new account via API.

        Args:
            email: User email
            password: User password
            name: User name (optional)

        Returns:
            True if successful
        """
        print("\n" + "=" * 70)
        print("CREATE ADAPTIVE ACCOUNT")
        print("=" * 70)
        print(f"\nCreating account for: {email}")

        try:
            response = requests.post(
                f"{API_BASE_URL}/auth/signup",
                json={
                    "email": email,
                    "password": password,
                    "name": name,
                },
                timeout=30,
            )

            if response.status_code == 201:
                data = response.json()
                print(f"\n✅ {data['message']}")

                # Prompt for verification
                verify = (
                    input("\nHave you received the verification code? (Y/n): ")
                    .strip()
                    .lower()
                )
                if verify in ["", "y", "yes"]:
                    code = input("Enter verification code: ").strip()
                    if self._verify_email(email, code):
                        print("\n✅ Email verified! You can now log in.\n")
                        # Auto-login after verification
                        return self.login_with_email(email, password)
                else:
                    print("\nYou can verify later with: adaptive auth verify\n")

                return True
            else:
                error_data = response.json()
                print(f"\n❌ {error_data.get('detail', 'Signup failed')}\n")
                return False

        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            print(f"\n❌ Connection error: {e}")
            print(f"Make sure the backend API is running at: {API_BASE_URL}\n")
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
        Verify email with confirmation code via API.

        Args:
            email: User email
            code: Verification code

        Returns:
            True if successful
        """
        try:
            response = requests.post(
                f"{API_BASE_URL}/auth/verify",
                json={"email": email, "code": code},
                timeout=30,
            )

            if response.status_code == 200:
                return True
            else:
                error_data = response.json()
                print(f"\n❌ {error_data.get('detail', 'Verification failed')}\n")
                return False

        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            print(f"\n❌ Connection error: {e}\n")
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
        Login with email and password via API.

        Args:
            email: User email
            password: User password

        Returns:
            True if successful
        """
        print("\n" + "=" * 70)
        print("ADAPTIVE - LOGIN")
        print("=" * 70)
        print(f"\nLogging in as: {email}")

        try:
            response = requests.post(
                f"{API_BASE_URL}/auth/login",
                json={"email": email, "password": password},
                timeout=30,
            )

            if response.status_code == 200:
                tokens = response.json()

                # Store authentication data
                self._auth_data = {
                    "email": tokens["user_email"],
                    "name": tokens.get("user_name"),
                    "access_token": tokens["access_token"],
                    "id_token": tokens["id_token"],
                    "refresh_token": tokens["refresh_token"],
                    "auth_method": "api",
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
            else:
                error_data = response.json()
                print(f"\n❌ {error_data.get('detail', 'Login failed')}\n")
                return False

        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            print(f"\n❌ Connection error: {e}")
            print(f"Make sure the backend API is running at: {API_BASE_URL}\n")
            return False
        except Exception as e:
            logger.error(f"Login failed: {e}")
            print(f"\n❌ Login failed: {e}\n")
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
        print("\nThis is a one-time setup.\n")

        response = input("Choose option (1/2) or 'q' to quit: ").strip().lower()

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
        elif response == "q":
            print("\n❌ Authentication is required to use Adaptive\n")
            raise SystemExit(1)

        print(
            "\n❌ Authentication failed. Please try again with: adaptive auth login\n"
        )
        raise SystemExit(1)

    def call_api(
        self,
        endpoint: str,
        method: str = "GET",
        **kwargs,
    ) -> dict:
        """
        Make authenticated API call.

        Args:
            endpoint: API endpoint (e.g., "/api/user/profile")
            method: HTTP method
            **kwargs: Additional arguments for requests

        Returns:
            Response JSON

        Raises:
            Exception: If request fails
        """
        token = self.get_access_token()
        if not token:
            raise Exception("Not authenticated. Run 'adaptive auth login'")

        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {token}"

        response = requests.request(
            method,
            f"{API_BASE_URL}{endpoint}",
            headers=headers,
            timeout=30,
            **kwargs,
        )

        if response.status_code == 401:
            print("❌ Session expired. Please login again.")
            raise SystemExit(1)

        response.raise_for_status()
        return response.json()
