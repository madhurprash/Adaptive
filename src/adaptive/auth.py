"""
Simple authentication for Adaptive CLI.
Supports Google OAuth for account creation and login.
"""

import json
import logging
import webbrowser
from pathlib import Path
from typing import Optional

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from google_auth_oauthlib import flow


logger = logging.getLogger(__name__)


# Configuration
AUTH_DIR = Path.home() / ".adaptive"
AUTH_FILE = AUTH_DIR / "auth.json"

# Google OAuth configuration
GOOGLE_CLIENT_ID = "your-client-id.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "your-client-secret"
OAUTH_SCOPES = ["openid", "email", "profile"]


class AuthManager:
    """Manages user authentication and credentials."""

    def __init__(self):
        """Initialize authentication manager."""
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
        return "email" in self._auth_data and "id_token" in self._auth_data

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

    def login_with_google(self) -> bool:
        """
        Authenticate with Google OAuth.

        Returns:
            True if successful

        Raises:
            Exception: If authentication fails
        """
        print("\n" + "=" * 70)
        print("ADAPTIVE - GOOGLE AUTHENTICATION")
        print("=" * 70)
        print("\n🔐 Opening browser for Google authentication...")
        print("If the browser doesn't open, please visit the URL shown below.\n")

        try:
            # Create OAuth flow
            oauth_flow = flow.InstalledAppFlow.from_client_config(
                {
                    "installed": {
                        "client_id": GOOGLE_CLIENT_ID,
                        "client_secret": GOOGLE_CLIENT_SECRET,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": ["http://localhost:8080"],
                    }
                },
                scopes=OAUTH_SCOPES,
            )

            # Run local server for OAuth callback
            credentials = oauth_flow.run_local_server(
                port=8080,
                prompt="consent",
                success_message="✅ Authentication successful! You can close this window and return to the terminal.",
            )

            # Verify and decode ID token
            idinfo = id_token.verify_oauth2_token(
                credentials.id_token,
                google_requests.Request(),
                GOOGLE_CLIENT_ID,
            )

            # Store user information
            self._auth_data = {
                "email": idinfo.get("email"),
                "name": idinfo.get("name"),
                "picture": idinfo.get("picture"),
                "id_token": credentials.id_token,
                "refresh_token": credentials.refresh_token,
            }

            self._save_auth()

            print("\n" + "=" * 70)
            print("✅ AUTHENTICATION SUCCESSFUL")
            print("=" * 70)
            print(f"\nLogged in as: {self._auth_data['name']} ({self._auth_data['email']})")
            print("\nYou can now use Adaptive!\n")

            return True

        except Exception as e:
            logger.error(f"Google OAuth failed: {e}")
            print(f"\n❌ Authentication failed: {e}\n")
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
            print(f"\n✅ Logged in")
            print(f"Email: {self._auth_data.get('email')}")
            print(f"Name:  {self._auth_data.get('name')}")
        else:
            print("\n❌ Not logged in")
            print("\nRun 'adaptive auth login' to authenticate with Google")

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
            # Show welcome message for authenticated users
            user_name = self.get_user_name() or self.get_user_email()
            print(f"\n✓ Authenticated as: {user_name}")
            return True

        # First-time user - show welcome and auth prompt
        print("\n" + "=" * 70)
        print("WELCOME TO ADAPTIVE")
        print("=" * 70)
        print("\nAdaptive helps optimize your AI agents through intelligent")
        print("observability and automated code evolution.")
        print("\n" + "=" * 70)
        print("AUTHENTICATION REQUIRED")
        print("=" * 70)
        print("\nTo get started, you need to authenticate with Google.")
        print("This will:")
        print("  • Create your Adaptive account (if new)")
        print("  • Or log you into your existing account")
        print("  • Takes less than 30 seconds")
        print("\nThis is a one-time setup.\n")

        response = input("Authenticate with Google now? (Y/n): ").strip().lower()
        if response in ["", "y", "yes"]:
            print("\n🔄 Starting authentication flow...\n")
            if self.login_with_google():
                print("\n✅ You're all set! Continuing to Adaptive...\n")
                return True
            else:
                print("\n❌ Authentication failed. Please try again with: adaptive auth login\n")
                raise SystemExit(1)
        else:
            print("\n❌ Authentication is required to use Adaptive")
            print("Run 'adaptive auth login' when you're ready\n")
            raise SystemExit(1)
