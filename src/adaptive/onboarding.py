"""
Onboarding flow for new Adaptive users.
Guides users through authentication and initial setup.
"""

import getpass
import logging
import os
from typing import Optional

from .cognito_auth import CognitoAuthManager
from .config_manager import ConfigManager


logger = logging.getLogger(__name__)


SUPPORTED_PLATFORMS = ["langsmith", "langfuse", "mlflow"]

PLATFORM_ENV_VARS = {
    "langsmith": "LANGSMITH_API_KEY",
    "langfuse": "LANGFUSE_API_KEY",
    "mlflow": "MLFLOW_TRACKING_URI",
}


def _print_header(
    title: str,
) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70 + "\n")


def _print_step(
    step_num: int,
    title: str,
) -> None:
    """Print a step header."""
    print(f"\n{'─' * 70}")
    print(f"STEP {step_num}: {title}")
    print(f"{'─' * 70}\n")


def run_onboarding() -> bool:
    """
    Run the complete onboarding flow.

    Returns:
        True if onboarding completed successfully
    """
    _print_header("WELCOME TO ADAPTIVE")

    print("Adaptive helps optimize your AI agents through intelligent")
    print("observability and automated code evolution.\n")
    print("Let's get you set up! This will take about 2 minutes.\n")

    # Step 1: Authentication
    _print_step(1, "AUTHENTICATION")
    if not _handle_authentication():
        return False

    # Step 2: Platform Selection
    _print_step(2, "PLATFORM SELECTION")
    platform = _select_platform()
    if not platform:
        return False

    # Step 3: API Key Configuration
    _print_step(3, "API KEY CONFIGURATION")
    if not _configure_api_key(platform):
        return False

    # Success!
    _print_header("SETUP COMPLETE")
    print("✅ You're all set! You can now use Adaptive.\n")
    print("Next steps:")
    print("  • Run 'adaptive run' to start the agent")
    print("  • Run 'adaptive config show' to view your configuration")
    print("  • Run 'adaptive --help' for more options\n")

    return True


def _handle_authentication() -> bool:
    """
    Handle user authentication with email/password or Google.

    Returns:
        True if authenticated successfully
    """
    auth = CognitoAuthManager()

    # Check if already authenticated
    if auth.is_authenticated():
        user_name = auth.get_user_name() or auth.get_user_email()
        print(f"✓ Already authenticated as: {user_name}\n")

        response = input("Continue with this account? (Y/n): ").strip().lower()
        if response in ["", "y", "yes"]:
            return True

        # User wants to switch accounts
        print("\nLogging out...")
        auth.logout()

    # Authenticate
    print("Choose authentication method:\n")
    print("  1. Create new account with email")
    print("  2. Login with existing email account")
    print("  3. Login with Google (if configured)\n")

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        return _signup_with_email(auth)
    elif choice == "2":
        return _login_with_email(auth)
    elif choice == "3":
        return _login_with_google(auth)
    else:
        print("\n❌ Invalid choice")
        print("Run 'adaptive' again when you're ready to continue\n")
        return False


def _signup_with_email(
    auth: CognitoAuthManager,
) -> bool:
    """
    Handle email signup flow.

    Args:
        auth: Authentication manager

    Returns:
        True if successful
    """
    print("\n" + "─" * 70)
    print("CREATE ACCOUNT")
    print("─" * 70 + "\n")

    email = input("Email: ").strip()
    if not email:
        print("❌ Email is required")
        return False

    name = input("Name (optional): ").strip()

    print("\nPassword requirements:")
    print("  • At least 8 characters")
    print("  • One uppercase letter")
    print("  • One lowercase letter")
    print("  • One number")
    print("  • One special character\n")

    password = getpass.getpass("Password: ")
    if not password:
        print("❌ Password is required")
        return False

    confirm_password = getpass.getpass("Confirm password: ")
    if password != confirm_password:
        print("❌ Passwords do not match")
        return False

    return auth.signup_with_email(email, password, name or None)


def _login_with_email(
    auth: CognitoAuthManager,
) -> bool:
    """
    Handle email login flow.

    Args:
        auth: Authentication manager

    Returns:
        True if successful
    """
    print("\n" + "─" * 70)
    print("LOGIN")
    print("─" * 70 + "\n")

    email = input("Email: ").strip()
    if not email:
        print("❌ Email is required")
        return False

    password = getpass.getpass("Password: ")
    if not password:
        print("❌ Password is required")
        return False

    return auth.login_with_email(email, password)


def _login_with_google(
    auth: CognitoAuthManager,
) -> bool:
    """
    Handle Google OAuth login flow.

    Args:
        auth: Authentication manager

    Returns:
        True if successful
    """
    print("\n🔄 Opening browser for Google authentication...\n")
    return auth.login_with_google()


def _select_platform() -> Optional[str]:
    """
    Let user select observability platform.

    Returns:
        Selected platform name or None
    """
    config = ConfigManager()

    # Check if platform already configured
    existing_platform = config.get("platform")
    if existing_platform and existing_platform in SUPPORTED_PLATFORMS:
        print(f"Current platform: {existing_platform}\n")
        response = input("Keep this platform? (Y/n): ").strip().lower()
        if response in ["", "y", "yes"]:
            return existing_platform

    # Select new platform
    print("Choose your observability platform:\n")
    for i, platform in enumerate(SUPPORTED_PLATFORMS, 1):
        print(f"  {i}. {platform.capitalize()}")

    print()
    while True:
        choice = input(f"Enter choice (1-{len(SUPPORTED_PLATFORMS)}): ").strip()

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(SUPPORTED_PLATFORMS):
                platform = SUPPORTED_PLATFORMS[idx]
                config.set("platform", platform)
                print(f"\n✅ Platform set to: {platform}")
                return platform
        except ValueError:
            pass

        print(f"❌ Please enter a number between 1 and {len(SUPPORTED_PLATFORMS)}")


def _configure_api_key(
    platform: str,
) -> bool:
    """
    Configure API key for the selected platform.

    Args:
        platform: Platform name

    Returns:
        True if configured successfully
    """
    config = ConfigManager()

    # Check if API key already exists
    existing_key = config.get_api_key(platform)
    if existing_key:
        print(f"✓ API key for {platform} is already configured\n")
        response = input("Update API key? (y/N): ").strip().lower()
        if response not in ["y", "yes"]:
            return True

    # Check environment variable first
    env_var = PLATFORM_ENV_VARS.get(platform)
    if env_var:
        env_value = os.getenv(env_var)
        if env_value:
            print(f"Found {env_var} environment variable\n")
            response = input("Use this API key? (Y/n): ").strip().lower()
            if response in ["", "y", "yes"]:
                config.set_api_key(platform, env_value)
                print(f"\n✅ API key for {platform} configured from environment")
                return True

    # Prompt for API key
    print(f"\nEnter your {platform.capitalize()} API key.")
    print("(Your input will be hidden for security)\n")

    api_key = getpass.getpass(f"{platform.capitalize()} API Key: ")

    if not api_key or not api_key.strip():
        print("\n❌ API key is required")
        print(f"You can set it later with: adaptive config set-key {platform}\n")
        return False

    # Store the API key
    config.set_api_key(platform, api_key.strip())
    print(f"\n✅ API key for {platform} stored successfully")
    return True


def needs_onboarding() -> bool:
    """
    Check if user needs to go through onboarding.

    Returns:
        True if onboarding is needed
    """
    auth = CognitoAuthManager()
    config = ConfigManager()

    # Check authentication
    if not auth.is_authenticated():
        return True

    # Check platform configuration
    platform = config.get("platform")
    if not platform or platform not in SUPPORTED_PLATFORMS:
        return True

    # Check if API key is configured for the platform
    if not config.get_api_key(platform):
        return True

    return False
