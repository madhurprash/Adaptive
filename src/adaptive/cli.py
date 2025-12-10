"""
Adaptive CLI - Simplified version inspired by Claude Code.

Simple, elegant CLI for Adaptive agent optimization system.
"""
import argparse
import getpass
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

from .cognito_auth import CognitoAuthManager
from .config_manager import ConfigManager
from .onboarding import needs_onboarding, run_onboarding


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__).resolve()

    # Try to find the project root by looking for pyproject.toml
    for parent in current_file.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    # If not found, use the current directory
    return Path.cwd()


def _run_agent(
    debug: bool = False,
    session_id: Optional[str] = None,
    skip_onboarding: bool = False,
    dev_mode: bool = False,
) -> int:
    """Run the Adaptive agent."""
    try:
        # Dev mode bypasses authentication
        if dev_mode:
            print("\n⚠️  DEV MODE ENABLED - Authentication bypassed")
            print("⚠️  This should ONLY be used for local development\n")
        # Check if onboarding is needed (unless explicitly skipped or in dev mode)
        elif not skip_onboarding and needs_onboarding():
            print("\n🚀 First time setup detected\n")
            if not run_onboarding():
                print("\n❌ Setup incomplete. Please try again.\n")
                return 1
        else:
            # Just ensure authentication for returning users
            auth = CognitoAuthManager()
            if not auth.is_authenticated():
                print("\n❌ Not authenticated. Run 'adaptive auth login'\n")
                return 1

            # Show quick welcome
            user_name = auth.get_user_name() or auth.get_user_email()
            print(f"\n✓ Authenticated as: {user_name}")

        # Set logging level
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

        # Generate session ID if not provided
        if not session_id:
            session_id = f"session-{uuid.uuid4().hex[:8]}"
            logger.info(f"Generated new session ID: {session_id}")

        # Get project root
        project_root = _get_project_root()
        logger.info(f"Project root: {project_root}")

        # Add project root to Python path for imports
        sys.path.insert(0, str(project_root))

        # Import the main adaptive module
        from adaptive import main as adaptive_main

        logger.info("Starting Adaptive agent...")

        # Call the adaptive main function
        adaptive_main(
            session_id=session_id,
            debug=debug,
            parse_cli_args=False,
        )

        logger.info("Adaptive agent completed successfully")
        return 0

    except Exception as e:
        logger.exception(f"Error running Adaptive agent: {e}")
        return 1


def _cmd_config_show() -> int:
    """Show current configuration."""
    config = ConfigManager()
    config.show()
    return 0


def _cmd_config_set(args) -> int:
    """Set a configuration value."""
    config = ConfigManager()

    if args.key == "model":
        config.set("model_id", args.value)
        print(f"✅ Model set to: {args.value}")
    elif args.key == "platform":
        if args.value not in ["langsmith", "langfuse", "mlflow"]:
            print("❌ Platform must be one of: langsmith, langfuse, mlflow")
            return 1
        config.set("platform", args.value)
        print(f"✅ Platform set to: {args.value}")
    elif args.key == "temperature":
        config.set("temperature", args.value)
        print(f"✅ Temperature set to: {args.value}")
    elif args.key == "max_tokens":
        config.set("max_tokens", args.value)
        print(f"✅ Max tokens set to: {args.value}")
    else:
        print(f"❌ Unknown configuration key: {args.key}")
        print("Available keys: model, platform, temperature, max_tokens")
        return 1

    return 0


def _cmd_config_set_api_key(args) -> int:
    """Set an API key for a platform."""
    config = ConfigManager()

    # If API key not provided, prompt for it
    api_key = args.value
    if not api_key:
        api_key = getpass.getpass(f"Enter API key for {args.platform}: ")

    if not api_key:
        print("❌ API key is required")
        return 1

    config.set_api_key(args.platform, api_key)
    print(f"✅ API key for {args.platform} stored successfully")
    return 0


def _cmd_config_list_keys() -> int:
    """List stored API keys."""
    config = ConfigManager()
    platforms = config.list_api_keys()

    print("\n" + "=" * 70)
    print("STORED API KEYS")
    print("=" * 70)
    if platforms:
        for platform in platforms:
            print(f"  ✓ {platform}")
    else:
        print("  No API keys stored")
    print("=" * 70 + "\n")
    return 0


def _cmd_config_delete_key(args) -> int:
    """Delete an API key."""
    config = ConfigManager()

    # Confirm deletion
    confirm = input(f"Delete API key for {args.platform}? (y/N): ")
    if confirm.lower() != 'y':
        print("Cancelled")
        return 0

    if config.delete_api_key(args.platform):
        print(f"✅ API key for {args.platform} deleted")
    else:
        print(f"❌ No API key found for {args.platform}")
        return 1

    return 0


def _cmd_auth_signup(args) -> int:
    """Create new account with email."""
    # Use Cognito directly for CLI tool
    auth = CognitoAuthManager()

    email = args.email or input("Email: ").strip()
    if not email:
        print("❌ Email is required")
        return 1

    name = args.name or input("Name (optional): ").strip()

    if not args.password:
        print("\nPassword requirements:")
        print("  • At least 8 characters")
        print("  • One uppercase letter")
        print("  • One lowercase letter")
        print("  • One number")
        print("  • One special character\n")

    password = args.password or getpass.getpass("Password: ")
    if not password:
        print("❌ Password is required")
        return 1

    if not args.password:
        confirm_password = getpass.getpass("Confirm password: ")
        if password != confirm_password:
            print("❌ Passwords do not match")
            return 1

    if auth.signup_with_email(email, password, name or None):
        return 0
    return 1


def _cmd_auth_login(args) -> int:
    """Log in with email/password."""
    auth = CognitoAuthManager()

    email = args.email or input("Email: ").strip()
    if not email:
        print("❌ Email is required")
        return 1

    password = args.password or getpass.getpass("Password: ")
    if not password:
        print("❌ Password is required")
        return 1

    if auth.login_with_email(email, password):
        return 0
    return 1


def _cmd_auth_google() -> int:
    """Log in with Google OAuth."""
    auth = CognitoAuthManager()
    if auth.login_with_google():
        return 0
    return 1


def _cmd_auth_verify(args) -> int:
    """Verify email with confirmation code."""
    auth = CognitoAuthManager()

    email = args.email or input("Email: ").strip()
    if not email:
        print("❌ Email is required")
        return 1

    code = args.code or input("Verification code: ").strip()
    if not code:
        print("❌ Verification code is required")
        return 1

    if auth.verify_email_standalone(email, code):
        return 0
    return 1


def _cmd_auth_logout() -> int:
    """Log out."""
    auth = CognitoAuthManager()

    if auth.logout():
        return 0
    return 1


def _cmd_auth_status() -> int:
    """Show authentication status."""
    auth = CognitoAuthManager()
    auth.show_status()
    return 0


def _cmd_generate_tasks(args) -> int:
    """Generate synthetic tasks from execution history."""
    try:
        # Set logging level
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

        # Get project root
        project_root = _get_project_root()
        sys.path.insert(0, str(project_root))

        # Import the task generation orchestrator
        from evolution.self_questioning.task_generation_orchestrator import (
            TaskGenerationOrchestrator
        )

        print("\n🔍 Generating synthetic tasks from execution history...\n")

        # Initialize orchestrator
        orchestrator = TaskGenerationOrchestrator()

        # Generate tasks
        tasks = orchestrator.generate_tasks(
            platform=args.platform,
            max_tasks=args.max_tasks,
            task_type=args.task_type,
            agent_repo_path=args.agent_repo,
        )

        # Display results
        print(f"\n✅ Generated {len(tasks)} synthetic tasks\n")

        if tasks:
            print("Task Summary:")
            print("=" * 70)
            for i, task in enumerate(tasks, 1):
                print(f"{i}. [{task.task_type.upper()}] {task.description[:60]}...")
                print(f"   Difficulty: {task.difficulty} | ID: {task.task_id}")
                print()

            print(f"✓ Tasks stored in AgentCore Memory")
            print(f"✓ Run 'adaptive evaluate-tasks' to evaluate agent performance\n")
        else:
            print("⚠️  No tasks generated. Try increasing execution history or adjusting filters.\n")

        return 0

    except Exception as e:
        logger.exception(f"Error generating tasks: {e}")
        print(f"\n❌ Error generating tasks: {e}\n")
        return 1


def _show_version() -> None:
    """Display version information."""
    try:
        from adaptive import __version__
        print(f"Adaptive v{__version__}")
    except ImportError:
        print("Adaptive (version unknown)")


def main() -> int:
    """Main CLI entry point."""
    # If no arguments provided, run the agent (with onboarding if needed)
    if len(sys.argv) == 1:
        return _run_agent()

    parser = argparse.ArgumentParser(
        prog="adaptive",
        description="Adaptive - Continuous optimization for AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # First time - Just run adaptive (will guide you through setup)
    adaptive

    # Run agent explicitly
    adaptive run

    # View configuration and auth status
    adaptive config show
    adaptive auth status

    # Manual configuration (optional)
    adaptive config set model us.anthropic.claude-sonnet-4-20250514-v1:0
    adaptive config set platform langsmith
    adaptive config set-key langsmith

    # Manage API keys
    adaptive config set-key langfuse
    adaptive config list-keys

    # Generate synthetic tasks from execution history
    adaptive generate-tasks --platform langsmith --max-tasks 20
    adaptive generate-tasks --task-type edge_case --agent-repo /path/to/agent

For more information, visit: https://github.com/madhurprash/adaptive
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ========================================================================
    # AUTH COMMAND
    # ========================================================================
    auth_parser = subparsers.add_parser("auth", help="Manage authentication")
    auth_subparsers = auth_parser.add_subparsers(dest="auth_command", required=True)

    # auth signup
    signup_parser = auth_subparsers.add_parser("signup", help="Create new account with email")
    signup_parser.add_argument("--email", type=str, help="Email address")
    signup_parser.add_argument("--password", type=str, help="Password (will prompt if not provided)")
    signup_parser.add_argument("--name", type=str, help="Your name (optional)")

    # auth login
    login_parser = auth_subparsers.add_parser("login", help="Log in with email/password")
    login_parser.add_argument("--email", type=str, help="Email address")
    login_parser.add_argument("--password", type=str, help="Password (will prompt if not provided)")

    # auth google
    auth_subparsers.add_parser("google", help="Log in with Google OAuth")

    # auth verify
    verify_parser = auth_subparsers.add_parser("verify", help="Verify email with confirmation code")
    verify_parser.add_argument("--email", type=str, help="Email address")
    verify_parser.add_argument("--code", type=str, help="Verification code")

    # auth logout
    auth_subparsers.add_parser("logout", help="Log out")

    # auth status
    auth_subparsers.add_parser("status", help="Show authentication status")

    # ========================================================================
    # CONFIG COMMAND
    # ========================================================================
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)

    # config show
    config_subparsers.add_parser("show", help="Show current configuration")

    # config set
    set_parser = config_subparsers.add_parser("set", help="Set configuration value")
    set_parser.add_argument("key", help="Configuration key (model, platform, temperature, max_tokens)")
    set_parser.add_argument("value", help="Configuration value")

    # config set-key
    set_key_parser = config_subparsers.add_parser("set-key", help="Store API key for platform")
    set_key_parser.add_argument("platform", help="Platform name (langsmith, langfuse, databricks, etc.)")
    set_key_parser.add_argument("value", nargs="?", help="API key (will prompt if not provided)")

    # config list-keys
    config_subparsers.add_parser("list-keys", help="List platforms with stored API keys")

    # config delete-key
    delete_key_parser = config_subparsers.add_parser("delete-key", help="Delete API key for platform")
    delete_key_parser.add_argument("platform", help="Platform name")

    # ========================================================================
    # RUN COMMAND
    # ========================================================================
    run_parser = subparsers.add_parser("run", help="Run the agent")
    run_parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    run_parser.add_argument("--session-id", type=str, help="Session ID (auto-generated if not provided)")
    run_parser.add_argument("--skip-onboarding", action="store_true", help="Skip onboarding checks")

    # ========================================================================
    # GENERATE-TASKS COMMAND
    # ========================================================================
    generate_tasks_parser = subparsers.add_parser(
        "generate-tasks",
        help="Generate synthetic tasks from execution history"
    )
    generate_tasks_parser.add_argument(
        "--platform",
        type=str,
        choices=["langsmith", "langfuse", "mlflow"],
        help="Observability platform to analyze (optional, will analyze all if not specified)"
    )
    generate_tasks_parser.add_argument(
        "--max-tasks",
        type=int,
        default=10,
        help="Maximum number of tasks to generate (default: 10)"
    )
    generate_tasks_parser.add_argument(
        "--task-type",
        type=str,
        choices=["exploration", "edge_case", "optimization", "regression"],
        help="Specific task type to generate (optional, will generate mixed types if not specified)"
    )
    generate_tasks_parser.add_argument(
        "--agent-repo",
        type=str,
        help="Path to agent repository to analyze (optional)"
    )
    generate_tasks_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    # ========================================================================
    # VERSION COMMAND
    # ========================================================================
    subparsers.add_parser("version", help="Show version")

    # Parse arguments
    args = parser.parse_args()

    # Handle commands
    if not args.command:
        parser.print_help()
        return 0

    if args.command == "auth":
        if args.auth_command == "signup":
            return _cmd_auth_signup(args)
        elif args.auth_command == "login":
            return _cmd_auth_login(args)
        elif args.auth_command == "google":
            return _cmd_auth_google()
        elif args.auth_command == "verify":
            return _cmd_auth_verify(args)
        elif args.auth_command == "logout":
            return _cmd_auth_logout()
        elif args.auth_command == "status":
            return _cmd_auth_status()

    elif args.command == "config":
        if args.config_command == "show":
            return _cmd_config_show()
        elif args.config_command == "set":
            return _cmd_config_set(args)
        elif args.config_command == "set-key":
            return _cmd_config_set_api_key(args)
        elif args.config_command == "list-keys":
            return _cmd_config_list_keys()
        elif args.config_command == "delete-key":
            return _cmd_config_delete_key(args)

    elif args.command == "run":
        return _run_agent(
            debug=args.debug,
            session_id=args.session_id,
            skip_onboarding=args.skip_onboarding,
        )

    elif args.command == "generate-tasks":
        return _cmd_generate_tasks(args)

    elif args.command == "version":
        _show_version()
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
