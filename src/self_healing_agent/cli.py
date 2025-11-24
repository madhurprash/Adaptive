"""
Self-Healing Agent CLI

Main command-line interface for the self-healing agent system.
"""
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _get_project_root() -> Path:
    """Get the project root directory."""
    # When installed, we need to find the config files
    # Check if we're in development mode or installed
    current_file = Path(__file__).resolve()

    # Try to find the project root by looking for pyproject.toml
    for parent in current_file.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    # If not found, use the current directory
    return Path.cwd()


def _run_agent(
    config_file: Optional[str] = None,
    debug: bool = False,
) -> int:
    """Run the self-healing agent.

    Args:
        config_file: Path to configuration file
        debug: Enable debug logging

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Set logging level
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

        # Get project root
        project_root = _get_project_root()
        logger.info(f"Project root: {project_root}")

        # Add project root to Python path for imports
        sys.path.insert(0, str(project_root))

        # Import the main evolve module
        # This needs to be done after adding to sys.path
        from evolve import main as evolve_main

        logger.info("Starting self-healing agent...")

        # Run the agent (evolve.py handles its own argument parsing)
        evolve_main()

        logger.info("Self-healing agent completed successfully")
        return 0

    except Exception as e:
        logger.exception(f"Error running self-healing agent: {e}")
        return 1


def _run_daemon(
    config_file: Optional[str] = None,
    debug: bool = False,
    interval: int = 3600,
) -> int:
    """Run the self-healing agent as a background daemon.

    Args:
        config_file: Path to configuration file
        debug: Enable debug logging
        interval: Check interval in seconds (default: 3600 = 1 hour)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    import time

    try:
        logger.info(f"Starting self-healing agent daemon (interval: {interval}s)")

        while True:
            logger.info("Running agent check...")
            exit_code = _run_agent(
                config_file=config_file,
                debug=debug
            )

            if exit_code != 0:
                logger.warning(f"Agent run failed with exit code {exit_code}")

            logger.info(f"Waiting {interval} seconds until next check...")
            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Daemon stopped by user")
        return 0
    except Exception as e:
        logger.exception(f"Daemon error: {e}")
        return 1


def _show_version() -> None:
    """Display version information."""
    from self_healing_agent import __version__
    print(f"Self-Healing Agent v{__version__}")


def _show_config() -> None:
    """Display current configuration."""
    project_root = _get_project_root()
    config_file = project_root / "configs" / "config.yaml"

    if config_file.exists():
        print(f"Configuration file: {config_file}")
        with open(config_file, 'r') as f:
            print(f.read())
    else:
        print(f"No configuration file found at: {config_file}")


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Self-Healing Agent - A multi-agent system for healing and evolving agentic applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run agent once (interactive mode)
    self-healing-agent run

    # Run agent with custom config
    self-healing-agent run --config /path/to/config.yaml

    # Run as background daemon (checks every hour)
    self-healing-agent daemon

    # Run daemon with custom interval (every 30 minutes)
    self-healing-agent daemon --interval 1800

    # Show version
    self-healing-agent version

    # Show current configuration
    self-healing-agent config
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run the agent once')
    run_parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    run_parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    # Daemon command
    daemon_parser = subparsers.add_parser('daemon', help='Run the agent as a background daemon')
    daemon_parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    daemon_parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    daemon_parser.add_argument(
        '--interval',
        type=int,
        default=3600,
        help='Check interval in seconds (default: 3600 = 1 hour)'
    )

    # Version command
    subparsers.add_parser('version', help='Show version information')

    # Config command
    subparsers.add_parser('config', help='Show current configuration')

    args = parser.parse_args()

    # Handle commands
    if args.command == 'run':
        return _run_agent(
            config_file=args.config,
            debug=args.debug
        )
    elif args.command == 'daemon':
        return _run_daemon(
            config_file=args.config,
            debug=args.debug,
            interval=args.interval
        )
    elif args.command == 'version':
        _show_version()
        return 0
    elif args.command == 'config':
        _show_config()
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
