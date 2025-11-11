"""
File operation tools for the self-healing agent.

These tools enable agents to write analysis results and reports to files.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool

from constants import (
    VALID_ABSOLUTE_PATH_PREFIXES,
    USE_SUDO_FOR_FILE_OPS,
    DEFAULT_OUTPUT_DIR,
)

logger = logging.getLogger(__name__)


@tool
def write_file(
    file_path: str,
    content: str,
    mode: str = "w",
    create_dirs: bool = True,
) -> str:
    """
    Write content to a file.

    This tool writes text content to a specified file path. It can create
    parent directories if needed and supports different write modes.

    Args:
        file_path: Path to the file to write (relative or absolute)
        content: Content to write to the file
        mode: Write mode - 'w' for write (overwrite), 'a' for append (default: 'w')
        create_dirs: Whether to create parent directories if they don't exist (default: True)

    Returns:
        Success message with file path and content length

    Raises:
        ValueError: If file_path is empty or mode is invalid
        OSError: If file operations fail

    Example:
        >>> result = write_file(
        ...     file_path="reports/error_analysis.md",
        ...     content="# Error Analysis\\n\\n...",
        ...     mode="w"
        ... )
        >>> print(result)
        Successfully wrote 1234 characters to reports/error_analysis.md
    """
    try:
        logger.info(f"Writing content to file: {file_path}")

        # Validate inputs
        if not file_path or not file_path.strip():
            raise ValueError("file_path cannot be empty")

        if mode not in ["w", "a"]:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'w' or 'a'")

        # Convert to Path object for easier manipulation
        path = Path(file_path)

        # Handle paths that try to write to root filesystem
        # If path starts with '/' but isn't a valid absolute path,
        # convert it to a relative path in the default output directory
        if file_path.startswith('/') and not file_path.startswith(VALID_ABSOLUTE_PATH_PREFIXES):
            # Remove leading slash to make it relative
            relative_filename = file_path.lstrip('/')
            # Place in default output directory
            path = Path(DEFAULT_OUTPUT_DIR) / relative_filename
            logger.warning(
                f"Converted invalid absolute path '{file_path}' to "
                f"relative path '{path}' (USE_SUDO_FOR_FILE_OPS={USE_SUDO_FOR_FILE_OPS})"
            )
        # Create parent directories if needed
        if create_dirs and not path.parent.exists():
            logger.info(f"Creating parent directories for: {path.parent}")
            path.parent.mkdir(parents=True, exist_ok=True)

        # Write the content
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)

        file_size = len(content)
        success_msg = f"Successfully wrote {file_size} characters to {file_path}"
        logger.info(success_msg)

        # Also log file absolute path for clarity
        abs_path = path.absolute()
        logger.info(f"Absolute path: {abs_path}")

        return success_msg

    except Exception as e:
        error_msg = f"Error writing to file {file_path}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise OSError(error_msg) from e


@tool
def read_file(
    file_path: str,
) -> str:
    """
    Read content from a file.

    This tool reads text content from a specified file path.

    Args:
        file_path: Path to the file to read (relative or absolute)

    Returns:
        File content as string

    Raises:
        ValueError: If file_path is empty
        FileNotFoundError: If file doesn't exist
        OSError: If file operations fail

    Example:
        >>> content = read_file("reports/error_analysis.md")
        >>> print(content[:100])
        # Error Analysis...
    """
    try:
        logger.info(f"Reading content from file: {file_path}")

        # Validate input
        if not file_path or not file_path.strip():
            raise ValueError("file_path cannot be empty")

        # Convert to Path object
        path = Path(file_path)

        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        # Read the content
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        logger.info(f"Successfully read {len(content)} characters from {file_path}")

        return content

    except Exception as e:
        error_msg = f"Error reading file {file_path}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise
