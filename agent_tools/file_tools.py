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

@tool
def search_files(
    query: str,
    search_path: str = ".",
    file_pattern: Optional[str] = None,
    search_contents: bool = True,
    max_results: int = 50,
) -> str:
    """
    Search for files whose names or contents match a query.

    This tool searches within a directory (optionally recursively) for files
    whose names or contents contain the given query string.

    Args:
        query: Text to search for in file names and/or contents
        search_path: Root directory to search (default: current directory)
        file_pattern: Optional glob pattern to filter files
                      (e.g., "*.md", "**/*.py"). If None, all files are scanned.
        search_contents: Whether to search inside file contents (default: True)
        max_results: Maximum number of matches to return (default: 50)

    Returns:
        A formatted string listing matching files and where they matched.

    Raises:
        ValueError: If query or search_path is empty
        FileNotFoundError: If search_path doesn't exist
        OSError: If directory or file operations fail

    Example:
        >>> result = search_files(
        ...     query="timeout error",
        ...     search_path="reports",
        ...     file_pattern="*.md",
        ... )
        >>> print(result)
        Search query: timeout error
        Search path: reports
        File pattern: *.md
        Search contents: True
        Matches found: 2 (showing up to 50)

        - error_analysis.md (1234 bytes) [matched in contents]
        - timeout_summary.md (5678 bytes) [matched in filename, contents]
    """
    try:
        logger.info(
            f"Searching for query '{query}' in path '{search_path}' "
            f"(pattern={file_pattern}, search_contents={search_contents})"
        )

        # Validate inputs
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if not search_path or not search_path.strip():
            raise ValueError("search_path cannot be empty")

        # Normalize search path
        path = Path(search_path)

        # If absolute path is outside allowed prefixes, redirect to DEFAULT_OUTPUT_DIR
        if search_path.startswith("/") and not search_path.startswith(VALID_ABSOLUTE_PATH_PREFIXES):
            logger.warning(
                f"Search path '{search_path}' is not in VALID_ABSOLUTE_PATH_PREFIXES; "
                f"using DEFAULT_OUTPUT_DIR='{DEFAULT_OUTPUT_DIR}' instead"
            )
            path = Path(DEFAULT_OUTPUT_DIR)

        # Check path exists and is directory
        if not path.exists():
            raise FileNotFoundError(f"Search path not found: {path}")

        if not path.is_dir():
            raise ValueError(f"Search path is not a directory: {path}")

        # Build list of candidate files
        if file_pattern:
            # Use rglob to be recursive by default for search
            files = list(path.rglob(file_pattern))
        else:
            files = [p for p in path.rglob("*") if p.is_file()]

        query_lower = query.lower()
        matches = []

        for file in files:
            if len(matches) >= max_results:
                break

            match_reasons = []

            # Check filename
            if query_lower in file.name.lower():
                match_reasons.append("filename")

            # Optionally check contents
            if search_contents:
                try:
                    # Read with ignore errors to handle mixed/binary files safely
                    with open(file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    if query_lower in content.lower():
                        match_reasons.append("contents")
                except Exception as e:
                    # Log and skip problematic files (e.g., huge/binary)
                    logger.warning(f"Skipping file during content search {file}: {e}")

            if match_reasons:
                try:
                    relative_path = file.relative_to(path)
                except ValueError:
                    # Fallback in weird path cases
                    relative_path = file

                size = file.stat().st_size
                matches.append((relative_path, size, ", ".join(match_reasons)))

        if not matches:
            msg = (
                f"No matches found for query '{query}' "
                f"in '{path}' (pattern={file_pattern or 'all files'})"
            )
            logger.info(msg)
            return msg

        # Format output
        lines = [
            f"Search query: {query}",
            f"Search path: {path}",
            f"File pattern: {file_pattern or 'all files'}",
            f"Search contents: {search_contents}",
            f"Matches found: {len(matches)} (showing up to {max_results})",
            "",
        ]

        for rel_path, size, reasons in matches:
            lines.append(f"- {rel_path} ({size} bytes) [matched in {reasons}]")

        result = "\n".join(lines)
        logger.info(
            f"Search for query '{query}' in '{path}' found {len(matches)} matches "
            f"(max_results={max_results})"
        )
        return result

    except Exception as e:
        error_msg = (
            f"Error searching for query '{query}' in '{search_path}': {str(e)}"
        )
        logger.error(error_msg, exc_info=True)
        raise

@tool
def list_directory(
    directory_path: str = ".",
    pattern: Optional[str] = None,
    recursive: bool = False,
) -> str:
    """
    List files and directories in a specified path.

    This tool lists contents of a directory, optionally filtering by pattern
    and recursively searching subdirectories.

    Args:
        directory_path: Path to the directory to list (default: current directory)
        pattern: Optional glob pattern to filter files (e.g., "*.md", "**/*.py")
        recursive: Whether to list files recursively (default: False)

    Returns:
        Formatted string listing the directory contents

    Raises:
        ValueError: If directory_path is empty
        FileNotFoundError: If directory doesn't exist
        OSError: If directory operations fail

    Example:
        >>> result = list_directory("reports", pattern="*.md")
        >>> print(result)
        Directory: reports
        Files found: 3
        
        - error_analysis.md (1234 bytes)
        - summary_report.md (5678 bytes)
        - recommendations.md (910 bytes)
    """
    try:
        logger.info(f"Listing directory: {directory_path}")

        # Validate input
        if not directory_path or not directory_path.strip():
            raise ValueError("directory_path cannot be empty")

        # Convert to Path object
        path = Path(directory_path)

        # Check if directory exists
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        # Get files based on pattern and recursive flag
        if pattern:
            if recursive:
                files = list(path.rglob(pattern))
            else:
                files = list(path.glob(pattern))
        else:
            if recursive:
                files = list(path.rglob("*"))
            else:
                files = list(path.glob("*"))

        # Sort files for consistent output
        files.sort()

        # Format output
        output_lines = [
            f"Directory: {directory_path}",
            f"Files found: {len(files)}",
            ""
        ]

        for file in files:
            if file.is_file():
                size = file.stat().st_size
                relative_path = file.relative_to(path) if recursive else file.name
                output_lines.append(f"- {relative_path} ({size} bytes)")
            elif file.is_dir():
                relative_path = file.relative_to(path) if recursive else file.name
                output_lines.append(f"- {relative_path}/ (directory)")

        result = "\n".join(output_lines)
        logger.info(f"Successfully listed {len(files)} items in {directory_path}")

        return result

    except Exception as e:
        error_msg = f"Error listing directory {directory_path}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise