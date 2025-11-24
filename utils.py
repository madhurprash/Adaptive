import yaml
import json
import logging
import boto3
from boto3.session import Session
from datetime import datetime
from pathlib import Path
from typing import Union, Dict, Optional, Any, List, Tuple

# set a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_config(config_file: Union[Path, str]) -> Optional[Dict]:
    """
    Load configuration from a local file.

    :param config_file: Path to the local file
    :return: Dictionary with the loaded configuration
    """
    try:
        config_data: Optional[Dict] = None
        logger.info(f"Loading config from local file system: {config_file}")
        content = Path(config_file).read_text()
        config_data = yaml.safe_load(content)
        logger.info(f"Loaded config from local file system: {config_data}")
    except Exception as e:
        logger.error(f"Error loading config from local file system: {e}")
        config_data = None
    return config_data

# Load the admin agent system prompt from the config file
def load_system_prompt(
    prompt_path: str
) -> str:
    """
    Load the system prompt from a file path.

    Args:
        prompt_path: Relative or absolute path to the system prompt file

    Returns:
        The system prompt as a string
    """
    try:
        # First try absolute path or relative to current directory
        prompt_file = Path(prompt_path)
        if prompt_file.exists():
            prompt_content = prompt_file.read_text()
            logger.info(f"Successfully loaded system prompt from {prompt_path}")
            return prompt_content

        # If not found, try relative to project root
        # (assuming utils.py is in the root of the project)
        project_root = Path(__file__).parent
        project_prompt_path = project_root / prompt_path
        if project_prompt_path.exists():
            prompt_content = project_prompt_path.read_text()
            logger.info(f"Successfully loaded system prompt from project root: {project_prompt_path}")
            return prompt_content

        # If still not found, raise error
        raise FileNotFoundError(f"System prompt file not found at {prompt_path}")
    except FileNotFoundError:
        logger.error(f"System prompt file not found at {prompt_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading system prompt from {prompt_path}: {str(e)}")
        raise