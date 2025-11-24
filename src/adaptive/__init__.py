"""
Adaptive - Continuous optimization for AI agents

A multi-agent system for continuous optimization and evolution of AI agents through intelligent observability.
"""
import sys
import importlib.util
from pathlib import Path

__version__ = "0.1.0"
__author__ = "Adaptive Team"
__license__ = "MIT"


def _load_main_module():
    """Load the main adaptive.py module from project root."""
    # First check if adaptive.py is in sys.path (when running from project directory)
    # This happens when cli.py adds the project root to sys.path
    for path in sys.path:
        adaptive_py_path = Path(path) / "adaptive.py"
        if adaptive_py_path.exists():
            # Load the module dynamically to avoid circular import
            spec = importlib.util.spec_from_file_location("_adaptive_main", adaptive_py_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    # Fallback: try to find it relative to this file (for development mode)
    project_root = Path(__file__).parent.parent.parent
    adaptive_py_path = project_root / "adaptive.py"

    if adaptive_py_path.exists():
        spec = importlib.util.spec_from_file_location("_adaptive_main", adaptive_py_path)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    raise ImportError(
        "Could not find adaptive.py. Make sure you're running from the project directory "
        "or that the project root is in your Python path."
    )


# Load and expose the main function
_adaptive_module = _load_main_module()
main = _adaptive_module.main

__all__ = ["__version__", "__author__", "__license__", "main"]
