"""Recursive Language Models for unbounded context processing."""

from typing import Optional, Any

from .repl import REPLError
from .config import get_settings, SandboxType
from .core_dspy import RLMDSPy, RLMError, MaxIterationsError, MaxDepthError
from .graph import RLMGraphTracker, GraphNode, REPLStep

__version__ = "0.2.0"


def create_rlm(
    model: str,
    sandbox: SandboxType = 'auto',
    **kwargs: Any
):
    """
    Create an RLM instance using DSPy backend.

    Args:
        model: Model name (e.g., "gpt-4o-mini", "claude-sonnet-4", "ollama/llama3.2")
        sandbox: Sandbox to use ('e2b', 'restricted', or 'auto')
        **kwargs: Additional parameters passed to RLM constructor

    Returns:
        RLMDSPy instance

    Examples:
        # Basic usage
        rlm = create_rlm("gpt-4o-mini")

        # With E2B sandbox
        rlm = create_rlm("gpt-4o-mini", sandbox='e2b')

        # With RestrictedPython sandbox
        rlm = create_rlm("gpt-4o-mini", sandbox='restricted')
    """
    return RLMDSPy(model=model, sandbox_type=sandbox, **kwargs)


class RLM:
    """
    Recursive Language Model using DSPy backend.

    This class provides the main interface for RLM, wrapping the DSPy implementation.

    Usage:
        # Basic usage
        rlm = RLM(model="gpt-4o-mini")
        result = rlm.completion(query="...", context="...")

        # With E2B sandbox
        rlm = RLM(model="gpt-4o-mini", sandbox='e2b')

        # With RestrictedPython sandbox
        rlm = RLM(model="gpt-4o-mini", sandbox='restricted')
    """

    def __new__(
        cls,
        model: str,
        sandbox: SandboxType = 'auto',
        **kwargs: Any
    ):
        """
        Create RLM instance using DSPy backend.

        Args:
            model: Model name
            sandbox: Sandbox to use ('e2b', 'restricted', or 'auto')
            **kwargs: Additional parameters

        Returns:
            RLMDSPy instance
        """
        return RLMDSPy(model=model, sandbox_type=sandbox, **kwargs)


# Export all public symbols
__all__ = [
    # Main classes
    "RLM",
    "create_rlm",

    # Backend implementation (for advanced usage)
    "RLMDSPy",

    # Graph tracking
    "RLMGraphTracker",
    "GraphNode",
    "REPLStep",

    # Errors
    "RLMError",
    "MaxIterationsError",
    "MaxDepthError",
    "REPLError",

    # Types
    "SandboxType",

    # Version
    "__version__",
]
