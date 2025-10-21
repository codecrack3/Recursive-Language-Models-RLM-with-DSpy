"""Configuration for RLM sandbox selection."""

import os
from typing import Literal, Optional
from dataclasses import dataclass


SandboxType = Literal['e2b', 'restricted', 'auto']


@dataclass
class RLMSettings:
    """Global settings for RLM."""

    sandbox: SandboxType = 'auto'
    e2b_api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> 'RLMSettings':
        """
        Create settings from environment variables.

        Environment variables:
            RLM_SANDBOX: 'e2b', 'restricted', or 'auto' (default: 'auto')
            E2B_API_KEY: API key for E2B cloud sandboxes

        Returns:
            RLMSettings instance
        """
        sandbox = os.getenv('RLM_SANDBOX', 'auto')
        e2b_api_key = os.getenv('E2B_API_KEY')

        # Validate sandbox
        if sandbox not in ('e2b', 'restricted', 'auto'):
            raise ValueError(
                f"Invalid RLM_SANDBOX: {sandbox}. Must be 'e2b', 'restricted', or 'auto'"
            )

        return cls(
            sandbox=sandbox,  # type: ignore
            e2b_api_key=e2b_api_key
        )

    def resolve_sandbox(self) -> Literal['e2b', 'restricted']:
        """
        Resolve 'auto' sandbox to concrete choice.

        Auto-resolution logic:
            - Try E2B cloud if API key is set
            - Try E2B local if e2b package is available
            - Fall back to RestrictedPython

        Returns:
            'e2b' or 'restricted'
        """
        if self.sandbox == 'e2b':
            return 'e2b'
        elif self.sandbox == 'restricted':
            return 'restricted'
        else:  # auto
            try:
                import e2b
                # If we have E2B, prefer it
                return 'e2b'
            except ImportError:
                return 'restricted'


# Global settings instance
_settings: Optional[RLMSettings] = None


def get_settings() -> RLMSettings:
    """
    Get global RLM settings.

    Lazy-loads settings from environment on first call.

    Returns:
        RLMSettings instance
    """
    global _settings
    if _settings is None:
        _settings = RLMSettings.from_env()
    return _settings


def set_settings(settings: RLMSettings) -> None:
    """
    Override global RLM settings.

    Args:
        settings: New settings to use
    """
    global _settings
    _settings = settings


def reset_settings() -> None:
    """Reset settings to reload from environment."""
    global _settings
    _settings = None
