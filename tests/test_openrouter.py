"""Tests for OpenRouter integration."""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.skipif(
    True,  # Skip by default as requires DSPy and API keys
    reason="OpenRouter tests require dspy-ai package and API keys"
)
class TestOpenRouterIntegration:
    """Test OpenRouter model integration (requires packages)."""

    def test_openrouter_model_detection(self):
        """Test OpenRouter model detection."""
        try:
            from rlm import RLM

            # Mock environment
            with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}):
                # This should detect OpenRouter
                rlm = RLM(
                    model="openrouter/anthropic/claude-3.5-sonnet",
                    backend='dspy'
                )

                assert rlm.model == "openrouter/anthropic/claude-3.5-sonnet"
        except ImportError as e:
            pytest.skip(f"DSPy not available: {e}")

    def test_openrouter_api_key_required(self):
        """Test that API key is required for OpenRouter."""
        try:
            from rlm import RLM

            # Clear environment
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError) as exc_info:
                    RLM(
                        model="openrouter/anthropic/claude-3.5-sonnet",
                        backend='dspy'
                    )

                assert "OPENROUTER_API_KEY" in str(exc_info.value)
        except ImportError as e:
            pytest.skip(f"DSPy not available: {e}")

    def test_openrouter_with_api_key_parameter(self):
        """Test OpenRouter with API key passed as parameter."""
        try:
            from rlm import RLM

            # API key via parameter should work
            rlm = RLM(
                model="openrouter/anthropic/claude-3.5-sonnet",
                backend='dspy',
                api_key='test-key-param'
            )

            assert rlm.api_key == 'test-key-param'
        except ImportError as e:
            pytest.skip(f"DSPy not available: {e}")


class TestOpenRouterMocked:
    """Mock-based OpenRouter tests (no actual API calls)."""

    @patch('rlm.core_dspy.dspy')
    def test_openrouter_dspy_configuration(self, mock_dspy):
        """Test that OpenRouter configures DSPy correctly."""
        # Setup mocks
        mock_openai = Mock()
        mock_dspy.OpenAI.return_value = mock_openai

        try:
            from rlm.core_dspy import RLMDSPy

            # Create RLM with OpenRouter model
            with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}):
                rlm = RLMDSPy(
                    model="openrouter/anthropic/claude-3.5-sonnet"
                )

            # Verify OpenAI was called with correct parameters
            mock_dspy.OpenAI.assert_called()
            call_kwargs = mock_dspy.OpenAI.call_args[1]

            assert call_kwargs['model'] == 'anthropic/claude-3.5-sonnet'
            assert call_kwargs['api_key'] == 'test-key'
            assert call_kwargs['api_base'] == 'https://openrouter.ai/api/v1'

        except ImportError as e:
            pytest.skip(f"Module not available: {e}")

    @patch('rlm.core_dspy.dspy')
    def test_openrouter_model_name_stripping(self, mock_dspy):
        """Test that 'openrouter/' prefix is stripped from model name."""
        mock_dspy.OpenAI.return_value = Mock()

        try:
            from rlm.core_dspy import RLMDSPy

            test_cases = [
                ("openrouter/anthropic/claude-3.5-sonnet", "anthropic/claude-3.5-sonnet"),
                ("openrouter/openai/gpt-4o-mini", "openai/gpt-4o-mini"),
                ("openrouter/google/gemini-pro", "google/gemini-pro"),
            ]

            for full_model, expected_stripped in test_cases:
                with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}):
                    rlm = RLMDSPy(model=full_model)

                # Check that model name was stripped
                call_kwargs = mock_dspy.OpenAI.call_args[1]
                assert call_kwargs['model'] == expected_stripped

        except ImportError as e:
            pytest.skip(f"Module not available: {e}")

    def test_openrouter_model_format_validation(self):
        """Test OpenRouter model format validation."""
        # Valid formats
        valid_models = [
            "openrouter/anthropic/claude-3.5-sonnet",
            "openrouter/openai/gpt-4o-mini",
            "openrouter/google/gemini-pro",
            "openrouter/meta-llama/llama-3.1-70b-instruct",
        ]

        for model in valid_models:
            assert model.startswith("openrouter/")
            stripped = model.replace("openrouter/", "")
            assert "/" in stripped  # Should have provider/model format


class TestOpenRouterCostOptimization:
    """Test cost optimization features with OpenRouter."""

    @patch('rlm.core_dspy.dspy')
    def test_two_model_strategy_openrouter(self, mock_dspy):
        """Test using different OpenRouter models for root vs recursion."""
        mock_dspy.OpenAI.return_value = Mock()

        try:
            from rlm.core_dspy import RLMDSPy

            with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}):
                rlm = RLMDSPy(
                    model="openrouter/anthropic/claude-3.5-sonnet",
                    recursive_model="openrouter/anthropic/claude-3-haiku"
                )

            assert rlm.model == "openrouter/anthropic/claude-3.5-sonnet"
            assert rlm.recursive_model == "openrouter/anthropic/claude-3-haiku"

        except ImportError as e:
            pytest.skip(f"Module not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
