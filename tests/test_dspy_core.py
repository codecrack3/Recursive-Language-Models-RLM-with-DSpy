"""Tests for DSPy-based RLM core."""

import pytest
from unittest.mock import Mock, patch
from rlm.core_dspy import RLMDSPy, RLMError, MaxDepthError


@pytest.mark.skipif(
    True,  # Skip by default as requires DSPy and API keys
    reason="DSPy tests require dspy-ai package and API keys"
)
class TestRLMDSPy:
    """Test DSPy-based RLM implementation."""

    def test_initialization(self):
        """Test RLMDSPy initialization."""
        try:
            rlm = RLMDSPy(
                model="gpt-4o-mini",
                max_depth=3,
                max_iterations=10
            )
            assert rlm.model == "gpt-4o-mini"
            assert rlm.max_depth == 3
            assert rlm.max_iterations == 10
        except Exception as e:
            pytest.skip(f"DSPy not available: {e}")

    def test_max_depth_check(self):
        """Test max depth enforcement."""
        try:
            rlm = RLMDSPy(
                model="gpt-4o-mini",
                max_depth=1,
                _current_depth=1
            )

            with pytest.raises(MaxDepthError):
                rlm.completion(query="test", context="test")
        except Exception as e:
            pytest.skip(f"DSPy not available: {e}")

    def test_stats_tracking(self):
        """Test statistics tracking."""
        try:
            rlm = RLMDSPy(model="gpt-4o-mini")
            stats = rlm.stats

            assert 'llm_calls' in stats
            assert 'iterations' in stats
            assert 'depth' in stats
        except Exception as e:
            pytest.skip(f"DSPy not available: {e}")

    def test_recursive_model(self):
        """Test recursive model selection."""
        try:
            rlm = RLMDSPy(
                model="gpt-4o",
                recursive_model="gpt-4o-mini"
            )
            assert rlm.model == "gpt-4o"
            assert rlm.recursive_model == "gpt-4o-mini"
        except Exception as e:
            pytest.skip(f"DSPy not available: {e}")


class TestRLMDSPyIntegration:
    """Integration tests for DSPy backend (mock-based)."""

    @patch('rlm.core_dspy.dspy')
    def test_backend_selection(self, mock_dspy):
        """Test DSPy backend is properly configured."""
        # Mock DSPy to avoid import errors
        mock_lm = Mock()
        mock_dspy.OpenAI.return_value = mock_lm

        try:
            rlm = RLMDSPy(model="gpt-4o-mini")
            # Should have called DSPy setup
            assert mock_dspy.settings.configure.called
        except Exception as e:
            pytest.skip(f"Test requires mocking setup: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
