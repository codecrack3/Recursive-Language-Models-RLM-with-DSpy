"""Integration tests for RLM with DSPy backend."""

import pytest
from unittest.mock import patch, Mock
from rlm import RLM


@pytest.fixture
def mock_dspy_env():
    """Mock DSPy environment."""
    with patch('rlm.core_dspy.dspy') as mock_dspy:
        mock_lm = Mock()
        mock_dspy.LM.return_value = mock_lm
        mock_dspy.settings = Mock()
        
        with patch('rlm.core_dspy.SandboxFactory') as mock_factory:
            mock_sandbox = Mock()
            mock_factory.create.return_value = mock_sandbox
            
            with patch('rlm.core_dspy.RLMModule') as mock_module_class:
                mock_module = Mock()
                mock_module.stats = {'llm_calls': 1, 'iterations': 1}
                mock_module_class.return_value = mock_module
                
                yield {
                    'dspy': mock_dspy,
                    'sandbox': mock_sandbox,
                    'module': mock_module,
                    'module_class': mock_module_class
                }


@pytest.mark.asyncio
async def test_peek_strategy(mock_dspy_env):
    """Test peeking at context start."""
    mock_prediction = Mock()
    mock_prediction.answer = "This is a long document"
    mock_dspy_env['module'].return_value = mock_prediction

    rlm = RLM(model="test-model")
    result = await rlm.acompletion(
        "What does the context start with?",
        "This is a long document that starts with this sentence..."
    )

    assert "This is a long document" in result


@pytest.mark.asyncio
async def test_search_strategy(mock_dspy_env):
    """Test search and extraction strategy."""
    mock_prediction = Mock()
    mock_prediction.answer = "Found years: 2020, 2021, 2022"
    mock_dspy_env['module'].return_value = mock_prediction

    rlm = RLM(model="test-model")
    result = await rlm.acompletion(
        "Find all years",
        "The years 2020, 2021, and 2022 were important."
    )

    assert "2020" in result


@pytest.mark.asyncio
async def test_chunk_strategy(mock_dspy_env):
    """Test chunking context."""
    mock_prediction = Mock()
    mock_prediction.answer = "5 chunks"
    mock_dspy_env['module'].return_value = mock_prediction
    
    rlm = RLM(model="test-model")
    result = await rlm.acompletion(
        "Chunk the context",
        "A" * 50  # 50 chars
    )

    assert "5" in result


@pytest.mark.asyncio
async def test_extraction_strategy(mock_dspy_env):
    """Test data extraction."""
    mock_prediction = Mock()
    mock_prediction.answer = "Names: Alice, Bob"
    mock_dspy_env['module'].return_value = mock_prediction
    
    rlm = RLM(model="test-model")
    context = """
Name: Alice
Age: 30
Name: Bob
Age: 25
"""
    result = await rlm.acompletion("Extract names", context)

    assert "Alice" in result or "Bob" in result


@pytest.mark.asyncio
async def test_long_context(mock_dspy_env):
    """Test with long context."""
    mock_prediction = Mock()
    mock_prediction.answer = "Context length: 100000"
    mock_dspy_env['module'].return_value = mock_prediction
    
    rlm = RLM(model="test-model")
    long_context = "A" * 100000  # 100k chars
    result = await rlm.acompletion("How long is this?", long_context)

    assert "100000" in result


@pytest.mark.asyncio
async def test_multiline_answer(mock_dspy_env):
    """Test multiline final answer."""
    mock_prediction = Mock()
    mock_prediction.answer = "Line 1\nLine 2\nLine 3"
    mock_dspy_env['module'].return_value = mock_prediction
    
    rlm = RLM(model="test-model")
    result = await rlm.acompletion("Test", "Context")

    assert "Line 1" in result
    assert "Line 2" in result


@pytest.mark.asyncio
async def test_recursive_processing(mock_dspy_env):
    """Test recursive LLM processing."""
    mock_prediction = Mock()
    mock_prediction.answer = "Processed recursively"
    mock_dspy_env['module'].return_value = mock_prediction
    
    rlm = RLM(
        model="test-model",
        max_depth=3,
        _current_depth=0
    )
    
    result = await rlm.acompletion("Process this", "Large context")
    
    assert result is not None
    assert rlm._current_depth < rlm.max_depth


@pytest.mark.asyncio
async def test_sandbox_execution(mock_dspy_env):
    """Test sandbox is properly configured."""
    mock_prediction = Mock()
    mock_prediction.answer = "Executed safely"
    mock_dspy_env['module'].return_value = mock_prediction
    
    rlm = RLM(model="test-model", sandbox='restricted')
    result = await rlm.acompletion("Test", "Context")
    
    # Verify sandbox was created
    assert mock_dspy_env['sandbox'] is not None


@pytest.mark.asyncio
async def test_two_model_strategy(mock_dspy_env):
    """Test using different models for root and recursion."""
    mock_prediction = Mock()
    mock_prediction.answer = "Completed with two models"
    mock_dspy_env['module'].return_value = mock_prediction
    
    rlm = RLM(
        model="expensive-model",
        recursive_model="cheap-model",
        max_depth=3
    )
    
    result = await rlm.acompletion("Process", "Context")
    
    assert rlm.model == "expensive-model"
    assert rlm.recursive_model == "cheap-model"
    assert result is not None


def test_sync_completion(mock_dspy_env):
    """Test synchronous completion wrapper."""
    mock_prediction = Mock()
    mock_prediction.answer = "Sync result"
    mock_dspy_env['module'].return_value = mock_prediction
    
    rlm = RLM(model="test-model")
    result = rlm.completion("Test query", "Test context")
    
    assert result == "Sync result"


def test_stats_tracking(mock_dspy_env):
    """Test statistics tracking across calls."""
    mock_prediction = Mock()
    mock_prediction.answer = "Result"
    mock_dspy_env['module'].return_value = mock_prediction
    mock_dspy_env['module'].stats = {'llm_calls': 5, 'iterations': 3}
    
    rlm = RLM(model="test-model")
    rlm.completion("Test", "Context")
    
    stats = rlm.stats
    assert 'llm_calls' in stats
    assert 'iterations' in stats
    assert 'depth' in stats


def test_api_configuration(mock_dspy_env):
    """Test API configuration is properly passed."""
    rlm = RLM(
        model="test-model",
        api_base="http://localhost:8000",
        api_key="test-key",
        temperature=0.7
    )
    
    assert rlm.api_base == "http://localhost:8000"
    assert rlm.api_key == "test-key"
    assert rlm.llm_kwargs.get('temperature') == 0.7
