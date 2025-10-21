"""Tests for core RLM using DSPy backend."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from rlm import RLM, MaxIterationsError, MaxDepthError


@pytest.fixture
def mock_dspy():
    """Mock DSPy module and LM."""
    with patch('rlm.core_dspy.dspy') as mock:
        # Setup mock LM
        mock_lm = Mock()
        mock.LM.return_value = mock_lm
        mock.settings = Mock()
        yield mock


@pytest.fixture
def mock_sandbox():
    """Mock sandbox executor."""
    with patch('rlm.core_dspy.SandboxFactory') as mock_factory:
        mock_sandbox = Mock()
        mock_sandbox.execute.return_value = "Execution result"
        mock_factory.create.return_value = mock_sandbox
        yield mock_sandbox


@pytest.fixture
def mock_rlm_module():
    """Mock RLMModule."""
    with patch('rlm.core_dspy.RLMModule') as mock_module_class:
        mock_module = Mock()
        mock_module.stats = {'llm_calls': 1, 'iterations': 1}
        
        # Create a prediction object
        mock_prediction = Mock()
        mock_prediction.answer = "The answer"
        
        mock_module.return_value = mock_prediction
        mock_module_class.return_value = mock_module
        yield mock_module


@pytest.mark.asyncio
async def test_simple_completion(mock_dspy, mock_sandbox, mock_rlm_module):
    """Test simple completion with DSPy backend."""
    mock_prediction = Mock()
    mock_prediction.answer = "The answer"
    mock_rlm_module.return_value = mock_prediction

    rlm = RLM(model="test-model")
    result = await rlm.acompletion("What is the answer?", "Some context")

    assert result == "The answer"
    assert mock_rlm_module.called


@pytest.mark.asyncio
async def test_multi_step_completion(mock_dspy, mock_sandbox, mock_rlm_module):
    """Test multi-step completion."""
    mock_prediction = Mock()
    mock_prediction.answer = "Done"
    mock_rlm_module.return_value = mock_prediction
    mock_rlm_module.stats = {'llm_calls': 2, 'iterations': 2}

    rlm = RLM(model="test-model")
    result = await rlm.acompletion("Test", "Hello World Test")

    assert result == "Done"


@pytest.mark.asyncio
async def test_max_depth_error(mock_dspy, mock_sandbox):
    """Test max depth exceeded."""
    rlm = RLM(model="test-model", max_depth=2, _current_depth=2)

    with pytest.raises(MaxDepthError):
        await rlm.acompletion("Test", "Context")


@pytest.mark.asyncio
async def test_context_operations(mock_dspy, mock_sandbox, mock_rlm_module):
    """Test context operations in REPL."""
    mock_prediction = Mock()
    mock_prediction.answer = "Hello Worl"
    mock_rlm_module.return_value = mock_prediction

    rlm = RLM(model="test-model")
    result = await rlm.acompletion("Get first 10 chars", "Hello World Example")

    assert result == "Hello Worl"


def test_sync_completion(mock_dspy, mock_sandbox, mock_rlm_module):
    """Test sync wrapper."""
    mock_prediction = Mock()
    mock_prediction.answer = "Sync result"
    mock_rlm_module.return_value = mock_prediction

    rlm = RLM(model="test-model")
    result = rlm.completion("Test", "Context")

    assert result == "Sync result"


@pytest.mark.asyncio
async def test_two_models(mock_dspy, mock_sandbox, mock_rlm_module):
    """Test using different models for root and recursive."""
    mock_prediction = Mock()
    mock_prediction.answer = "Answer"
    mock_rlm_module.return_value = mock_prediction

    rlm = RLM(
        model="expensive-model",
        recursive_model="cheap-model",
        _current_depth=0
    )

    await rlm.acompletion("Test", "Context")

    # Verify initialization
    assert rlm.model == "expensive-model"
    assert rlm.recursive_model == "cheap-model"


@pytest.mark.asyncio
async def test_stats(mock_dspy, mock_sandbox, mock_rlm_module):
    """Test statistics tracking."""
    mock_prediction = Mock()
    mock_prediction.answer = "Done"
    mock_rlm_module.return_value = mock_prediction
    mock_rlm_module.stats = {'llm_calls': 3, 'iterations': 3}

    rlm = RLM(model="test-model")
    await rlm.acompletion("Test", "Context")

    stats = rlm.stats
    assert stats['llm_calls'] >= 3
    assert stats['iterations'] >= 3
    assert stats['depth'] == 0


@pytest.mark.asyncio
async def test_api_base_and_key(mock_dspy, mock_sandbox, mock_rlm_module):
    """Test API base and key passing."""
    mock_prediction = Mock()
    mock_prediction.answer = "Answer"
    mock_rlm_module.return_value = mock_prediction

    rlm = RLM(
        model="test-model",
        api_base="http://localhost:8000",
        api_key="test-key"
    )

    await rlm.acompletion("Test", "Context")

    # Verify attributes were set
    assert rlm.api_base == "http://localhost:8000"
    assert rlm.api_key == "test-key"


def test_initialization(mock_dspy, mock_sandbox):
    """Test RLM initialization."""
    rlm = RLM(
        model="gpt-4o-mini",
        max_depth=3,
        max_iterations=10
    )
    
    assert rlm.model == "gpt-4o-mini"
    assert rlm.max_depth == 3
    assert rlm.max_iterations == 10


def test_sandbox_selection(mock_dspy):
    """Test sandbox type selection."""
    with patch('rlm.core_dspy.SandboxFactory') as mock_factory:
        mock_sandbox = Mock()
        mock_factory.create.return_value = mock_sandbox
        
        # Test explicit E2B
        rlm = RLM(model="test-model", sandbox='e2b')
        mock_factory.create.assert_called_with(sandbox_type='e2b')
        
        # Test explicit restricted
        rlm = RLM(model="test-model", sandbox='restricted')
        mock_factory.create.assert_called_with(sandbox_type='restricted')


@pytest.mark.asyncio
async def test_recursive_llm_function(mock_dspy, mock_sandbox, mock_rlm_module):
    """Test recursive LLM function creation."""
    mock_prediction = Mock()
    mock_prediction.answer = "Recursive result"
    mock_rlm_module.return_value = mock_prediction

    rlm = RLM(model="test-model", max_depth=3)
    
    # Create recursive function
    recursive_fn = rlm._make_recursive_fn()
    
    # Call it
    result = recursive_fn("sub query", "sub context")
    
    assert result is not None
