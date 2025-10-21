"""Tests for sandbox executors."""

import pytest
from rlm.sandbox import (
    SandboxFactory,
    RestrictedPythonAdapter,
    SandboxExecutor,
    REPLError
)


class TestRestrictedPythonAdapter:
    """Test RestrictedPython sandbox adapter."""

    def test_basic_execution(self):
        """Test basic code execution."""
        sandbox = RestrictedPythonAdapter()
        env = {'x': 10, 'y': 20}

        result = sandbox.execute("print(x + y)", env)
        assert "30" in result

    def test_expression_evaluation(self):
        """Test expression evaluation."""
        sandbox = RestrictedPythonAdapter()
        env = {'numbers': [1, 2, 3, 4, 5]}

        result = sandbox.execute("sum(numbers)", env)
        assert "15" in result

    def test_error_handling(self):
        """Test error handling."""
        sandbox = RestrictedPythonAdapter()
        env = {}

        with pytest.raises(REPLError):
            sandbox.execute("undefined_variable", env)

    def test_code_extraction(self):
        """Test markdown code block extraction."""
        sandbox = RestrictedPythonAdapter()
        env = {'x': 5}

        # Test with python code block
        code_with_markdown = "```python\nprint(x * 2)\n```"
        result = sandbox.execute(code_with_markdown, env)
        assert "10" in result

        # Test with generic code block
        code_with_generic = "```\nprint(x * 3)\n```"
        result = sandbox.execute(code_with_generic, env)
        assert "15" in result


class TestSandboxFactory:
    """Test sandbox factory."""

    def test_create_restricted(self):
        """Test creating RestrictedPython sandbox."""
        sandbox = SandboxFactory.create(sandbox_type='restricted')
        assert isinstance(sandbox, RestrictedPythonAdapter)

    def test_create_auto_without_e2b(self):
        """Test auto creation without E2B."""
        # Should fall back to RestrictedPython
        sandbox = SandboxFactory.create(sandbox_type='auto')
        assert isinstance(sandbox, SandboxExecutor)

    def test_timeout_parameter(self):
        """Test timeout parameter."""
        sandbox = SandboxFactory.create(sandbox_type='restricted', timeout=10)
        assert sandbox.repl.timeout == 10

    def test_max_output_chars(self):
        """Test max_output_chars parameter."""
        sandbox = SandboxFactory.create(
            sandbox_type='restricted',
            max_output_chars=1000
        )
        assert sandbox.repl.max_output_chars == 1000


@pytest.mark.skipif(
    True,  # Skip by default as E2B requires API key
    reason="E2B tests require E2B_API_KEY environment variable"
)
class TestE2BSandbox:
    """Test E2B sandbox (requires API key)."""

    def test_basic_execution(self):
        """Test basic E2B execution."""
        try:
            sandbox = SandboxFactory.create(sandbox_type='e2b')
            env = {'x': 10, 'y': 20}

            result = sandbox.execute("print(x + y)", env)
            assert "30" in result
        except Exception as e:
            pytest.skip(f"E2B not available: {e}")

    def test_async_execution(self):
        """Test async E2B execution."""
        import asyncio

        try:
            sandbox = SandboxFactory.create(sandbox_type='e2b')
            env = {'x': 5}

            async def run_test():
                result = await sandbox.aexecute("print(x * 2)", env)
                return result

            result = asyncio.run(run_test())
            assert "10" in result
        except Exception as e:
            pytest.skip(f"E2B not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
