"""Sandbox executors for safe code execution (E2B and RestrictedPython)."""

import asyncio
import concurrent.futures
import os
from typing import Dict, Any, Optional, Literal
from abc import ABC, abstractmethod

from .repl import REPLExecutor, REPLError


def normalize_e2b_api_key(value: Optional[str]) -> Optional[str]:
    """Return a sanitized E2B API key without optional prefixes or whitespace."""
    if not isinstance(value, str):
        return value

    cleaned = value.strip()
    if not cleaned:
        return None

    lower_cleaned = cleaned.lower()
    for prefix in ('bearer ', 'apikey ', 'token '):
        if lower_cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            lower_cleaned = cleaned.lower()

    return cleaned or None


class SandboxExecutor(ABC):
    """Abstract base class for sandbox executors."""

    @abstractmethod
    def execute(self, code: str, env: Dict[str, Any]) -> str:
        """
        Execute code in sandbox.

        Args:
            code: Python code to execute
            env: Environment with context, query, recursive_llm, etc.

        Returns:
            String result of execution

        Raises:
            REPLError: If code execution fails
        """
        pass

    @abstractmethod
    async def aexecute(self, code: str, env: Dict[str, Any]) -> str:
        """
        Async execute code in sandbox.

        Args:
            code: Python code to execute
            env: Environment with context, query, recursive_llm, etc.

        Returns:
            String result of execution

        Raises:
            REPLError: If code execution fails
        """
        pass


class E2BSandboxExecutor(SandboxExecutor):
    """E2B cloud sandbox executor.
    
    This executor allows full Python execution without restrictions.
    All operators, imports, and standard Python operations are permitted.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_output_chars: int = 2000,
        api_key: Optional[str] = None
    ):
        """
        Initialize E2B sandbox executor.

        Args:
            timeout: Execution timeout in seconds
            max_output_chars: Maximum characters to return (truncate if longer)
            api_key: E2B API key (if None, will use E2B_API_KEY env var)
        """
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        raw_api_key = api_key or os.getenv('E2B_API_KEY')
        self.api_key = normalize_e2b_api_key(raw_api_key)

        # Validate E2B is available
        try:
            from e2b_code_interpreter import Sandbox
            self._Sandbox = Sandbox
        except ImportError as e:
            raise ImportError(
                "e2b-code-interpreter package is required for E2B sandbox. "
                "Install with: pip install e2b-code-interpreter"
            ) from e

        # Validate API key for cloud usage
        if not self.api_key:
            raise ValueError(
                "E2B API key is required. Set E2B_API_KEY environment variable "
                "or pass api_key parameter. Get your key at https://e2b.dev"
            )

    def execute(self, code: str, env: Dict[str, Any]) -> str:
        """
        Sync wrapper for aexecute.

        Args:
            code: Python code to execute
            env: Environment dict

        Returns:
            Execution result

        Raises:
            REPLError: If execution fails
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aexecute(code, env))

        # We are already inside an event loop; run the coroutine in a worker thread.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(self.aexecute(code, env)))
            return future.result()

    async def aexecute(self, code: str, env: Dict[str, Any]) -> str:
        """
        Execute code in E2B sandbox asynchronously.

        Args:
            code: Python code to execute
            env: Environment with context, query, recursive_llm, etc.

        Returns:
            String result of execution

        Raises:
            REPLError: If execution fails
        """
        # Extract code from markdown blocks if present
        code = self._extract_code(code)

        if not code.strip():
            return "No code to execute"

        # Create E2B sandbox
        sandbox_kwargs = {"timeout": self.timeout}
        if self.api_key:
            sandbox_kwargs["api_key"] = self.api_key

        try:
            sandbox = await asyncio.wait_for(
                asyncio.to_thread(self._Sandbox.create, **sandbox_kwargs),
                timeout=self.timeout
            )
        except asyncio.TimeoutError as e:
            raise REPLError(f"Failed to create E2B sandbox: timeout after {self.timeout}s") from e
        except Exception as e:
            raise REPLError(f"Failed to create E2B sandbox: {str(e)}") from e

        try:
            # Prepare environment setup code
            setup_code = self._build_env_setup(env)

            # Execute setup code first (set variables)
            if setup_code:
                await asyncio.wait_for(
                    asyncio.to_thread(sandbox.run_code, setup_code),
                    timeout=self.timeout
                )

            # Execute user code
            result = await asyncio.wait_for(
                asyncio.to_thread(sandbox.run_code, code),
                timeout=self.timeout
            )

            # Process results
            output = self._process_result(result, code, env)

            return output

        except asyncio.TimeoutError:
            raise REPLError(f"Execution timeout ({self.timeout}s) exceeded")
        except Exception as e:
            raise REPLError(f"Execution error: {str(e)}") from e
        finally:
            # Clean up sandbox
            try:
                cleanup = getattr(sandbox, 'kill', None)
                if cleanup is None:
                    cleanup = getattr(sandbox, 'close', None)
                if cleanup is not None:
                    await asyncio.to_thread(cleanup)
            except Exception:
                pass  # Best effort cleanup

    def _extract_code(self, text: str) -> str:
        """
        Extract code from markdown code blocks if present.

        Args:
            text: Raw text that might contain code

        Returns:
            Extracted code
        """
        # Check for markdown code blocks
        if '```python' in text:
            start = text.find('```python') + len('```python')
            end = text.find('```', start)
            if end != -1:
                return text[start:end].strip()

        if '```' in text:
            start = text.find('```') + 3
            end = text.find('```', start)
            if end != -1:
                return text[start:end].strip()

        return text

    def _build_env_setup(self, env: Dict[str, Any]) -> str:
        """
        Build Python code to set up environment variables.
        
        E2B sandbox allows all Python operations including:
        - All imports (standard library and installed packages)
        - All operators (arithmetic, logical, bitwise, etc.)
        - All Python features without restrictions
        
        Pre-imports common modules for convenience and backward compatibility.

        Args:
            env: Environment dict

        Returns:
            Python code to execute for setup
        """
        setup_lines = []

        # Pre-import commonly used modules (for backward compatibility and convenience)
        # Users can still import any additional modules they need
        setup_lines.append('import re')
        setup_lines.append('import json')
        setup_lines.append('import math')
        setup_lines.append('from datetime import datetime, timedelta')
        setup_lines.append('from collections import Counter, defaultdict')

        # Set string variables
        if 'context' in env:
            context = env['context'].replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            setup_lines.append(f'context = """{context}"""')

        if 'query' in env:
            query = env['query'].replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            setup_lines.append(f'query = """{query}"""')

        return '\n'.join(setup_lines)

    def _process_result(self, result: Any, code: str, env: Dict[str, Any]) -> str:
        """
        Process E2B execution result.

        Args:
            result: Result from E2B sandbox
            code: Original code
            env: Environment dict

        Returns:
            Processed output string
        """
        output = ""

        # Get stdout/stderr
        if hasattr(result, 'logs') and result.logs:
            if hasattr(result.logs, 'stdout'):
                output += ''.join(result.logs.stdout)
            if hasattr(result.logs, 'stderr'):
                stderr = ''.join(result.logs.stderr)
                if stderr:
                    output += f"\nStderr: {stderr}"

        # Get result value
        if hasattr(result, 'results') and result.results:
            for res in result.results:
                if hasattr(res, 'text'):
                    output += res.text + '\n'

        # Check for errors
        if hasattr(result, 'error') and result.error:
            raise REPLError(f"Execution error: {result.error}")

        # Handle expression evaluation (last line)
        if not output.strip():
            lines = code.strip().split('\n')
            if lines:
                last_line = lines[-1].strip()
                # Check if last line is an expression
                if last_line and not any(kw in last_line for kw in ['=', 'import', 'def', 'class', 'if', 'for', 'while', 'with']):
                    # The E2B result should include this, but add message if missing
                    if not output:
                        output = "Code executed successfully"

        if not output.strip():
            return "Code executed successfully (no output)"

        # Truncate output if too long
        if len(output) > self.max_output_chars:
            truncated = output[:self.max_output_chars]
            return f"{truncated}\n\n[Output truncated: {len(output)} chars total, showing first {self.max_output_chars}]"

        return output.strip()


class SandboxFactory:
    """Factory for creating sandbox executors with auto-detection.
    
    Sandbox types:
    - 'e2b': E2B cloud sandbox (fully permissive, allows all imports and operators)
    - 'restricted': RestrictedPython local sandbox (has import/operation restrictions)
    - 'auto': Automatically choose E2B if available, otherwise use RestrictedPython
    """

    @staticmethod
    def create(
        sandbox_type: Literal['e2b', 'restricted', 'auto'] = 'auto',
        timeout: int = 30,
        max_output_chars: int = 2000,
        e2b_api_key: Optional[str] = None
    ) -> SandboxExecutor:
        """
        Create a sandbox executor with auto-detection.

        Args:
            sandbox_type: Type of sandbox ('e2b', 'restricted', 'auto')
                - 'e2b': E2B cloud sandbox (unrestricted)
                - 'restricted': RestrictedPython (import restrictions apply)
                - 'auto': Try E2B first, fallback to RestrictedPython
            timeout: Execution timeout in seconds
            max_output_chars: Maximum output characters
            e2b_api_key: E2B API key (optional)

        Returns:
            SandboxExecutor instance

        Raises:
            ImportError: If required package is not available
        """
        if sandbox_type == 'restricted':
            # Use RestrictedPython
            return RestrictedPythonAdapter(timeout=timeout, max_output_chars=max_output_chars)

        elif sandbox_type == 'e2b':
            # Use E2B (will raise if not available)
            return E2BSandboxExecutor(
                timeout=timeout,
                max_output_chars=max_output_chars,
                api_key=e2b_api_key
            )

        else:  # auto
            # Try E2B first, fall back to RestrictedPython
            e2b_api_key = e2b_api_key or os.getenv('E2B_API_KEY')

            # Check if E2B is available and configured
            if e2b_api_key:
                try:
                    import e2b_code_interpreter
                    return E2BSandboxExecutor(
                        timeout=timeout,
                        max_output_chars=max_output_chars,
                        api_key=e2b_api_key
                    )
                except ImportError:
                    pass  # Fall through to RestrictedPython

            # Fall back to RestrictedPython
            return RestrictedPythonAdapter(timeout=timeout, max_output_chars=max_output_chars)


class RestrictedPythonAdapter(SandboxExecutor):
    """Adapter to make REPLExecutor conform to SandboxExecutor interface.
    
    NOTE: This executor has restrictions on imports and some operations.
    For unrestricted execution, use E2BSandboxExecutor instead.
    
    Restrictions include:
    - Import statements are blocked (pre-imported modules: re, json, math, datetime, Counter, defaultdict)
    - Some built-in functions are limited
    - File system and network access are blocked
    """

    def __init__(self, timeout: int = 5, max_output_chars: int = 2000):
        """
        Initialize RestrictedPython adapter.

        Args:
            timeout: Execution timeout (not enforced by RestrictedPython)
            max_output_chars: Maximum output characters
        """
        self.repl = REPLExecutor(timeout=timeout, max_output_chars=max_output_chars)

    def execute(self, code: str, env: Dict[str, Any]) -> str:
        """
        Execute code using RestrictedPython.

        Args:
            code: Python code
            env: Environment dict

        Returns:
            Execution result
        """
        return self.repl.execute(code, env)

    async def aexecute(self, code: str, env: Dict[str, Any]) -> str:
        """
        Async execute code using RestrictedPython.

        Args:
            code: Python code
            env: Environment dict

        Returns:
            Execution result
        """
        # RestrictedPython is sync, so run in thread pool
        return await asyncio.to_thread(self.repl.execute, code, env)
