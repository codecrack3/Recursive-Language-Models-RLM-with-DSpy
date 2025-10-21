"""Custom DSPy modules for Recursive Language Models."""

import re
from typing import Dict, Any, Optional, Callable, Tuple, Literal

regex_module = re

import dspy

from .signatures import RLMSignature, RLMInstructionSignature
from .parser import is_final, parse_response
from .sandbox import SandboxExecutor, SandboxFactory, normalize_e2b_api_key
from .types import SandboxType

class RLMError(Exception):
    """Base error for RLM modules."""
    pass


class MaxIterationsError(RLMError):
    """Max iterations exceeded."""
    pass


class RLMModule(dspy.Module):
    """
    Custom DSPy module for Recursive Language Model pattern.

    This module implements the REPL iteration loop:
    1. Generate code using DSPy LM
    2. Execute code in sandbox
    3. Feed results back to LM
    4. Repeat until FINAL() is called or max iterations reached
    """

    def __init__(
        self,
        max_iterations: int = 30,
        max_depth: int = 5,
        sandbox: Optional[SandboxExecutor] = None,
        use_instruction_signature: bool = True,
        logger: Optional[Any] = None,
        sandbox_type: Literal['e2b', 'restricted', 'auto'] = 'auto',
        e2b_api_key: Optional[str] = None
    ):
        """
        Initialize RLM module.

        Args:
            max_iterations: Maximum REPL iterations
            max_depth: Maximum recursion depth
            sandbox: Sandbox executor (if None, will auto-create)
            use_instruction_signature: Use enhanced signature with instructions
            logger: Optional console logger for verbose hooks
            sandbox_type: Preferred sandbox backend when auto-creating
            e2b_api_key: Explicit E2B API key override when using the E2B backend
        """
        super().__init__()

        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self._sandbox_type = sandbox_type
        self.sandbox_type = sandbox_type
        self._e2b_api_key = normalize_e2b_api_key(e2b_api_key)

        factory_kwargs = {'sandbox_type': self._sandbox_type}
        if self._e2b_api_key is not None:
            factory_kwargs['e2b_api_key'] = self._e2b_api_key

        self.sandbox = sandbox or SandboxFactory.create(**factory_kwargs)
        self.use_instruction_signature = use_instruction_signature
        self.logger = logger

        # Choose signature based on flag
        if use_instruction_signature:
            self.predict = dspy.Predict(RLMInstructionSignature)
        else:
            self.predict = dspy.Predict(RLMSignature)

        # Stats
        self._iterations = 0
        self._llm_calls = 0

    def set_sandbox_type(self, sandbox_type: Literal['e2b', 'restricted', 'auto']) -> None:
        """Set the sandbox type."""
        if sandbox_type not in ('e2b', 'restricted', 'auto'):
            raise ValueError(f"Invalid sandbox type: {sandbox_type}")
        factory_kwargs = {'sandbox_type': sandbox_type}
        if self._e2b_api_key is not None:
            factory_kwargs['e2b_api_key'] = self._e2b_api_key
        self.sandbox = SandboxFactory.create(**factory_kwargs)
        self._sandbox_type = sandbox_type
        self.sandbox_type = sandbox_type

    def forward(
        self,
        query: str,
        context: Optional[str] = None,
        *,
        depth: int = 0,
        recursive_llm_fn: Optional[Callable] = None,
        graph_tracker: Optional[Any] = None,
        node_id: Optional[str] = None
    ) -> dspy.Prediction:
        """
        Forward pass: execute the REPL loop.

        Args:
            query: User query
            context: Context to process
            depth: Current recursion depth
            recursive_llm_fn: Function for recursive calls
            graph_tracker: Optional graph tracker for visualization
            node_id: Optional node ID for graph tracking

        Returns:
            dspy.Prediction with answer and metadata

        Raises:
            MaxIterationsError: If max iterations exceeded
        """
        context_text = "" if context is None else str(context)
        context_size = len(context_text)

        # Build REPL environment
        repl_env = self._build_repl_env(
            query=query,
            context=context_text,
            recursive_llm_fn=recursive_llm_fn
        )
        
        # Populate env container for recursion tracking if present
        if recursive_llm_fn is not None and hasattr(recursive_llm_fn, '__closure__'):
            # Try to find the env_container in the closure and populate it
            if recursive_llm_fn.__closure__:
                for cell in recursive_llm_fn.__closure__:
                    try:
                        obj = cell.cell_contents
                        if isinstance(obj, dict) and 'env' in obj and obj.get('env') is None:
                            obj['env'] = repl_env
                            break
                    except (ValueError, AttributeError):
                        continue

        # Conversation history
        previous_output = ""
        last_code: Optional[str] = None
        
        # Track the current parent node ID for graph tracking
        # This will be updated after each operation to chain nodes correctly
        current_parent_id = node_id

        # Main REPL loop
        for iteration in range(self.max_iterations):
            self._iterations = iteration + 1
            iteration_index = iteration + 1

            if self.logger:
                self.logger.iteration_start(
                    iteration=iteration_index,
                    depth=depth,
                    query=query,
                    context_size=context_size
                )

            # Generate code using DSPy
            if self.use_instruction_signature:
                if self.logger:
                    self.logger.before_llm_call(
                        iteration=iteration_index,
                        depth=depth,
                        query=query,
                        context_size=context_size,
                        history=previous_output
                    )
                prediction = self.predict(
                    query=query,
                    context_size=str(context_size),
                    depth=str(depth),
                    previous_output=previous_output
                )
                code = prediction.code
                self._llm_calls += 1
                if self.logger:
                    self.logger.after_llm_call(
                        iteration=iteration_index,
                        depth=depth,
                        code=code,
                        reasoning=getattr(prediction, 'rationale', None)
                    )

                # Track LLM call as a separate node in graph
                llm_node_id = None
                if graph_tracker is not None:
                    from datetime import datetime

                    # Extract prompt/response from prediction
                    prompt_text = f"Query: {query}\nContext size: {context_size}\nDepth: {depth}\nPrevious: {previous_output}"
                    response_text = code

                    llm_node_id = graph_tracker.create_llm_call_node(
                        prompt=prompt_text,
                        response=response_text,
                        model='unknown',
                        depth=depth,
                        parent_id=current_parent_id,
                        iteration=iteration + 1,
                        timestamp=datetime.now().isoformat()
                    )

                    # Store current LLM node ID in REPL env for recursion tracking
                    if repl_env and '_current_llm_call_id' in repl_env:
                        repl_env['_current_llm_call_id']['value'] = llm_node_id
            else:
                # Build previous_attempts string
                previous_attempts = previous_output if previous_output else ""
                if self.logger:
                    self.logger.before_llm_call(
                        iteration=iteration_index,
                        depth=depth,
                        query=query,
                        context_size=context_size,
                        history=previous_attempts
                    )

                prediction = self.predict(
                    query=query,
                    context_size=str(context_size),
                    depth=str(depth),
                    previous_attempts=previous_attempts
                )
                code = prediction.code
                self._llm_calls += 1
                if self.logger:
                    self.logger.after_llm_call(
                        iteration=iteration_index,
                        depth=depth,
                        code=code,
                        reasoning=getattr(prediction, 'rationale', None)
                    )

                # Track LLM call as a separate node in graph
                llm_node_id = None
                if graph_tracker is not None:
                    from datetime import datetime

                    prompt_text = f"Query: {query}\nContext size: {context_size}\nDepth: {depth}\nPrevious: {previous_attempts}"
                    response_text = code

                    llm_node_id = graph_tracker.create_llm_call_node(
                        prompt=prompt_text,
                        response=response_text,
                        model='unknown',
                        depth=depth,
                        parent_id=current_parent_id,
                        iteration=iteration + 1,
                        timestamp=datetime.now().isoformat()
                    )

                    # Store current LLM node ID in REPL env for recursion tracking
                    if repl_env and '_current_llm_call_id' in repl_env:
                        repl_env['_current_llm_call_id']['value'] = llm_node_id

            # Sanitize code for RestrictedPython
            sanitized_code, blocked_reason = _sanitize_code_for_restricted_python(code)
            if self.logger:
                self.logger.before_execution(
                    iteration=iteration_index,
                    depth=depth,
                    code=sanitized_code,
                    blocked_reason=blocked_reason
                )

            # Early stop on duplicate code across iterations to reduce wasted calls
            if last_code is not None and sanitized_code.strip() == last_code.strip():
                if self.logger:
                    self.logger.on_duplicate_code(
                        iteration=iteration_index,
                        depth=depth,
                        code=sanitized_code
                    )
                raise MaxIterationsError(
                    "Generated identical code in consecutive iterations; stopping early to reduce LLM calls"
                )
            last_code = sanitized_code

            # Execute code in sandbox
            exec_error = None
            try:
                # If sanitizer blocked execution, surface concise guidance instead of running
                if blocked_reason is not None:
                    exec_result = blocked_reason
                else:
                    # print(f'Execution code: {sanitized_code}')
                    exec_result = self.sandbox.execute(sanitized_code, repl_env)
            except Exception as e:
                if self.logger:
                    self.logger.execution_exception(
                        iteration=iteration_index,
                        depth=depth,
                        error=str(e)
                    )
                exec_result = f"Error: {str(e)}"
                exec_error = str(e)

            # Check if FINAL was called during execution
            if repl_env.get('_final_result', {}).get('called', False):
                answer = repl_env['_final_result']['value']
                if answer is not None:
                    if self.logger:
                        self.logger.after_execution(
                            iteration=iteration_index,
                            depth=depth,
                            output=exec_result,
                            error=exec_error
                        )
                    if self.logger:
                        self.logger.on_final(
                            iteration=iteration_index,
                            depth=depth,
                            answer=answer
                        )
                    return dspy.Prediction(
                        answer=answer,
                        iterations=self._iterations,
                        llm_calls=self._llm_calls,
                        depth=depth
                    )

            # Fallback: Check for FINAL in code text (for backward compatibility)
            if is_final(code):
                answer = parse_response(code, repl_env)
                if answer is not None:
                    if self.logger:
                        self.logger.after_execution(
                            iteration=iteration_index,
                            depth=depth,
                            output=exec_result,
                            error=exec_error
                        )
                    if self.logger:
                        self.logger.on_final(
                            iteration=iteration_index,
                            depth=depth,
                            answer=answer
                        )
                    return dspy.Prediction(
                        answer=answer,
                        iterations=self._iterations,
                        llm_calls=self._llm_calls,
                        depth=depth
                    )

            # Map common RestrictedPython errors to concise hints
            hint = _map_restricted_error_to_hint(exec_result)
            if hint:
                exec_result = f"{exec_result}\n\nHint: {hint}"

            # Track code execution as a separate node in graph
            if graph_tracker is not None and llm_node_id is not None:
                exec_node_id = graph_tracker.create_code_execution_node(
                    code=code,
                    output=str(exec_result),
                    iteration=iteration + 1,
                    depth=depth,
                    parent_id=llm_node_id,  # Parent is the LLM call that generated this code
                    error=exec_error
                )
                
                # Update current parent for next iteration
                current_parent_id = exec_node_id
                
                # Store execution node ID in REPL env for recursion tracking
                if repl_env and '_current_exec_node_id' in repl_env:
                    repl_env['_current_exec_node_id']['value'] = exec_node_id

            # Update conversation context
            previous_output = f"Code:\n{sanitized_code}\n\nOutput:\n{exec_result}"
            if self.logger:
                self.logger.after_execution(
                    iteration=iteration_index,
                    depth=depth,
                    output=exec_result,
                    error=exec_error
                )

        # Max iterations exceeded
        if self.logger:
            self.logger.on_max_iterations(
                max_iterations=self.max_iterations,
                depth=depth
            )
        raise MaxIterationsError(
            f"Max iterations ({self.max_iterations}) exceeded without FINAL()"
        )

    def _build_repl_env(
        self,
        query: str,
        context: str,
        recursive_llm_fn: Optional[Callable]
    ) -> Dict[str, Any]:
        """
        Build REPL environment.

        Args:
            query: User query
            context: Context string
            recursive_llm_fn: Function for recursive calls

        Returns:
            Environment dict
        """
        # Create a container to store final results
        final_result = {'value': None, 'called': False}

        def FINAL(answer):
            """Store final answer and return it."""
            final_result['value'] = str(answer)
            final_result['called'] = True
            return answer

        def FINAL_VAR(var_name):
            """Store variable value as final answer."""
            # This will be called from within the executed code
            # so it has access to local variables through the environment
            final_result['value'] = str(var_name)
            final_result['called'] = True
            return var_name

        env: Dict[str, Any] = {
            'context': context,
            'query': query,
            're': regex_module,
            'FINAL': FINAL,
            'FINAL_VAR': FINAL_VAR,
            '_final_result': final_result,  # Store reference for later retrieval
            '_current_llm_call_id': {'value': None},  # Track current LLM call for recursion tracking
            '_current_exec_node_id': {'value': None},  # Track current execution node for recursion tracking
        }

        # Add recursive_llm if provided
        # We need to store a reference to the env in the recursive function's closure
        # so it can access _current_exec_node_id when called
        if recursive_llm_fn is not None:
            # Store env reference for the recursive function to access
            env['recursive_llm'] = recursive_llm_fn
            env['llm_query'] = recursive_llm_fn
            env['_recursive_llm_env_ref'] = env  # Self-reference for nested access

        return env

    @property
    def stats(self) -> Dict[str, int]:
        """Get execution statistics."""
        return {
            'iterations': self._iterations,
            'llm_calls': self._llm_calls,
        }

    def set_logger(self, logger: Optional[Any]) -> None:
        """Assign or update logging hooks."""
        self.logger = logger

def _sanitize_code_for_restricted_python(code: str) -> Tuple[str, Optional[str]]:
    """
    Remove disallowed imports and block execution when unsafe patterns are detected.

    Returns a tuple of (sanitized_code, blocked_reason). If blocked_reason is not None,
    the caller should not execute the code and instead return the reason to the LM.
    """
    if not code:
        return code, None

    original = code

    # Strip markdown fences if present (LM may include)
    if '```' in code:
        for fence in ('```repl', '```python', '```'):
            start = code.find(fence)
            if start != -1:
                start = start + len(fence)
                end = code.find('```', start)
                if end != -1:
                    code = code[start:end].strip()
                break

    # Disallow any import statements. The sandbox already provides needed modules.
    if re.search(r'^\s*import\s+|^\s*from\s+.+\s+import\s+', code, re.MULTILINE):
        # Try to remove benign imports for provided modules; block others.
        safe_modules = ['re', 'json', 'math', 'datetime', 'collections', 'Counter', 'defaultdict']
        sanitized = []
        blocked = False
        for line in code.split('\n'):
            if re.match(r'^\s*(import|from)\s+', line):
                if any(sm in line for sm in safe_modules):
                    # Drop the line silently (already provided)
                    continue
                else:
                    blocked = True
                    continue
            sanitized.append(line)
        code = '\n'.join(sanitized)
        if blocked:
            reason = (
                "Execution blocked: imports are not allowed in the restricted sandbox. "
                "Use provided modules (re, json, math, datetime, Counter, defaultdict) without importing."
            )
            return code, reason

    # Disallow dangerous builtins/keywords patterns
    dangerous_patterns = [
        r'__import__\s*\(',
        r'open\s*\(',
        r'eval\s*\(',
        r'exec\s*\(',
        r'os\.',
        r'subprocess\.',
        r'sys\.(?:modules|stdin|stdout|stderr|exit)'
    ]
    for pat in dangerous_patterns:
        if re.search(pat, code):
            return code, (
                "Execution blocked: disallowed operation detected for RestrictedPython. "
                "Avoid file/network/OS/process access and eval/exec."
            )

    return code, None


def _map_restricted_error_to_hint(output: str) -> Optional[str]:
    """Map common RestrictedPython/Eval errors to concise guidance for the LLM."""
    if not output:
        return None
    text = str(output)
    mappings = [
        (r'__import__\s+not\s+found|Compilation error: __import__',
         'Do not import modules. Use provided: re, json, math, datetime, Counter, defaultdict.'),
        (r'NameError: re\s+is\s+not\s+defined',
         'The regex module `re` is already injected. Do not import; just use `re`.'),
        (r'Compilation error:',
         'Ensure your output is pure Python code (no markdown).'),
        (r'Execution error: .*timeout',
         'Reduce work; operate on smaller slices of `context` and print concise results.'),
        (r'NotImplementedError|Restricted',
         'Avoid restricted features (I/O, OS, subprocess). Work only with strings and provided modules.')
    ]
    for pattern, hint in mappings:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return hint
    return None


class RLMChainOfThought(RLMModule):
    """
    RLM module with chain-of-thought reasoning.

    This variant adds explicit reasoning steps before code generation,
    which can improve the quality of generated code.
    """

    def __init__(
        self,
        max_iterations: int = 30,
        max_depth: int = 5,
        sandbox: Optional[SandboxExecutor] = None,
        logger: Optional[Any] = None,
        sandbox_type: Literal['e2b', 'restricted', 'auto'] = 'auto',
        e2b_api_key: Optional[str] = None
    ):
        """
        Initialize RLM with chain-of-thought.

        Args:
            max_iterations: Maximum REPL iterations
            max_depth: Maximum recursion depth
            sandbox: Sandbox executor
            logger: Optional logger object for logging hooks
            sandbox_type: Preferred sandbox backend when auto-creating
            e2b_api_key: Explicit E2B API key override when using the E2B backend
        """
        super().__init__(
            max_iterations=max_iterations,
            max_depth=max_depth,
            sandbox=sandbox,
            use_instruction_signature=False,
            logger=logger,
            sandbox_type=sandbox_type,
            e2b_api_key=e2b_api_key
        )

        self._cot_predictor = dspy.ChainOfThought(RLMSignature)
        self._last_reasoning: Optional[str] = None
        self.predict = self._PredictWrapper(self, self._cot_predictor)

    class _PredictWrapper:
        """Proxy that tracks rationale from ChainOfThought predictions."""

        def __init__(self, parent: "RLMChainOfThought", predictor: Callable):
            self._parent = parent
            self._predictor = predictor

        def __call__(self, *args, **kwargs):
            prediction = self._predictor(*args, **kwargs)
            self._parent._last_reasoning = getattr(prediction, 'rationale', None)
            return prediction

        def __getattr__(self, item):
            return getattr(self._predictor, item)

    def forward(
        self,
        query: str,
        context: Optional[str] = None,
        *,
        depth: int = 0,
        recursive_llm_fn: Optional[Callable] = None,
        graph_tracker: Optional[Any] = None,
        node_id: Optional[str] = None
    ) -> dspy.Prediction:
        """
        Forward pass with chain-of-thought that reuses the optimized RLM loop.
        """
        self._last_reasoning = None
        prediction = super().forward(
            query=query,
            context=context,
            depth=depth,
            recursive_llm_fn=recursive_llm_fn,
            graph_tracker=graph_tracker,
            node_id=node_id
        )

        if self._last_reasoning is not None:
            setattr(prediction, 'reasoning', self._last_reasoning)

        return prediction
