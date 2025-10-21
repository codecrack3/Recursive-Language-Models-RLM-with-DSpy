"""DSPy-based RLM implementation."""

import asyncio
import json
import os
from typing import Optional, Dict, Any, Literal

import dspy
import networkx as nx

from .modules import RLMModule, RLMChainOfThought, MaxIterationsError, RLMError
from .sandbox import SandboxExecutor, SandboxFactory, normalize_e2b_api_key
from .graph import RLMGraphTracker
from .logger import RLMRunLogger


DEFAULT_QUERY = (
    "Please read through the context and answer any queries or respond to any "
    "instructions contained within it."
)


class MaxDepthError(RLMError):
    """Max recursion depth exceeded."""
    pass


class RLMDSPy:
    """
    Recursive Language Model using DSPy.

    This is the DSPy-based implementation of RLM that uses:
    - DSPy for LLM orchestration and prompting
    - E2B or RestrictedPython for sandbox execution
    - Custom RLMModule for the REPL iteration pattern
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
        recursive_model: Optional[str] = None,
        max_tokens: Optional[int] = 16000,
        temperature: Optional[float] = 0.2,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_depth: int = 5,
        max_iterations: int = 10,
        sandbox_type: Literal['e2b', 'restricted', 'auto'] = 'auto',
        e2b_api_key: Optional[str] = None,
        module_type: Literal['basic', 'cot'] = 'basic',
        enable_graph_tracking: bool = False,
        graph_output_path: str = "./rlm_graph.html",
        enable_history: bool = False,
        enable_logging: bool = False,
        truncate_text: bool = True,
        truncate_code: bool = True,
        _current_depth: int = 0,
        _sandbox: Optional[SandboxExecutor] = None,
        _graph_tracker: Optional[RLMGraphTracker] = None,
        _node_id: Optional[str] = None,
        _logger: Optional[RLMRunLogger] = None,
        **llm_kwargs: Any
    ):
        """
        Initialize DSPy-based RLM.

        Args:
            model: Model name for DSPy (e.g., "gpt-4o-mini", "claude-3-5-sonnet")
            recursive_model: Optional cheaper model for recursive calls
            max_tokens: Maximum tokens per LLM call (default: 2048)
            temperature: LLM temperature for generation (default: 0.3)
            api_base: Optional API base URL
            api_key: Optional API key
            max_depth: Maximum recursion depth
            max_iterations: Maximum REPL iterations per call
            sandbox_type: Type of sandbox ('e2b', 'restricted', 'auto')
            e2b_api_key: Optional API key for E2B sandbox usage
            module_type: RLM module type ('basic' or 'cot' for chain-of-thought)
            enable_graph_tracking: Enable NetworkX graph tracking and visualization
            graph_output_path: Path to save graph HTML visualization
            enable_history: Enable DSPy history tracking for debugging LLM calls
            enable_logging: Enable colorful console logging
            truncate_text: Truncate non-code text when logging (ignored if logger provided)
            truncate_code: Truncate code blocks when logging (ignored if logger provided)
            _current_depth: Internal depth tracker
            _sandbox: Internal sandbox instance (shared across recursive calls)
            _graph_tracker: Internal graph tracker (shared across recursive calls)
            _node_id: Internal node ID for current call
            _logger: Internal logger instance (shared across recursive calls)
            **llm_kwargs: Additional DSPy LM parameters
        """
        self.model = model
        self.recursive_model = recursive_model or model
        self.api_base = api_base
        self.api_key = api_key
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.sandbox_type = sandbox_type
        self.module_type = module_type
        self.enable_graph_tracking = enable_graph_tracking
        self.graph_output_path = graph_output_path
        self.enable_history = enable_history
        self.enable_logging = enable_logging or (_logger is not None)
        self._truncate_text = truncate_text
        self._truncate_code = truncate_code
        self._current_depth = _current_depth
        self._node_id = _node_id
        self._repl_env = None  # Will be set by module during execution
        self.logger = _logger if _logger is not None else (
            RLMRunLogger(
                enabled=True,
                truncate_text=truncate_text,
                truncate_code=truncate_code
            ) if self.enable_logging else None
        )

        # Build LLM kwargs, only including non-None values
        self.llm_kwargs = {**llm_kwargs}
        if max_tokens is not None:
            self.llm_kwargs['max_tokens'] = max_tokens
        if temperature is not None:
            self.llm_kwargs['temperature'] = temperature

        # Track sandbox credentials
        provided_e2b_key = normalize_e2b_api_key(e2b_api_key)
        env_e2b_key = normalize_e2b_api_key(os.getenv('E2B_API_KEY'))
        self._provided_e2b_api_key = provided_e2b_key or None
        self.e2b_api_key = provided_e2b_key or env_e2b_key or None

        # Initialize DSPy LM
        self.lm = self._setup_dspy_lm()

        # Create or reuse sandbox
        if _sandbox is not None:
            self.sandbox = _sandbox
            if self.e2b_api_key is None and hasattr(_sandbox, 'api_key'):
                sandbox_key = getattr(_sandbox, 'api_key')
                if sandbox_key:
                    self.e2b_api_key = sandbox_key
        else:
            factory_kwargs = {'sandbox_type': sandbox_type}
            if self._provided_e2b_api_key is not None:
                factory_kwargs['e2b_api_key'] = self._provided_e2b_api_key
            self.sandbox = SandboxFactory.create(**factory_kwargs)
        
        # Create or reuse graph tracker
        if _graph_tracker is not None:
            self.graph_tracker = _graph_tracker
        elif enable_graph_tracking and _current_depth == 0:
            self.graph_tracker = RLMGraphTracker()
        else:
            self.graph_tracker = None

        # Create RLM module
        if module_type == 'cot':
            self.rlm_module = RLMChainOfThought(
                max_iterations=max_iterations,
                max_depth=max_depth,
                sandbox=self.sandbox,
                logger=self.logger,
                sandbox_type=self.sandbox_type,
                e2b_api_key=self.e2b_api_key
            )
        else:
            self.rlm_module = RLMModule(
                max_iterations=max_iterations,
                max_depth=max_depth,
                sandbox=self.sandbox,
                use_instruction_signature=True,
                logger=self.logger,
                sandbox_type=self.sandbox_type,
                e2b_api_key=self.e2b_api_key
            )

        # Stats
        self._total_llm_calls = 0

    def _setup_dspy_lm(self):
        """Set up DSPy language model and return the LM instance."""
        import os

        # Choose model based on depth
        model = self.model if self._current_depth == 0 else self.recursive_model

        # Build kwargs for DSPy LM
        lm_kwargs = {**self.llm_kwargs}
        if self.api_key:
            lm_kwargs['api_key'] = self.api_key
        if self.api_base:
            lm_kwargs['api_base'] = self.api_base

        # Detect provider from model name
        if model.startswith('openrouter/'):
            # Get API key (prefer parameter, fallback to env var)
            api_key = lm_kwargs.pop('api_key', None) or os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError(
                    "OpenRouter requires OPENROUTER_API_KEY. "
                    "Get your key at https://openrouter.ai/keys"
                )

            # Configure OpenRouter via OpenAI-compatible API
            lm = dspy.LM(
                model=model,
                api_key=api_key,
                **lm_kwargs
            )
        elif model.startswith('gpt-') or model.startswith('openai/'):
            # OpenAI models
            lm = dspy.LM(model=model.replace('openai/', ''), **lm_kwargs)
        elif model.startswith('claude-') or model.startswith('anthropic/'):
            # Anthropic models
            lm = dspy.LM(model=model.replace('anthropic/', ''), **lm_kwargs)
        elif model.startswith('ollama/'):
            # Ollama local models
            model_name = model.replace('ollama/', '')
            base_url = lm_kwargs.pop('api_base', 'http://localhost:11434')
            lm = dspy.LM(model=model_name, base_url=base_url, **lm_kwargs)
        else:
            # Generic LM (try OpenAI-compatible)
            lm = dspy.LM(model=model, **lm_kwargs)

        # Configure DSPy to use this LM
        # Enable experimental features if history tracking is enabled
        if self.enable_history:
            dspy.settings.configure(lm=lm, experimental=True)
        else:
            dspy.settings.configure(lm=lm)
        
        return lm

    def completion(
        self,
        context: Any,
        query: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Given a query and a (potentially long) context, recursively call the LM
        to explore the context and provide an answer using a REPL environment.

        Args:
            context: The document or messages to analyze (string, dict, or list)
            query: Optional question to answer (defaults to DEFAULT_QUERY)
            **kwargs: Additional parameters forwarded to the async variant

        Returns:
            Final answer string
        """
        # Check if there's already a running event loop
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(self.acompletion(context=context, query=query, **kwargs))
        
        # We're inside an async context, need to run in a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, self.acompletion(context=context, query=query, **kwargs))
            return future.result()

    async def acompletion(
        self,
        context: Any,
        query: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Main async completion method mirroring CODE.MD semantics.

        Args:
            context: Context to process (optional, can pass query here)
            query: Optional user query (falls back to DEFAULT_QUERY)
            **kwargs: Additional parameters

        Returns:
            Final answer string

        Raises:
            MaxIterationsError: If max iterations exceeded
            MaxDepthError: If max recursion depth exceeded
        """
        if query is None:
            query = DEFAULT_QUERY

        context_text = "" if context is None else str(context)
        context_size = len(context_text)

        if self.logger:
            self.logger.run_start(
                depth=self._current_depth,
                query=query,
                context_size=context_size
            )

        # Check depth
        if self._current_depth >= self.max_depth:
            if self.logger:
                self.logger.on_error(
                    depth=self._current_depth,
                    error=f"Max recursion depth ({self.max_depth}) exceeded"
                )
            raise MaxDepthError(f"Max recursion depth ({self.max_depth}) exceeded")

        # Create node in graph if tracking enabled (for legacy compatibility)
        # In fine-grained tracking mode, we don't create a root node here
        # Instead, nodes are created per LLM call and code execution
        if self.graph_tracker is not None and self._node_id is None:
            # Don't create a legacy root node in fine-grained mode
            # The first LLM call will be the root
            pass

        # Create recursive function
        # We'll pass a wrapper that will be updated with the actual env by the module
        env_container = {'env': None}
        recursive_llm_fn = self._make_recursive_fn(env_container)

        # Execute RLM module
        try:
            prediction = self.rlm_module(
                query=query,
                context=context_text,
                depth=self._current_depth,
                recursive_llm_fn=recursive_llm_fn,
                graph_tracker=self.graph_tracker,
                node_id=self._node_id
            )

            # Track stats
            self._total_llm_calls += self.rlm_module.stats['llm_calls']

            # Update graph node with results
            if self.graph_tracker is not None and self._node_id is not None:
                self.graph_tracker.update_node(
                    node_id=self._node_id,
                    answer=prediction.answer,
                    iterations=self.rlm_module.stats.get('iterations', 0),
                    llm_calls=self.rlm_module.stats.get('llm_calls', 0)
                )
            
            # Save graph if this is root node (moved outside to work with fine-grained tracking)
            if self.graph_tracker is not None and self._current_depth == 0:
                self.graph_tracker.save_html(self.graph_output_path)

            if self.logger:
                self.logger.run_end(depth=self._current_depth, answer=prediction.answer)

            return prediction.answer

        except MaxIterationsError as e:
            # Update node with error
            if self.graph_tracker is not None and self._node_id is not None:
                self.graph_tracker.update_node(
                    node_id=self._node_id,
                    error=str(e)
                )
            if self.logger:
                self.logger.on_error(depth=self._current_depth, error=str(e))
            raise e
        except Exception as e:
            # Update node with error
            if self.graph_tracker is not None and self._node_id is not None:
                self.graph_tracker.update_node(
                    node_id=self._node_id,
                    error=str(e)
                )
            if self.logger:
                self.logger.on_error(
                    depth=self._current_depth,
                    error=str(e)
                )
            raise RLMError(f"RLM execution failed: {str(e)}") from e

    def _make_recursive_fn(self, env_container=None):
        """
        Create recursive LLM function for REPL.
        
        Args:
            env_container: Optional dict with 'env' key that will be populated with REPL env
        
        Returns:
            Function that can be called from REPL for recursive processing
        """
        def recursive_llm(sub_query: str, sub_context: str) -> str:
            """
            Recursively process sub-context.

            Args:
                sub_query: Query for sub-context
                sub_context: Sub-context to process

            Returns:
                Answer from recursive call
            """
            if self._current_depth + 1 >= self.max_depth:
                return f"Max recursion depth ({self.max_depth}) reached"

            # Determine the parent node ID for the child
            # In the new fine-grained tracking system, we want to link to the
            # execution node that called recursive_llm, not the RLM root node
            parent_node_id = self._node_id
            
            # Try to get the current execution node from the REPL environment
            # This will be the actual parent for the child's first LLM call
            if env_container is not None and env_container.get('env') is not None:
                repl_env = env_container['env']
                if '_current_exec_node_id' in repl_env:
                    exec_node_id = repl_env['_current_exec_node_id'].get('value')
                    if exec_node_id is not None:
                        parent_node_id = exec_node_id
            
            # Create sub-RLM with increased depth
            sub_rlm = RLMDSPy(
                model=self.recursive_model,
                recursive_model=self.recursive_model,
                api_base=self.api_base,
                api_key=self.api_key,
                max_depth=self.max_depth,
                max_iterations=self.max_iterations,
                sandbox_type=self.sandbox_type,
                module_type=self.module_type,
                enable_graph_tracking=False,  # Don't reinitialize tracker
                graph_output_path=self.graph_output_path,
                enable_history=self.enable_history,  # Preserve history setting
                enable_logging=self.enable_logging,
                truncate_text=self._truncate_text,
                truncate_code=self._truncate_code,
                _current_depth=self._current_depth + 1,
                _sandbox=self.sandbox,  # Share sandbox
                _graph_tracker=self.graph_tracker,  # Share tracker
                _node_id=parent_node_id,  # Pass parent node ID (execution node)
                _logger=self.logger,  # Share logger for consistent logging
                e2b_api_key=self.e2b_api_key,
                **self.llm_kwargs
            )

            # Synchronous call (required for REPL environment). Match CODE.MD signature.
            return sub_rlm.completion(sub_context, sub_query)

        return recursive_llm

    def get_history(self) -> list:
        """
        Get the LLM call history for debugging.
        
        Returns:
            List of LLM call history entries (prompts and responses)
            
        Note:
            History tracking must be enabled via enable_history=True
        """
        if not self.enable_history:
            print("History tracking is not enabled. Set enable_history=True when creating RLM instance.")
            return []
        
        # Get history from DSPy LM
        if hasattr(self.lm, 'history'):
            return self.lm.history
        return []
    
    def print_history(self, detailed: bool = True, max_length: int = 1000) -> None:
        """
        Print the LLM call history in a readable format.
        
        Args:
            detailed: If True, show full prompts/responses. If False, show summary only.
            max_length: Maximum length of text to display per field
        """
        if not self.enable_history:
            print("History tracking is not enabled. Set enable_history=True when creating RLM instance.")
            return
        
        history = self.get_history()
        
        if not history:
            print("No history available yet.")
            return
        
        print("\n" + "=" * 80)
        print(f"LLM CALL HISTORY ({len(history)} calls)")
        print("=" * 80)
        
        for i, entry in enumerate(history, 1):
            print(f"\n{'─' * 80}")
            print(f"Call #{i}")
            print(f"{'─' * 80}")
            
            if not isinstance(entry, dict):
                print(entry)
                continue
                
            # Show model and timestamp info
            if 'model' in entry:
                print(f"Model: {entry['model']}")
            if 'timestamp' in entry:
                print(f"Timestamp: {entry['timestamp']}")
            if 'usage' in entry and entry['usage']:
                usage = entry['usage']
                if hasattr(usage, '__dict__'):
                    print(f"Usage: {usage.__dict__}")
                else:
                    print(f"Usage: {usage}")
            
            # Show messages (the actual prompt/input)
            if 'messages' in entry and entry['messages']:
                print(f"\n{'─' * 40}")
                print("MESSAGES (Input):")
                print(f"{'─' * 40}")
                messages = entry['messages']
                if isinstance(messages, list):
                    for msg in messages:
                        if isinstance(msg, dict):
                            role = msg.get('role', 'unknown')
                            content = msg.get('content', '')
                            if detailed:
                                # Show full content with truncation
                                content_str = str(content)
                                if len(content_str) > max_length:
                                    print(f"  [{role.upper()}]: {content_str[:max_length]}...")
                                    print(f"  ... (truncated, {len(content_str)} total chars)")
                                else:
                                    print(f"  [{role.upper()}]: {content_str}")
                            else:
                                print(f"  [{role.upper()}]: {str(content)[:200]}...")
                        else:
                            print(f"  {msg}")
                else:
                    print(f"  {str(messages)[:500]}")
            
            # Show response/output
            if 'outputs' in entry and entry['outputs']:
                print(f"\n{'─' * 40}")
                print("OUTPUTS (Response):")
                print(f"{'─' * 40}")
                outputs = entry['outputs']
                if isinstance(outputs, list):
                    for j, output in enumerate(outputs, 1):
                        if detailed:
                            output_str = str(output)
                            if len(output_str) > max_length:
                                print(f"  Output #{j}: {output_str[:max_length]}...")
                                print(f"  ... (truncated, {len(output_str)} total chars)")
                            else:
                                print(f"  Output #{j}: {output_str}")
                        else:
                            print(f"  Output #{j}: {str(output)[:200]}...")
                else:
                    print(f"  {str(outputs)[:500]}")
            
            # Show response object
            if 'response' in entry and entry['response']:
                response = entry['response']
                if hasattr(response, 'choices') and response.choices:
                    print(f"\n{'─' * 40}")
                    print("RESPONSE CHOICES:")
                    print(f"{'─' * 40}")
                    for j, choice in enumerate(response.choices, 1):
                        if hasattr(choice, 'message'):
                            msg = choice.message
                            content = getattr(msg, 'content', str(msg))
                            if detailed:
                                content_str = str(content)
                                if len(content_str) > max_length:
                                    print(f"  Choice #{j}: {content_str[:max_length]}...")
                                    print(f"  ... (truncated, {len(content_str)} total chars)")
                                else:
                                    print(f"  Choice #{j}: {content_str}")
                            else:
                                print(f"  Choice #{j}: {str(content)[:200]}...")
        
        print("\n" + "=" * 80)
    
    def clear_history(self) -> None:
        """Clear the LLM call history."""
        if hasattr(self.lm, 'history'):
            self.lm.history = []
    
    def save_history(self, filepath: str, pretty: bool = True) -> None:
        """
        Save LLM call history to a JSON file.
        
        Args:
            filepath: Path to save JSON file
            pretty: If True, save with indentation for readability
            
        Example:
            rlm.save_history("./logs/history.json")
            rlm.save_history("./history.json", pretty=False)  # Compact format
        """
        import json
        import os
        from datetime import datetime

        # create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        
        if not self.enable_history:
            print("History tracking is not enabled. Set enable_history=True when creating RLM instance.")
            return
        
        history = self.get_history()
        
        if not history:
            print("No history to save.")
            return
        
        # Convert history to JSON-serializable format
        serializable_history = []
        for entry in history:
            if isinstance(entry, dict):
                serialized_entry = {}
                for key, value in entry.items():
                    # Convert non-serializable objects to strings
                    if key == 'response' or key == 'usage':
                        if hasattr(value, '__dict__'):
                            serialized_entry[key] = str(value.__dict__)
                        else:
                            serialized_entry[key] = str(value)
                    elif key == 'outputs':
                        # Convert outputs list
                        if isinstance(value, list):
                            serialized_entry[key] = [str(item) for item in value]
                        else:
                            serialized_entry[key] = str(value)
                    elif key == 'messages':
                        # Keep messages as-is (already dicts)
                        serialized_entry[key] = value
                    else:
                        try:
                            # Try to serialize directly
                            json.dumps(value)
                            serialized_entry[key] = value
                        except (TypeError, ValueError):
                            # Convert to string if not serializable
                            serialized_entry[key] = str(value)
                serializable_history.append(serialized_entry)
            else:
                serializable_history.append(str(entry))
        
        # Create output structure
        output = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_calls': len(history),
                'model': self.model,
                'max_iterations': self.max_iterations,
                'max_depth': self.max_depth
            },
            'history': serializable_history
        }
        
        # Save to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(output, f, ensure_ascii=False)
            print(f"✅ History saved to: {filepath}")
            print(f"   Total calls: {len(history)}")
        except Exception as e:
            print(f"❌ Error saving history: {e}")

    @property
    def stats(self) -> Dict[str, int]:
        """Get execution statistics."""
        stats = {
            'llm_calls': self._total_llm_calls + self.rlm_module.stats.get('llm_calls', 0),
            'iterations': self.rlm_module.stats.get('iterations', 0),
            'depth': self._current_depth,
        }
        
        # Add graph stats if tracking enabled
        if self.graph_tracker is not None:
            stats['graph'] = self.graph_tracker.get_stats()
        
        return stats
    
    def get_graph(self) -> Optional[nx.DiGraph]:
        """
        Get the NetworkX graph object.
        
        Returns:
            NetworkX DiGraph if tracking is enabled, None otherwise
        """
        if self.graph_tracker is None:
            return None
        return self.graph_tracker.get_graph()
    
    def save_graph(self, output_path: Optional[str] = None) -> None:
        """
        Save graph visualization to HTML file.
        
        Args:
            output_path: Path to save HTML (uses default if None)
        """
        if self.graph_tracker is None:
            print("Graph tracking is not enabled. Set enable_graph_tracking=True")
            return
        
        path = output_path or self.graph_output_path
        self.graph_tracker.save_html(path)

class RLM(RLMDSPy):
    """
    Alias for RLMDSPy to maintain API compatibility.

    This allows users to use `from rlm import RLM` and get the DSPy implementation.
    For explicit backend selection, use the factory in __init__.py
    """
    pass
