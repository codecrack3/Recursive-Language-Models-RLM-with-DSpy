"""Graph tracking and visualization for recursive LLM calls."""

import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Literal
import networkx as nx
from pyvis.network import Network


@dataclass
class LLMCall:
    """Single LLM API call metadata.

    Attributes:
        call_id: Unique identifier for this call
        iteration: REPL iteration number when call was made
        prompt: Full prompt sent to LLM
        response: Full response from LLM
        model: Model name used
        tokens_prompt: Token count for prompt (optional)
        tokens_completion: Token count for completion (optional)
        latency_ms: Call latency in milliseconds (optional)
        timestamp: ISO timestamp of call (optional)
        sequence_number: Global sequence number across all nodes (optional)
        triggered_recursion: Whether this call triggered a recursive call (optional)
        spawned_node_id: ID of child node spawned by this call (optional)
    """
    call_id: str
    iteration: int
    prompt: str
    response: str
    model: str
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    latency_ms: Optional[float] = None
    timestamp: Optional[str] = None
    sequence_number: Optional[int] = None
    triggered_recursion: bool = False
    spawned_node_id: Optional[str] = None

    @property
    def total_tokens(self) -> Optional[int]:
        """Get total tokens (prompt + completion)."""
        if self.tokens_prompt is not None and self.tokens_completion is not None:
            return self.tokens_prompt + self.tokens_completion
        return None

    def get_prompt_preview(self, max_len: int = 100) -> str:
        """Get truncated prompt for display."""
        if len(self.prompt) <= max_len:
            return self.prompt
        return self.prompt[:max_len] + "..."

    def get_response_preview(self, max_len: int = 100) -> str:
        """Get truncated response for display."""
        if len(self.response) <= max_len:
            return self.response
        return self.response[:max_len] + "..."


@dataclass
class REPLStep:
    """Single REPL iteration step."""
    iteration: int
    code: str
    output: str
    timestamp: Optional[str] = None


@dataclass
class GraphNode:
    """
    Node representing a single operation in the recursion graph.
    
    Can represent three types of operations:
    - 'llm_call': A single LLM API call
    - 'code_execution': A single code execution step
    - 'rlm_root': Legacy RLM root node (deprecated, for backward compatibility)

    Attributes:
        node_id: Unique identifier for this node
        node_type: Type of operation ('llm_call', 'code_execution', 'rlm_root')
        depth: Recursion depth level
        parent_id: ID of parent node (None for root)
        status: Node status ('pending', 'success', 'error')
        error: Error message if call failed
        sequence_number: Global sequence number for ordering
        
        # For LLM call nodes:
        prompt: LLM prompt text
        response: LLM response text
        model: Model name
        tokens_prompt: Prompt token count
        tokens_completion: Completion token count
        latency_ms: Call latency in milliseconds
        iteration: REPL iteration number
        
        # For code execution nodes:
        code: Executed code
        output: Execution output
        iteration: REPL iteration number
        
        # For legacy RLM root nodes (deprecated):
        query: User query/instruction
        context: Context provided
        answer: Final answer
        iterations: Number of REPL iterations
        llm_calls: Number of LLM API calls
        repl_steps: List of intermediate REPL steps
        llm_calls_list: List of individual LLM calls with metadata
        total_tokens: Total tokens consumed
        total_latency_ms: Total latency
        triggering_llm_call_id: ID of parent LLM call that spawned this node
    """
    node_id: str
    node_type: Literal['llm_call', 'code_execution', 'rlm_root'] = 'rlm_root'
    depth: int = 0
    parent_id: Optional[str] = None
    status: str = "pending"
    error: Optional[str] = None
    sequence_number: Optional[int] = None
    
    # LLM call fields
    prompt: str = ""
    response: str = ""
    model: str = ""
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    latency_ms: Optional[float] = None
    
    # Code execution fields
    code: str = ""
    output: str = ""
    
    # Common field
    iteration: int = 0
    
    # Legacy fields (for backward compatibility with rlm_root nodes)
    query: str = ""
    context: str = ""
    answer: str = ""
    iterations: int = 0
    llm_calls: int = 0
    repl_steps: List[REPLStep] = field(default_factory=list)
    llm_calls_list: List[LLMCall] = field(default_factory=list)
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    triggering_llm_call_id: Optional[str] = None
    
    @property
    def total_tokens_computed(self) -> Optional[int]:
        """Get total tokens for LLM call nodes."""
        if self.node_type == 'llm_call' and self.tokens_prompt is not None and self.tokens_completion is not None:
            return self.tokens_prompt + self.tokens_completion
        return None
    
    def get_context_preview(self, max_len: int = 100) -> str:
        """Get truncated context for display."""
        if len(self.context) <= max_len:
            return self.context
        return self.context[:max_len] + "..."
    
    def get_query_preview(self, max_len: int = 50) -> str:
        """Get truncated query for display."""
        if len(self.query) <= max_len:
            return self.query
        return self.query[:max_len] + "..."
    
    def get_prompt_preview(self, max_len: int = 100) -> str:
        """Get truncated prompt for display."""
        if len(self.prompt) <= max_len:
            return self.prompt
        return self.prompt[:max_len] + "..."
    
    def get_response_preview(self, max_len: int = 100) -> str:
        """Get truncated response for display."""
        if len(self.response) <= max_len:
            return self.response
        return self.response[:max_len] + "..."
    
    def get_code_preview(self, max_len: int = 100) -> str:
        """Get truncated code for display."""
        if len(self.code) <= max_len:
            return self.code
        return self.code[:max_len] + "..."
    
    def get_output_preview(self, max_len: int = 100) -> str:
        """Get truncated output for display."""
        if len(self.output) <= max_len:
            return self.output
        return self.output[:max_len] + "..."


class RLMGraphTracker:
    """
    Tracks recursive LLM calls as a directed graph.
    
    Uses NetworkX to build a call hierarchy graph and pyvis for
    interactive HTML visualization.
    """
    
    def __init__(self):
        """Initialize graph tracker."""
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.root_node_id: Optional[str] = None
        self._llm_call_sequence_counter: int = 0  # Global sequence counter for timeline
        self._current_call_ids: Dict[str, str] = {}  # Track current call ID for each node
        self._operation_sequence_counter: int = 0  # Global sequence counter for all operations
    
    def create_node(
        self,
        query: str = "",
        context: str = "",
        depth: int = 0,
        parent_id: Optional[str] = None
    ) -> str:
        """
        Create a new node in the graph.
        
        Args:
            query: User query
            context: Context string
            depth: Recursion depth
            parent_id: Parent node ID (None for root)
        
        Returns:
            Unique node ID
        """
        node_id = str(uuid.uuid4())
        
        node = GraphNode(
            node_id=node_id,
            query=query,
            context=context,
            depth=depth,
            parent_id=parent_id
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **self._node_to_dict(node))
        
        # Set root if this is the first node
        if self.root_node_id is None:
            self.root_node_id = node_id
        
        # Add edge from parent if exists
        if parent_id is not None:
            self.graph.add_edge(parent_id, node_id)
        
        return node_id
    
    def create_llm_call_node(
        self,
        prompt: str,
        response: str,
        model: str,
        depth: int,
        parent_id: Optional[str] = None,
        iteration: int = 0,
        tokens_prompt: Optional[int] = None,
        tokens_completion: Optional[int] = None,
        latency_ms: Optional[float] = None,
        timestamp: Optional[str] = None
    ) -> str:
        """
        Create a new LLM call node in the graph.
        
        Args:
            prompt: LLM prompt text
            response: LLM response text
            model: Model name
            depth: Recursion depth
            parent_id: Parent node ID (None for root)
            iteration: REPL iteration number
            tokens_prompt: Token count for prompt
            tokens_completion: Token count for completion
            latency_ms: Call latency in milliseconds
            timestamp: ISO timestamp
        
        Returns:
            Unique node ID
        """
        node_id = str(uuid.uuid4())
        
        # Increment sequence counter
        self._operation_sequence_counter += 1
        self._llm_call_sequence_counter += 1
        
        node = GraphNode(
            node_id=node_id,
            node_type='llm_call',
            depth=depth,
            parent_id=parent_id,
            prompt=prompt,
            response=response,
            model=model,
            iteration=iteration,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            latency_ms=latency_ms,
            sequence_number=self._operation_sequence_counter,
            status='success'  # LLM calls that complete are successful
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **self._node_to_dict(node))
        
        # Set root if this is the first node
        if self.root_node_id is None:
            self.root_node_id = node_id
        
        # Add edge from parent if exists
        if parent_id is not None:
            self.graph.add_edge(parent_id, node_id)
        
        # Track current call ID
        self._current_call_ids[node_id] = node_id
        
        return node_id
    
    def create_code_execution_node(
        self,
        code: str,
        output: str,
        iteration: int,
        depth: int,
        parent_id: Optional[str] = None,
        error: Optional[str] = None
    ) -> str:
        """
        Create a new code execution node in the graph.
        
        Args:
            code: Executed code
            output: Execution output
            iteration: REPL iteration number
            depth: Recursion depth
            parent_id: Parent node ID (typically the LLM call that generated this code)
            error: Error message if execution failed
        
        Returns:
            Unique node ID
        """
        node_id = str(uuid.uuid4())
        
        # Increment sequence counter
        self._operation_sequence_counter += 1
        
        node = GraphNode(
            node_id=node_id,
            node_type='code_execution',
            depth=depth,
            parent_id=parent_id,
            code=code,
            output=output,
            iteration=iteration,
            error=error,
            sequence_number=self._operation_sequence_counter,
            status='error' if error else 'success'
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **self._node_to_dict(node))
        
        # Add edge from parent if exists
        if parent_id is not None:
            self.graph.add_edge(parent_id, node_id)
        
        return node_id
    
    def update_node(
        self,
        node_id: str,
        answer: str = "",
        iterations: Optional[int] = None,
        llm_calls: Optional[int] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Update node with results after execution.

        Args:
            node_id: Node to update
            answer: Final answer
            iterations: Number of iterations (None = don't update)
            llm_calls: Number of LLM calls (None = don't update)
            error: Error message if failed
        """
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        # Update fields with explicit None checks to handle 0 values
        if answer:
            node.answer = answer
        if iterations is not None:
            node.iterations = iterations
        if llm_calls is not None:
            node.llm_calls = llm_calls
        if error is not None:
            node.error = error
            node.status = "error"
        elif answer:
            # If we got an answer and no error, mark as success
            node.status = "success"

        # Update graph node attributes
        self.graph.nodes[node_id].update(self._node_to_dict(node))
    
    def add_repl_step(
        self,
        node_id: str,
        iteration: int,
        code: str,
        output: str
    ) -> None:
        """
        Add a REPL iteration step to a node.
        
        Args:
            node_id: Node ID
            iteration: Iteration number
            code: Generated code
            output: Execution output
        """
        if node_id not in self.nodes:
            return
        
        step = REPLStep(
            iteration=iteration,
            code=code,
            output=output
        )
        
        self.nodes[node_id].repl_steps.append(step)

        # Update graph node
        self.graph.nodes[node_id].update(self._node_to_dict(self.nodes[node_id]))

    def add_llm_call(
        self,
        node_id: str,
        call_id: str,
        iteration: int,
        prompt: str,
        response: str,
        model: str,
        tokens_prompt: Optional[int] = None,
        tokens_completion: Optional[int] = None,
        latency_ms: Optional[float] = None,
        timestamp: Optional[str] = None
    ) -> None:
        """
        Add an LLM API call to a node.

        Args:
            node_id: Node ID
            call_id: Unique call identifier
            iteration: REPL iteration number
            prompt: Full prompt text
            response: Full response text
            model: Model name
            tokens_prompt: Token count for prompt
            tokens_completion: Token count for completion
            latency_ms: Call latency in milliseconds
            timestamp: ISO timestamp
        """
        if node_id not in self.nodes:
            return

        # Assign global sequence number
        self._llm_call_sequence_counter += 1

        call = LLMCall(
            call_id=call_id,
            iteration=iteration,
            prompt=prompt,
            response=response,
            model=model,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            latency_ms=latency_ms,
            timestamp=timestamp,
            sequence_number=self._llm_call_sequence_counter
        )

        node = self.nodes[node_id]
        node.llm_calls_list.append(call)

        # Update aggregated metrics
        if call.total_tokens is not None:
            node.total_tokens += call.total_tokens
        if latency_ms is not None:
            node.total_latency_ms += latency_ms

        # Track current call ID for this node (for recursion tracking)
        self._current_call_ids[node_id] = call_id

        # Update graph node attributes
        self.graph.nodes[node_id].update(self._node_to_dict(node))

        # Return call for potential use in tracking recursion triggers
        return call

    def get_current_call_id(self, node_id: str) -> Optional[str]:
        """
        Get the current (most recent) LLM call ID for a node.

        Args:
            node_id: Node ID

        Returns:
            Call ID if available, None otherwise
        """
        return self._current_call_ids.get(node_id)

    def mark_call_triggered_recursion(
        self,
        node_id: str,
        call_id: str,
        spawned_node_id: str
    ) -> None:
        """
        Mark an LLM call as having triggered a recursive call.

        Args:
            node_id: Parent node ID
            call_id: LLM call ID that triggered recursion
            spawned_node_id: ID of the child node that was spawned
        """
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        # Find the call and update it
        for call in node.llm_calls_list:
            if call.call_id == call_id:
                call.triggered_recursion = True
                call.spawned_node_id = spawned_node_id
                break

        # Update graph node attributes
        self.graph.nodes[node_id].update(self._node_to_dict(node))

    def get_graph(self) -> nx.DiGraph:
        """
        Get the NetworkX graph.

        Returns:
            NetworkX DiGraph object
        """
        return self.graph
    
    def save_html(
        self,
        output_path: str = "./rlm_graph.html",
        height: str = "800px",
        width: str = "100%"
    ) -> None:
        """
        Save interactive HTML visualization.
        
        Args:
            output_path: Path to save HTML file
            height: Graph height (CSS format)
            width: Graph width (CSS format)
        """
        # Create pyvis network
        net = Network(
            height=height,
            width=width,
            directed=True,
            notebook=False
        )
        
        # Configure physics for hierarchical layout
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "hierarchicalRepulsion": {
              "centralGravity": 0.0,
              "springLength": 200,
              "springConstant": 0.01,
              "nodeDistance": 150,
              "damping": 0.09
            },
            "solver": "hierarchicalRepulsion"
          },
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "UD",
              "sortMethod": "directed",
              "levelSeparation": 150,
              "nodeSpacing": 200
            }
          },
          "nodes": {
            "shape": "box",
            "margin": 10,
            "widthConstraint": {
              "maximum": 300
            }
          },
          "edges": {
            "arrows": "to",
            "smooth": {
              "enabled": true,
              "type": "cubicBezier"
            }
          }
        }
        """)
        
        # Add nodes with visualization attributes
        for node_id, node_data in self.nodes.items():
            # Build label based on node type
            if node_data.node_type == 'llm_call':
                # LLM call node
                status_emoji = "❌" if node_data.status == "error" else "🤖"
                label = f"{status_emoji} LLM #{node_data.sequence_number or 0}"
                if node_data.depth > 0:
                    label += f" (D{node_data.depth})"
                
                # Add iteration info
                if node_data.iteration > 0:
                    label += f" [i{node_data.iteration}]"
                
                # Add prompt preview
                prompt_preview = node_data.get_prompt_preview(50)
                if prompt_preview:
                    label += f"\n{prompt_preview}"
                
                # Add token info if available
                if node_data.total_tokens_computed:
                    label += f"\n💬 {node_data.total_tokens_computed} tokens"
                
            elif node_data.node_type == 'code_execution':
                # Code execution node
                status_emoji = "❌" if node_data.status == "error" else "⚙️"
                label = f"{status_emoji} Exec #{node_data.sequence_number or 0}"
                if node_data.depth > 0:
                    label += f" (D{node_data.depth})"
                
                # Add iteration info
                if node_data.iteration > 0:
                    label += f" [i{node_data.iteration}]"
                
                # Add code preview
                code_preview = node_data.get_code_preview(50)
                if code_preview:
                    label += f"\n{code_preview}"
                
            else:
                # Legacy rlm_root node
                status_emoji = {
                    "success": "✅",
                    "error": "❌",
                    "pending": "⏳"
                }.get(node_data.status, "⏳")

                query_preview = node_data.get_query_preview(40)
                label = f"{status_emoji} D{node_data.depth}"

                # Add iteration/call info to label
                if node_data.iterations > 0:
                    label += f" ({node_data.iterations}i/{node_data.llm_calls}c)"

                # Add cumulative metrics if node has descendants
                cumulative = self.get_cumulative_stats(node_id)
                if cumulative['descendant_count'] > 0:
                    label += f"\n📊 +{cumulative['descendant_count']} descendants"
                    label += f" ({cumulative['cumulative_iterations']}i/{cumulative['cumulative_llm_calls']}c total)"

                if query_preview:
                    label += f"\n{query_preview}"

            # Color based on status and depth
            color = self._get_color_for_node(node_data)

            # Build detailed title (tooltip)
            title = self._build_node_tooltip(node_data)

            # Add node
            net.add_node(
                node_id,
                label=label,
                title=title,
                color=color,
                level=node_data.depth
            )
        
        # Add edges
        for edge in self.graph.edges():
            net.add_edge(edge[0], edge[1])
        
        # Save HTML
        net.save_graph(output_path)
        print(f"Graph visualization saved to: {output_path}")
    
    def _node_to_dict(self, node: GraphNode) -> Dict[str, Any]:
        """Convert node to dict for graph attributes."""
        return {
            'query': node.query,
            'context_preview': node.get_context_preview(),
            'answer': node.answer,
            'depth': node.depth,
            'iterations': node.iterations,
            'llm_calls': node.llm_calls,
            'llm_calls_list': node.llm_calls_list,
            'llm_calls_count': len(node.llm_calls_list),
            'total_tokens': node.total_tokens,
            'total_latency_ms': node.total_latency_ms,
            'repl_steps_count': len(node.repl_steps),
            'error': node.error,
            'status': node.status
        }
    
    def _get_color_for_node(self, node: GraphNode) -> str:
        """
        Get color based on node type, status, and depth.

        Color scheme:
        - LLM call nodes: Blue gradient (darker at root, lighter at higher depths)
        - Code execution nodes: Green gradient (darker at root, lighter at higher depths)
        - Error nodes: Red (overrides depth gradient)
        - Legacy nodes: Original color scheme
        
        Priority: Error status > Node type > Depth
        """
        # Error nodes are always red
        if node.status == "error":
            return "#dc2626"  # Red for errors
        
        # LLM call nodes: Blue gradient by depth
        if node.node_type == 'llm_call':
            llm_colors = [
                "#1e3a8a",  # Dark blue - depth 0
                "#3b82f6",  # Medium blue - depth 1
                "#60a5fa",  # Light blue - depth 2
                "#93c5fd",  # Very light blue - depth 3+
            ]
            return llm_colors[min(node.depth, len(llm_colors) - 1)]
        
        # Code execution nodes: Green gradient by depth
        if node.node_type == 'code_execution':
            exec_colors = [
                "#166534",  # Dark green - depth 0
                "#22c55e",  # Medium green - depth 1
                "#4ade80",  # Light green - depth 2
                "#86efac",  # Very light green - depth 3+
            ]
            return exec_colors[min(node.depth, len(exec_colors) - 1)]
        
        # Legacy rlm_root nodes: Original color scheme
        if node.status == "success":
            if node.depth == 0:
                return "#27ae60"  # Dark green for root success
            else:
                return "#2ecc71"  # Light green for child success

        # Pending nodes use depth-based colors
        depth_colors = [
            "#3498db",  # Blue - depth 0
            "#5dade2",  # Light blue - depth 1
            "#f39c12",  # Orange - depth 2
            "#e67e22",  # Dark orange - depth 3
            "#9b59b6",  # Purple - depth 4
            "#1abc9c",  # Turquoise - depth 5+
        ]
        return depth_colors[min(node.depth, len(depth_colors) - 1)]
    
    def _build_node_tooltip(self, node: GraphNode) -> str:
        """Build plain text tooltip for node."""
        lines = []
        lines.append(f"Node ID: {node.node_id[:8]}...")
        lines.append(f"Type: {node.node_type}")
        lines.append(f"Depth: {node.depth}")
        lines.append(f"Status: {node.status}")
        
        if node.sequence_number:
            lines.append(f"Sequence: #{node.sequence_number}")
        
        if node.parent_id:
            lines.append(f"Parent: {node.parent_id[:8]}...")
        
        # LLM call node details
        if node.node_type == 'llm_call':
            lines.append("")
            lines.append("LLM Call Details:")
            lines.append(f"  Iteration: {node.iteration}")
            
            if node.model:
                lines.append(f"  Model: {node.model}")
            
            if node.prompt:
                prompt_display = node.prompt[:300] + "..." if len(node.prompt) > 300 else node.prompt
                lines.append("")
                lines.append("  Prompt:")
                lines.append(f"    {prompt_display}")
            
            if node.response:
                response_display = node.response[:300] + "..." if len(node.response) > 300 else node.response
                lines.append("")
                lines.append("  Response:")
                lines.append(f"    {response_display}")
            
            # Token and latency info
            if node.tokens_prompt is not None:
                lines.append("")
                lines.append(f"  Tokens (prompt): {node.tokens_prompt}")
            if node.tokens_completion is not None:
                lines.append(f"  Tokens (completion): {node.tokens_completion}")
            if node.total_tokens_computed:
                lines.append(f"  Total Tokens: {node.total_tokens_computed}")
            if node.latency_ms is not None:
                lines.append(f"  Latency: {node.latency_ms:.1f}ms")
        
        # Code execution node details
        elif node.node_type == 'code_execution':
            lines.append("")
            lines.append("Code Execution Details:")
            lines.append(f"  Iteration: {node.iteration}")
            
            if node.code:
                code_display = node.code[:300] + "..." if len(node.code) > 300 else node.code
                lines.append("")
                lines.append("  Code:")
                lines.append(f"    {code_display}")
            
            if node.output:
                output_display = node.output[:300] + "..." if len(node.output) > 300 else node.output
                lines.append("")
                lines.append("  Output:")
                lines.append(f"    {output_display}")
        
        # Legacy rlm_root node details
        else:
            # Recursion trigger information
            if node.triggering_llm_call_id:
                lines.append(f"Spawned by LLM call: {node.triggering_llm_call_id[:8]}...")

            if node.query:
                query_display = node.query[:200] + "..." if len(node.query) > 200 else node.query
                lines.append(f"Query: {query_display}")

            if node.context:
                context_display = node.context[:200] + "..." if len(node.context) > 200 else node.context
                lines.append(f"Context: {context_display}")

            if node.answer:
                answer_display = node.answer[:200] + "..." if len(node.answer) > 200 else node.answer
                lines.append(f"Answer: {answer_display}")

            lines.append(f"Iterations: {node.iterations}")
            lines.append(f"LLM Calls: {node.llm_calls}")

            # Token and latency info
            if node.total_tokens > 0:
                lines.append(f"Total Tokens: {node.total_tokens}")
            if node.total_latency_ms > 0:
                lines.append(f"Total Latency: {node.total_latency_ms:.1f}ms")

            # Cumulative metrics for subtree
            cumulative = self.get_cumulative_stats(node.node_id)
            if cumulative['descendant_count'] > 0:
                lines.append("")
                lines.append("Subtree Metrics:")
                lines.append(f"  Descendants: {cumulative['descendant_count']}")
                lines.append(f"  Total Iterations: {cumulative['cumulative_iterations']}")
                lines.append(f"  Total LLM Calls: {cumulative['cumulative_llm_calls']}")

            # LLM call details showing recursion triggers
            if node.llm_calls_list:
                lines.append("")
                lines.append(f"LLM Calls ({len(node.llm_calls_list)}):")
                for i, call in enumerate(node.llm_calls_list[:3], 1):
                    call_info = f"  #{call.sequence_number if call.sequence_number else i}"
                    if call.triggered_recursion:
                        call_info += f" SPAWNED -> {call.spawned_node_id[:8] if call.spawned_node_id else 'unknown'}..."
                    if call.total_tokens:
                        call_info += f" ({call.total_tokens} tokens"
                        if call.latency_ms:
                            call_info += f", {call.latency_ms:.0f}ms"
                        call_info += ")"
                    lines.append(call_info)
                if len(node.llm_calls_list) > 3:
                    lines.append(f"  ... and {len(node.llm_calls_list) - 3} more calls")

            if node.repl_steps:
                lines.append("")
                lines.append(f"REPL Steps ({len(node.repl_steps)}):")
                for step in node.repl_steps[:2]:  # Show first 2 steps
                    code_preview = step.code[:80] + "..." if len(step.code) > 80 else step.code
                    output_preview = step.output[:80] + "..." if len(step.output) > 80 else step.output
                    lines.append(f"  Iter {step.iteration}:")
                    lines.append(f"    Code: {code_preview}")
                    lines.append(f"    Output: {output_preview}")
                if len(node.repl_steps) > 2:
                    lines.append(f"  ... and {len(node.repl_steps) - 2} more steps")

        if node.error:
            lines.append("")
            lines.append(f"Error: {node.error}")

        return "\n".join(lines)
    
    def get_cumulative_stats(self, node_id: str) -> Dict[str, Any]:
        """
        Get cumulative statistics for a node and all its descendants.

        Args:
            node_id: Node ID to calculate stats for

        Returns:
            Dict with cumulative metrics including:
            - cumulative_iterations: Total iterations for node + descendants
            - cumulative_llm_calls: Total LLM calls for node + descendants
            - descendant_count: Number of descendants
        """
        if node_id not in self.nodes:
            return {
                'cumulative_iterations': 0,
                'cumulative_llm_calls': 0,
                'descendant_count': 0
            }

        # Get all descendants using NetworkX
        descendants = nx.descendants(self.graph, node_id)

        # Include the node itself in calculations
        all_nodes = [node_id] + list(descendants)

        # Sum up metrics
        cumulative_iterations = sum(
            self.nodes[nid].iterations for nid in all_nodes if nid in self.nodes
        )
        cumulative_llm_calls = sum(
            self.nodes[nid].llm_calls for nid in all_nodes if nid in self.nodes
        )

        return {
            'cumulative_iterations': cumulative_iterations,
            'cumulative_llm_calls': cumulative_llm_calls,
            'descendant_count': len(descendants)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        stats = {
            'total_nodes': len(self.nodes),
            'max_depth': max((n.depth for n in self.nodes.values()), default=0),
            'total_iterations': sum(n.iterations for n in self.nodes.values()),
            'total_llm_calls': sum(n.llm_calls for n in self.nodes.values()),
            'nodes_with_errors': sum(1 for n in self.nodes.values() if n.error),
            'success_nodes': sum(1 for n in self.nodes.values() if n.status == 'success'),
            'pending_nodes': sum(1 for n in self.nodes.values() if n.status == 'pending'),
        }

        # Add root cumulative stats if root exists
        if self.root_node_id:
            root_cumulative = self.get_cumulative_stats(self.root_node_id)
            stats['root_cumulative_iterations'] = root_cumulative['cumulative_iterations']
            stats['root_cumulative_llm_calls'] = root_cumulative['cumulative_llm_calls']

        return stats

