"""Tests for NetworkX graph tracking functionality."""

import os
import tempfile
import pytest
import networkx as nx

from rlm import RLM
from rlm.graph import RLMGraphTracker, GraphNode, REPLStep, LLMCall


class TestGraphNode:
    """Test GraphNode dataclass."""
    
    def test_create_node(self):
        """Test creating a GraphNode."""
        node = GraphNode(
            node_id="test-123",
            query="What is this?",
            context="Some context",
            depth=0
        )

        assert node.node_id == "test-123"
        assert node.query == "What is this?"
        assert node.context == "Some context"
        assert node.depth == 0
        assert node.iterations == 0
        assert node.llm_calls == 0
        assert len(node.repl_steps) == 0
        assert node.status == "pending"  # New: default status
    
    def test_context_preview(self):
        """Test context preview truncation."""
        long_context = "x" * 200
        node = GraphNode(
            node_id="test-123",
            context=long_context,
            depth=0
        )
        
        preview = node.get_context_preview(max_len=50)
        assert len(preview) <= 53  # 50 + "..."
        assert preview.endswith("...")
    
    def test_query_preview(self):
        """Test query preview truncation."""
        long_query = "What is " + "x" * 100
        node = GraphNode(
            node_id="test-123",
            query=long_query,
            depth=0
        )
        
        preview = node.get_query_preview(max_len=20)
        assert len(preview) <= 23  # 20 + "..."
        assert preview.startswith("What is")
    
    def test_llm_call_node(self):
        """Test creating an LLM call node."""
        node = GraphNode(
            node_id="llm-123",
            node_type="llm_call",
            depth=0,
            prompt="What is 2+2?",
            response="4",
            model="gpt-4o-mini",
            iteration=1,
            tokens_prompt=5,
            tokens_completion=1,
            latency_ms=100.0
        )
        
        assert node.node_type == "llm_call"
        assert node.prompt == "What is 2+2?"
        assert node.response == "4"
        assert node.model == "gpt-4o-mini"
        assert node.iteration == 1
        assert node.total_tokens_computed == 6
        assert node.latency_ms == 100.0
    
    def test_code_execution_node(self):
        """Test creating a code execution node."""
        node = GraphNode(
            node_id="exec-123",
            node_type="code_execution",
            depth=0,
            code="result = 2 + 2",
            output="4",
            iteration=1
        )
        
        assert node.node_type == "code_execution"
        assert node.code == "result = 2 + 2"
        assert node.output == "4"
        assert node.iteration == 1
    
    def test_node_preview_methods(self):
        """Test all preview methods for different node types."""
        # LLM node previews
        llm_node = GraphNode(
            node_id="llm-123",
            node_type="llm_call",
            prompt="A" * 200,
            response="B" * 200,
            depth=0
        )
        
        assert len(llm_node.get_prompt_preview(50)) <= 53
        assert len(llm_node.get_response_preview(50)) <= 53
        
        # Code execution node previews
        code_node = GraphNode(
            node_id="exec-123",
            node_type="code_execution",
            code="x = " + "y" * 200,
            output="result: " + "z" * 200,
            depth=0
        )
        
        assert len(code_node.get_code_preview(50)) <= 53
        assert len(code_node.get_output_preview(50)) <= 53


class TestREPLStep:
    """Test REPLStep dataclass."""

    def test_create_repl_step(self):
        """Test creating a REPLStep."""
        step = REPLStep(
            iteration=1,
            code="x = 5",
            output="None"
        )

        assert step.iteration == 1
        assert step.code == "x = 5"
        assert step.output == "None"


class TestLLMCall:
    """Test LLMCall dataclass."""

    def test_llm_call_dataclass_creation(self):
        """Test LLMCall dataclass stores call metadata."""
        call = LLMCall(
            call_id="call-1",
            iteration=1,
            prompt="Test prompt",
            response="Test response",
            model="gpt-4o-mini",
            tokens_prompt=10,
            tokens_completion=20,
            latency_ms=150.5
        )

        assert call.call_id == "call-1"
        assert call.iteration == 1
        assert call.model == "gpt-4o-mini"
        assert call.tokens_prompt == 10
        assert call.tokens_completion == 20
        assert call.total_tokens == 30
        assert call.latency_ms == 150.5

    def test_llm_call_preview_truncation(self):
        """Test LLMCall provides truncated previews."""
        call = LLMCall(
            call_id="call-1",
            iteration=1,
            prompt="A" * 200,
            response="B" * 200,
            model="gpt-4o-mini"
        )

        preview_prompt = call.get_prompt_preview(max_len=50)
        preview_response = call.get_response_preview(max_len=50)

        assert len(preview_prompt) == 53  # 50 + "..."
        assert preview_prompt.endswith("...")
        assert len(preview_response) == 53
        assert preview_response.endswith("...")

    def test_graph_node_stores_llm_calls(self):
        """Test GraphNode can store multiple LLM calls."""
        node = GraphNode(
            node_id="node-1",
            query="test query",
            context="test context"
        )

        call1 = LLMCall(
            call_id="call-1",
            iteration=1,
            prompt="prompt1",
            response="response1",
            model="gpt-4o-mini",
            tokens_prompt=10,
            tokens_completion=20
        )

        call2 = LLMCall(
            call_id="call-2",
            iteration=2,
            prompt="prompt2",
            response="response2",
            model="gpt-4o-mini",
            tokens_prompt=15,
            tokens_completion=25
        )

        node.llm_calls_list = [call1, call2]
        node.total_tokens = 70

        assert len(node.llm_calls_list) == 2
        assert node.total_tokens == 70
        assert node.llm_calls_list[0].call_id == "call-1"


class TestRLMGraphTracker:
    """Test RLMGraphTracker."""
    
    def test_create_tracker(self):
        """Test creating a graph tracker."""
        tracker = RLMGraphTracker()
        
        assert tracker.graph is not None
        assert isinstance(tracker.graph, nx.DiGraph)
        assert len(tracker.nodes) == 0
        assert tracker.root_node_id is None
    
    def test_create_node(self):
        """Test creating a node in the graph."""
        tracker = RLMGraphTracker()
        
        node_id = tracker.create_node(
            query="Test query",
            context="Test context",
            depth=0
        )
        
        assert node_id is not None
        assert node_id in tracker.nodes
        assert tracker.root_node_id == node_id
        assert tracker.graph.number_of_nodes() == 1
    
    def test_create_child_node(self):
        """Test creating a child node with parent."""
        tracker = RLMGraphTracker()
        
        parent_id = tracker.create_node(
            query="Parent query",
            context="Parent context",
            depth=0
        )
        
        child_id = tracker.create_node(
            query="Child query",
            context="Child context",
            depth=1,
            parent_id=parent_id
        )
        
        assert child_id in tracker.nodes
        assert tracker.nodes[child_id].parent_id == parent_id
        assert tracker.graph.number_of_edges() == 1
        assert tracker.graph.has_edge(parent_id, child_id)
    
    def test_create_llm_call_node(self):
        """Test creating an LLM call node."""
        tracker = RLMGraphTracker()
        
        llm_node_id = tracker.create_llm_call_node(
            prompt="What is 2+2?",
            response="4",
            model="gpt-4o-mini",
            depth=0,
            iteration=1,
            tokens_prompt=5,
            tokens_completion=1,
            latency_ms=100.0
        )
        
        assert llm_node_id is not None
        assert llm_node_id in tracker.nodes
        node = tracker.nodes[llm_node_id]
        assert node.node_type == "llm_call"
        assert node.prompt == "What is 2+2?"
        assert node.response == "4"
        assert node.model == "gpt-4o-mini"
        assert node.tokens_prompt == 5
        assert node.tokens_completion == 1
        assert node.latency_ms == 100.0
        assert node.status == "success"
    
    def test_create_code_execution_node(self):
        """Test creating a code execution node."""
        tracker = RLMGraphTracker()
        
        exec_node_id = tracker.create_code_execution_node(
            code="result = 2 + 2",
            output="4",
            iteration=1,
            depth=0
        )
        
        assert exec_node_id is not None
        assert exec_node_id in tracker.nodes
        node = tracker.nodes[exec_node_id]
        assert node.node_type == "code_execution"
        assert node.code == "result = 2 + 2"
        assert node.output == "4"
        assert node.iteration == 1
        assert node.status == "success"
    
    def test_fine_grained_tracking_chain(self):
        """Test creating a chain of LLM → Code → LLM → Code nodes."""
        tracker = RLMGraphTracker()
        
        # First LLM call (root)
        llm1_id = tracker.create_llm_call_node(
            prompt="Calculate 2+2",
            response="result = 2 + 2",
            model="gpt-4o-mini",
            depth=0,
            iteration=1
        )
        
        # First code execution (child of LLM1)
        exec1_id = tracker.create_code_execution_node(
            code="result = 2 + 2",
            output="4",
            iteration=1,
            depth=0,
            parent_id=llm1_id
        )
        
        # Second LLM call (child of Exec1)
        llm2_id = tracker.create_llm_call_node(
            prompt="FINAL(result)",
            response="FINAL(4)",
            model="gpt-4o-mini",
            depth=0,
            iteration=2,
            parent_id=exec1_id
        )
        
        # Verify chain structure
        assert tracker.graph.number_of_nodes() == 3
        assert tracker.graph.number_of_edges() == 2
        assert tracker.graph.has_edge(llm1_id, exec1_id)
        assert tracker.graph.has_edge(exec1_id, llm2_id)
        
        # Verify node types
        assert tracker.nodes[llm1_id].node_type == "llm_call"
        assert tracker.nodes[exec1_id].node_type == "code_execution"
        assert tracker.nodes[llm2_id].node_type == "llm_call"
    
    def test_update_node(self):
        """Test updating a node with results."""
        tracker = RLMGraphTracker()

        node_id = tracker.create_node(
            query="Test",
            context="Context",
            depth=0
        )

        tracker.update_node(
            node_id=node_id,
            answer="Test answer",
            iterations=5,
            llm_calls=3
        )

        node = tracker.nodes[node_id]
        assert node.answer == "Test answer"
        assert node.iterations == 5
        assert node.llm_calls == 3
        assert node.status == "success"  # New: status should be success

    def test_update_node_with_zero_values(self):
        """Test that updating with 0 values works correctly (bug fix)."""
        tracker = RLMGraphTracker()

        node_id = tracker.create_node(
            query="Test",
            context="Context",
            depth=0
        )

        # Update with 0 values - this should work now
        tracker.update_node(
            node_id=node_id,
            answer="Quick answer",
            iterations=0,
            llm_calls=0
        )

        node = tracker.nodes[node_id]
        assert node.iterations == 0  # Should be 0, not unchanged
        assert node.llm_calls == 0  # Should be 0, not unchanged
        assert node.answer == "Quick answer"
        assert node.status == "success"

    def test_update_node_with_error(self):
        """Test updating a node with an error."""
        tracker = RLMGraphTracker()

        node_id = tracker.create_node(
            query="Test",
            context="Context",
            depth=0
        )

        tracker.update_node(
            node_id=node_id,
            error="Something went wrong"
        )

        node = tracker.nodes[node_id]
        assert node.error == "Something went wrong"
        assert node.status == "error"  # Should be marked as error
    
    def test_add_repl_step(self):
        """Test adding REPL steps to a node."""
        tracker = RLMGraphTracker()
        
        node_id = tracker.create_node(
            query="Test",
            context="Context",
            depth=0
        )
        
        tracker.add_repl_step(
            node_id=node_id,
            iteration=1,
            code="x = 5",
            output="None"
        )
        
        tracker.add_repl_step(
            node_id=node_id,
            iteration=2,
            code="FINAL(x)",
            output="5"
        )
        
        node = tracker.nodes[node_id]
        assert len(node.repl_steps) == 2
        assert node.repl_steps[0].iteration == 1
        assert node.repl_steps[1].code == "FINAL(x)"
    
    def test_get_graph(self):
        """Test getting the NetworkX graph."""
        tracker = RLMGraphTracker()
        
        tracker.create_node(query="Root", context="", depth=0)
        
        graph = tracker.get_graph()
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == 1
    
    def test_save_html(self):
        """Test saving graph to HTML."""
        tracker = RLMGraphTracker()
        
        tracker.create_node(
            query="Test query",
            context="Test context",
            depth=0
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            tracker.save_html(output_path=temp_path)
            assert os.path.exists(temp_path)
            
            # Check file has content
            with open(temp_path, 'r') as f:
                content = f.read()
                assert len(content) > 0
                assert "html" in content.lower()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_get_stats(self):
        """Test getting graph statistics."""
        tracker = RLMGraphTracker()

        parent_id = tracker.create_node(query="Parent", context="", depth=0)
        tracker.update_node(parent_id, answer="Answer", iterations=5, llm_calls=3)

        child_id = tracker.create_node(
            query="Child",
            context="",
            depth=1,
            parent_id=parent_id
        )
        tracker.update_node(child_id, answer="Answer2", iterations=3, llm_calls=2)

        stats = tracker.get_stats()

        assert stats['total_nodes'] == 2
        assert stats['max_depth'] == 1
        assert stats['total_iterations'] == 8  # 5 + 3
        assert stats['total_llm_calls'] == 5  # 3 + 2
        assert stats['nodes_with_errors'] == 0
        assert stats['success_nodes'] == 2  # Both marked as success
        assert stats['pending_nodes'] == 0
        # New: cumulative stats from root
        assert stats['root_cumulative_iterations'] == 8
        assert stats['root_cumulative_llm_calls'] == 5

    def test_get_cumulative_stats(self):
        """Test getting cumulative statistics for a subtree."""
        tracker = RLMGraphTracker()

        # Create a tree structure:
        #       root (5 iter, 3 calls)
        #      /    \
        #  child1   child2 (2 iter, 1 call)
        #  (3,2)       |
        #           grandchild (1,1)

        root_id = tracker.create_node(query="Root", context="", depth=0)
        tracker.update_node(root_id, answer="Root answer", iterations=5, llm_calls=3)

        child1_id = tracker.create_node(query="Child1", context="", depth=1, parent_id=root_id)
        tracker.update_node(child1_id, answer="Child1 answer", iterations=3, llm_calls=2)

        child2_id = tracker.create_node(query="Child2", context="", depth=1, parent_id=root_id)
        tracker.update_node(child2_id, answer="Child2 answer", iterations=2, llm_calls=1)

        grandchild_id = tracker.create_node(query="Grandchild", context="", depth=2, parent_id=child2_id)
        tracker.update_node(grandchild_id, answer="Grandchild answer", iterations=1, llm_calls=1)

        # Test cumulative stats for root (should include all descendants)
        root_stats = tracker.get_cumulative_stats(root_id)
        assert root_stats['cumulative_iterations'] == 11  # 5+3+2+1
        assert root_stats['cumulative_llm_calls'] == 7  # 3+2+1+1
        assert root_stats['descendant_count'] == 3  # child1, child2, grandchild

        # Test cumulative stats for child2 (should include grandchild)
        child2_stats = tracker.get_cumulative_stats(child2_id)
        assert child2_stats['cumulative_iterations'] == 3  # 2+1
        assert child2_stats['cumulative_llm_calls'] == 2  # 1+1
        assert child2_stats['descendant_count'] == 1  # grandchild only

        # Test cumulative stats for leaf node (no descendants)
        child1_stats = tracker.get_cumulative_stats(child1_id)
        assert child1_stats['cumulative_iterations'] == 3  # just itself
        assert child1_stats['cumulative_llm_calls'] == 2  # just itself
        assert child1_stats['descendant_count'] == 0  # no descendants


class TestRLMWithGraphTracking:
    """Test RLM with graph tracking enabled."""
    
    def test_graph_tracking_disabled_by_default(self):
        """Test that graph tracking is disabled by default."""
        rlm = RLM(model="gpt-4o-mini")
        
        assert rlm.graph_tracker is None
        assert rlm.get_graph() is None
    
    def test_enable_graph_tracking(self):
        """Test enabling graph tracking."""
        rlm = RLM(
            model="gpt-4o-mini",
            enable_graph_tracking=True
        )
        
        assert rlm.graph_tracker is not None
        assert isinstance(rlm.graph_tracker, RLMGraphTracker)
    
    def test_get_graph_method(self):
        """Test get_graph method."""
        rlm = RLM(
            model="gpt-4o-mini",
            enable_graph_tracking=True
        )
        
        graph = rlm.get_graph()
        assert isinstance(graph, nx.DiGraph)
    
    def test_save_graph_method(self):
        """Test save_graph method."""
        rlm = RLM(
            model="gpt-4o-mini",
            enable_graph_tracking=True
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create a dummy node so graph is not empty
            rlm.graph_tracker.create_node(query="Test", context="", depth=0)
            
            rlm.save_graph(temp_path)
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_graph_disabled(self, capsys):
        """Test save_graph when tracking is disabled."""
        rlm = RLM(model="gpt-4o-mini", enable_graph_tracking=False)
        
        rlm.save_graph("test.html")
        
        captured = capsys.readouterr()
        assert "not enabled" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

