"""
Example: NetworkX Graph Tracking for Recursive LLM Calls

This example demonstrates how to use NetworkX graph tracking to visualize
recursive LLM calls, including:
- Hierarchical call structure
- Input/output for each node
- REPL iteration details
- Interactive HTML visualization
"""

import os
from pathlib import Path
from rlm import RLM

# Sample document with sections for recursive processing
def generate_long_document():
    """Generate a long document for testing."""
    chapters = []

    for i in range(1, 51):  # 30 chapters
        chapter = f"""
Chapter {i}: Topic {i}

This chapter discusses important concept {i} in great detail. The key findings include:

1. First major point about topic {i}
   - Supporting detail A
   - Supporting detail B
   - Supporting detail C

2. Second major point about topic {i}
   - Evidence from study X
   - Evidence from study Y
   - Conclusion based on evidence

3. Third major point about topic {i}
   - Historical context
   - Current applications
   - Future implications

Key Statistics:
- Metric A: {i * 10}%
- Metric B: {i * 100} units
- Metric C: ${i * 1000}

Important dates:
- Event 1: January {i}, 2024
- Event 2: February {i}, 2024
- Event 3: March {i}, 2024

Conclusion:
Topic {i} represents a critical area of research with significant implications
for the field. Further investigation is warranted.

References:
[1] Author {i}. "Study on Topic {i}". Journal of Research. 2024.
[2] Researcher {i}. "Analysis of Topic {i}". Scientific Papers. 2024.

""" + "Additional context paragraph. " * 100  # Make each chapter longer
        chapters.append(chapter)

    return "\n\n".join(chapters)

LONG_DOCUMENT = generate_long_document()

query = "How many total references are cited in the document?.",

def example_basic_tracking():
    """Basic example with graph tracking enabled."""
    print("=" * 70)
    print("Example 1: Basic Graph Tracking")
    print("=" * 70)
    
    # Initialize RLM with graph tracking enabled
    rlm = RLM(
        model="openrouter/anthropic/claude-sonnet-4.5",
        recursive_model="openrouter/anthropic/claude-haiku-4.5",
        enable_graph_tracking=True,
        max_depth=2,
        max_iterations=10  # Sufficient for simple queries
    )
    
    # Simple query
    result = rlm.completion(
        query="What are the main topics discussed?",
        context=LONG_DOCUMENT[:2000]
    )
    
    print(f"\nResult: {result}")
    print(f"\nStats: {rlm.stats}")
    print(f"\nGraph saved to: basic_graph.html")
    print(f"Open the HTML file in a browser to view the interactive visualization!")


def example_recursive_tracking():
    """Example with recursive calls and graph tracking."""
    print("\n" + "=" * 70)
    print("Example 2: Recursive Processing with Graph Tracking")
    print("=" * 70)
    
    # Initialize RLM with graph tracking
    rlm = RLM(
        model="openrouter/anthropic/claude-sonnet-4.5",
        recursive_model="openrouter/anthropic/claude-haiku-4.5",
        enable_graph_tracking=True,
        graph_output_path="./recursive_graph.html",
        max_depth=3,
        max_iterations=10,  # Increased from 5 to allow sufficient iterations
        max_tokens=16000,
        temperature=0.3
    )
    
    # Query that requires recursive processing
    result = rlm.completion(
        query=query,
        context=LONG_DOCUMENT
    )
    
    print(f"\nResult: {result}")
    print(f"\nStats: {rlm.stats}")
    
    # Get the NetworkX graph object
    graph = rlm.get_graph()
    if graph:
        print(f"\nGraph nodes: {graph.number_of_nodes()}")
        print(f"Graph edges: {graph.number_of_edges()}")
        
        # Print node information
        print("\nNode Details:")
        for node_id, node_data in graph.nodes(data=True):
            print(f"  Node {node_id[:8]}... - Depth: {node_data.get('depth', 0)}, "
                  f"Iterations: {node_data.get('iterations', 0)}")
    
    print(f"\nGraph saved to: recursive_graph.html")


def example_with_different_output():
    """Example saving graph to custom location."""
    print("\n" + "=" * 70)
    print("Example 3: Custom Output Path")
    print("=" * 70)
    
    output_dir = Path("./graph_outputs")
    output_dir.mkdir(exist_ok=True)
    
    rlm = RLM(
        model="openrouter/anthropic/claude-sonnet-4.5",
        recursive_model="openrouter/anthropic/claude-haiku-4.5",
        enable_graph_tracking=True,
        graph_output_path=str(output_dir / "custom_graph.html"),
        max_depth=2
    )
    
    result = rlm.completion(
        query="Extract key insights from the document",
        context=LONG_DOCUMENT
    )
    
    print(f"\nResult: {result}")
    print(f"\nGraph saved to: {output_dir / 'custom_graph.html'}")
    
    # Can also save to different location after processing
    rlm.save_graph(str(output_dir / "alternative_location.html"))
    print(f"Also saved to: {output_dir / 'alternative_location.html'}")


def example_without_tracking():
    """Example without graph tracking (for comparison)."""
    print("\n" + "=" * 70)
    print("Example 4: Without Graph Tracking (Faster)")
    print("=" * 70)
    
    rlm = RLM(
        model="openrouter/anthropic/claude-sonnet-4.5",
        recursive_model="openrouter/anthropic/claude-haiku-4.5",
        enable_graph_tracking=False,  # Disabled
        max_depth=2
    )
    
    result = rlm.completion(
        query="What are the types of machine learning?",
        context=LONG_DOCUMENT
    )
    
    print(f"\nResult: {result}")
    print(f"\nStats: {rlm.stats}")
    print("\nNo graph generated (tracking disabled)")


def example_with_errors():
    """Example showing how errors are tracked in the graph."""
    print("\n" + "=" * 70)
    print("Example 5: Error Tracking in Graph")
    print("=" * 70)
    
    rlm = RLM(
        model="gpt-4o-mini",
        enable_graph_tracking=True,
        graph_output_path="./error_graph.html",
        max_depth=2,
        max_iterations=3  # Very low to potentially cause MaxIterationsError
    )
    
    try:
        result = rlm.completion(
            query="Provide a very detailed analysis of each section",
            context=LONG_DOCUMENT
        )
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"\nError occurred (as expected): {type(e).__name__}")
        print(f"Error message: {str(e)}")
    
    print(f"\nGraph saved to: error_graph.html")
    print("The graph will show which nodes encountered errors (in red)")


def example_graph_analysis():
    """Example of analyzing the graph programmatically."""
    print("\n" + "=" * 70)
    print("Example 6: Programmatic Graph Analysis")
    print("=" * 70)
    
    rlm = RLM(
        model="openrouter/anthropic/claude-sonnet-4.5",
        recursive_model="openrouter/anthropic/claude-haiku-4.5",
        enable_graph_tracking=True,
        graph_output_path="./analysis_graph.html",
        max_depth=3,
        max_tokens=1024,
        temperature=0.3
    )
    
    result = rlm.completion(
        query="Analyze the document structure and content",
        context=LONG_DOCUMENT
    )
    
    # Get graph for analysis
    import networkx as nx
    graph = rlm.get_graph()
    
    if graph:
        print("\n--- Graph Analysis ---")
        print(f"Total nodes: {graph.number_of_nodes()}")
        print(f"Total edges: {graph.number_of_edges()}")
        print(f"Is tree: {nx.is_tree(graph)}")
        print(f"Is DAG: {nx.is_directed_acyclic_graph(graph)}")
        
        # Find root node (node with no predecessors)
        root_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        print(f"\nRoot nodes: {len(root_nodes)}")
        
        # Calculate max depth
        if root_nodes:
            depths = nx.single_source_shortest_path_length(graph, root_nodes[0])
            max_depth = max(depths.values()) if depths else 0
            print(f"Max depth reached: {max_depth}")
        
        # Statistics by depth
        print("\n--- Stats by Depth ---")
        depth_stats = {}
        for node_id, node_data in graph.nodes(data=True):
            depth = node_data.get('depth', 0)
            if depth not in depth_stats:
                depth_stats[depth] = {'count': 0, 'total_iterations': 0, 'total_llm_calls': 0}
            depth_stats[depth]['count'] += 1
            depth_stats[depth]['total_iterations'] += node_data.get('iterations', 0)
            depth_stats[depth]['total_llm_calls'] += node_data.get('llm_calls', 0)
        
        for depth in sorted(depth_stats.keys()):
            stats = depth_stats[depth]
            print(f"Depth {depth}: {stats['count']} nodes, "
                  f"{stats['total_iterations']} iterations, "
                  f"{stats['total_llm_calls']} LLM calls")
    
    print(f"\nResult: {result[:200]}...")
    print(f"\nGraph saved to: analysis_graph.html")


def example_with_history_tracking():
    """Example combining graph tracking with LLM call history."""
    print("\n" + "=" * 70)
    print("Example 7: Graph Tracking + LLM History Debugging")
    print("=" * 70)
    
    # Initialize RLM with both graph tracking and history tracking
    rlm = RLM(
        model="openrouter/anthropic/claude-sonnet-4.5",
        recursive_model="openrouter/anthropic/claude-haiku-4.5",
        enable_graph_tracking=True,
        enable_history=False,  # Enable LLM call history,
        enable_logging=True,
        graph_output_path="./debug_graph.html",
        history_output_path="./logs/debug_history.json",  # Auto-save history
        max_depth=3,
        max_iterations=15,
        max_tokens=2048,
        temperature=0.2,
        truncate_code=False
    )
    
    # Simple query for debugging
    result = rlm.completion(
        query=query,
        context=LONG_DOCUMENT  # Use smaller context for debugging
    )
    
    print(f"\nResult: {result}")
    print(f"\nStats: {rlm.stats}")
    
    # Show history summary
    print("\n" + "=" * 70)
    print("LLM Call History (for debugging):")
    print("=" * 70)
    rlm.print_history(detailed=False)
    
    # Get raw history for programmatic access
    history = rlm.get_history()
    print(f"\nTotal LLM calls: {len(history)}")
    
    print("\n💡 Tips:")
    print("  - Use enable_history=True to debug prompts and responses")
    print("  - Use history_output_path to auto-save history as JSON")
    print("  - Use rlm.print_history(detailed=True) for full details")
    print("  - Use rlm.save_history() to manually export to JSON")
    print("  - Combine with graph tracking to visualize call flow")

    # save the history to a file
    rlm.save_history("./logs/debug_history.json")





def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("NetworkX Graph Tracking Examples")
    print("=" * 70)
    print("\nThese examples demonstrate recursive LLM call visualization.")
    print("Interactive HTML graphs will be generated that you can open in a browser.")
    print("\nNote: Set OPENAI_API_KEY or configure your API provider before running.")
    print("=" * 70)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("Or use a different provider (see examples/openrouter_usage.py)")
        return
    
    # Run examples
    try:
        # example_basic_tracking()
        # example_recursive_tracking()
        # example_with_different_output()
        # example_without_tracking()
        # example_with_errors()  # Uncomment to test error tracking
        # example_graph_analysis()
        example_with_history_tracking()  # Uncomment to test history + graph tracking
        
        print("\n" + "=" * 70)
        print("All examples completed!")
        print("=" * 70)
        print("\nGenerated graph files:")
        print("- basic_graph.html")
        print("- recursive_graph.html")
        print("- graph_outputs/custom_graph.html")
        print("- graph_outputs/alternative_location.html")
        print("- analysis_graph.html")
        print("\nOpen any of these HTML files in your web browser to explore")
        print("the interactive visualization of recursive LLM calls!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

