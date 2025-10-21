"""Example: Debug LLM calls with history tracking."""

import os
from dotenv import load_dotenv
from rlm import RLM

# Load environment variables
load_dotenv()


def main():
    """Demo history tracking for debugging."""
    print("=" * 70)
    print("LLM Call History Tracking - Debug Example")
    print("=" * 70)
    
    # Sample document
    document = """
    Python Programming Basics
    
    Variables and Data Types:
    - Strings: "hello"
    - Integers: 42
    - Floats: 3.14
    - Booleans: True, False
    
    Control Flow:
    - if/elif/else statements
    - for loops
    - while loops
    
    Functions:
    - def function_name(params):
    - return values
    - lambda functions
    """
    
    # Initialize RLM with history tracking
    rlm = RLM(
        model="openrouter/anthropic/claude-sonnet-4.5",
        recursive_model="openrouter/anthropic/claude-haiku-4.5",
        enable_history=True,  # 🔍 Enable history tracking
        history_output_path="./logs/history.json",  # Auto-save to JSON
        max_iterations=10,
        temperature=0.5
    )
    
    query = "What data types are mentioned in the document?"
    
    print(f"\nQuery: {query}")
    print("Processing with history tracking enabled...\n")
    
    try:
        # Make the call
        result = rlm.completion(query, document)
        
        print("=" * 70)
        print("Result:")
        print("=" * 70)
        print(result)
        
        # Show statistics
        print("\n" + "=" * 70)
        print("Execution Statistics:")
        print("=" * 70)
        print(f"  LLM Calls: {rlm.stats['llm_calls']}")
        print(f"  Iterations: {rlm.stats['iterations']}")
        
        # Show history summary
        print("\n" + "=" * 70)
        print("LLM Call History (Summary):")
        print("=" * 70)
        rlm.print_history(detailed=False)
        
        # Access raw history
        history = rlm.get_history()
        print(f"\n📊 Total tracked calls: {len(history)}")
        
        print("\n" + "=" * 70)
        print("💡 Usage Tips:")
        print("=" * 70)
        print("• Use rlm.print_history(detailed=True) for full prompts/responses")
        print("• Use rlm.get_history() to access raw history data")
        print("• Use rlm.clear_history() to clear history between runs")
        print("• Use rlm.save_history('path.json') to manually save history")
        print("• Use history_output_path parameter to auto-save on completion")
        print("• Combine with enable_graph_tracking=True for visual debugging")
        print("\n📁 History was automatically saved to: ./logs/history.json")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def demo_detailed_history():
    """Demo detailed history output."""
    print("\n\n" + "=" * 70)
    print("Detailed History Example")
    print("=" * 70)
    
    rlm = RLM(
        model="gpt-4o-mini",
        enable_history=True,
        max_iterations=5
    )
    
    # Simple query
    result = rlm.completion(
        query="Count the numbers",
        context="Numbers: 1, 2, 3, 4, 5"
    )
    
    print(f"\nResult: {result}")
    
    # Print detailed history
    print("\n" + "=" * 70)
    print("Detailed LLM Call History:")
    print("=" * 70)
    rlm.print_history(detailed=True)


def demo_programmatic_access():
    """Demo programmatic access to history and JSON export."""
    print("\n\n" + "=" * 70)
    print("Programmatic History Access & JSON Export")
    print("=" * 70)
    
    rlm = RLM(
        model="gpt-4o-mini",
        enable_history=True,
        max_iterations=5
    )
    
    result = rlm.completion(
        query="What is 2+2?",
        context="Math problem"
    )
    
    print(f"\nResult: {result}")
    
    # Access history programmatically
    history = rlm.get_history()
    
    print(f"\n📊 History Analysis:")
    print(f"   Total calls: {len(history)}")
    
    # Process each entry
    for i, entry in enumerate(history, 1):
        print(f"\n   Call #{i}:")
        if isinstance(entry, dict):
            print(f"     Keys available: {list(entry.keys())}")
            # You can access specific fields based on your needs
            # e.g., entry['prompt'], entry['response'], etc.
    
    # Save history to JSON manually
    print("\n" + "─" * 70)
    print("Saving history to JSON files...")
    print("─" * 70)
    
    # Pretty formatted JSON
    rlm.save_history("./logs/history_pretty.json", pretty=True)
    
    # Compact JSON
    rlm.save_history("./logs/history_compact.json", pretty=False)
    
    print("\n💡 You can analyze tokens, costs, patterns, etc. from JSON files!")


if __name__ == "__main__":
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found!")
        print()
        print("Please set up your API key:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your OpenAI API key to .env")
        print("  3. Or run: export OPENAI_API_KEY='your-key'")
        exit(1)
    
    # Run examples
    main()
    
    # Uncomment to run additional demos:
    # demo_detailed_history()
    # demo_programmatic_access()

