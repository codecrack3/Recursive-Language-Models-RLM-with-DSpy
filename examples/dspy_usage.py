"""Example: Using RLM with DSPy backend."""

import os
from dotenv import load_dotenv
from rlm import RLM

# Load environment variables
load_dotenv()

# Sample long document
long_document = """
The History of Artificial Intelligence

Introduction
Artificial Intelligence (AI) has transformed from a theoretical concept to a practical reality
over the past several decades. This document explores key milestones in AI development.

The 1950s: The Birth of AI
In 1950, Alan Turing published "Computing Machinery and Intelligence," introducing the famous
Turing Test. The term "Artificial Intelligence" was coined in 1956 at the Dartmouth Conference
by John McCarthy, Marvin Minsky, and others.

The 1960s-1970s: Early Optimism
During this period, researchers developed early AI programs like ELIZA (1966) and expert systems.
However, limitations in computing power led to the first "AI Winter" in the 1970s.

The 1980s-1990s: Expert Systems and Neural Networks
Expert systems became commercially successful in the 1980s. The backpropagation algorithm
revitalized neural network research in 1986.

The 2000s-2010s: Machine Learning Revolution
The rise of big data and powerful GPUs enabled deep learning breakthroughs. In 2012,
AlexNet won the ImageNet competition, marking a turning point for deep learning.

The 2020s: Large Language Models
GPT-3 (2020) and ChatGPT (2022) demonstrated unprecedented language understanding capabilities.
These models have billions of parameters and are trained on vast amounts of text data.

Conclusion
AI continues to evolve rapidly, with applications in healthcare, transportation, education,
and countless other domains. The future promises even more exciting developments.
""" * 10  # Multiply to make it longer


def main():
    """Run DSPy RLM example."""
    print("=" * 70)
    print("RLM with DSPy Backend Example")
    print("=" * 70)

    # Initialize RLM (uses DSPy backend)
    # This will use DSPy for LLM orchestration and auto-detect E2B/RestrictedPython
    rlm = RLM(
        model="gpt-4o-mini",  # or "claude-sonnet-4", "ollama/llama3.2", etc.
        sandbox='auto',       # Auto-detect E2B or fall back to RestrictedPython
        max_iterations=15,
        temperature=0.7
    )

    # Ask a question about the document
    query = "What were the key milestones in AI development according to this document?"

    print(f"\nQuery: {query}")
    print(f"Context length: {len(long_document):,} characters")
    print(f"Backend: DSPy")
    print("\nProcessing with RLM...\n")

    try:
        # Process with RLM
        result = rlm.completion(query, long_document)

        print("=" * 70)
        print("Result:")
        print("=" * 70)
        print(result)
        print("\n" + "=" * 70)
        print("Stats:")
        print("=" * 70)
        print(f"  LLM calls: {rlm.stats['llm_calls']}")
        print(f"  Iterations: {rlm.stats['iterations']}")
        print(f"  Depth: {rlm.stats['depth']}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def demo_history_tracking():
    """Demo history tracking for debugging LLM calls."""
    print("\n" + "=" * 70)
    print("History Tracking Demo (Debug LLM Calls)")
    print("=" * 70)
    
    # Initialize RLM with history tracking enabled
    rlm = RLM(
        model="gpt-4o-mini",
        enable_history=True,  # Enable history tracking
        max_iterations=10,
        temperature=0.7
    )
    
    # Simple query
    query = "What year was AI coined as a term?"
    context = long_document[:1000]  # Use just first part
    
    print(f"\nQuery: {query}")
    print("Processing with history tracking enabled...\n")
    
    try:
        result = rlm.completion(query, context)
        
        print("=" * 70)
        print("Result:")
        print("=" * 70)
        print(result)
        
        # Print history in summary mode
        print("\n" + "=" * 70)
        print("LLM Call History (Summary):")
        print("=" * 70)
        rlm.print_history(detailed=False)
        
        # Get raw history for programmatic access
        history = rlm.get_history()
        print(f"\nTotal LLM calls tracked: {len(history)}")
        
        # Optionally print detailed history
        print("\nTo see detailed history with full prompts/responses:")
        print("  rlm.print_history(detailed=True)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def demo_e2b_sandbox():
    """Demo E2B sandbox with package installation."""
    print("\n" + "=" * 70)
    print("E2B Sandbox Demo (Advanced Features)")
    print("=" * 70)

    # This example shows E2B's advantage: ability to install packages
    context_with_dates = """
    Important dates in the project:
    - Project start: 2024-01-15
    - First milestone: 2024-03-20
    - Second milestone: 2024-06-10
    - Project deadline: 2024-12-31
    """

    query = "Calculate how many days between the first and second milestones"

    print(f"\nQuery: {query}")
    print("Context: [dates document]")
    print("\nNote: E2B sandbox can install packages like 'datetime' if needed")
    print("\nProcessing...\n")

    try:
        rlm = RLM(
            model="gpt-4o-mini",
            sandbox='e2b',  # Explicitly use E2B
            max_iterations=10
        )
        result = rlm.completion(query, context_with_dates)
        print(f"Result: {result}")
        print(f"Stats: {rlm.stats}")
    except Exception as e:
        print(f"Note: E2B requires API key. Error: {e}")
        print("Falling back to RestrictedPython...")
        rlm = RLM(
            model="gpt-4o-mini",
            sandbox='restricted',
            max_iterations=10
        )
        result = rlm.completion(query, context_with_dates)
        print(f"Result: {result}")


if __name__ == "__main__":
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found!")
        print()
        print("Please set up your API key:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your OpenAI API key to .env")
        print("  3. Or run: python setup_env.py")
        exit(1)

    # Run main example
    main()

    # Uncomment to run additional demos:
    # demo_history_tracking()
    # demo_e2b_sandbox()
