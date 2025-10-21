"""Example: Using RLM with E2B cloud sandbox."""

import os
from dotenv import load_dotenv
from rlm import RLM

# Load environment variables
load_dotenv()


def demo_e2b_basic():
    """Basic E2B sandbox usage."""
    print("=" * 70)
    print("E2B Sandbox - Basic Usage")
    print("=" * 70)

    context = """
    Sales data for Q1 2024:
    - January: $125,000
    - February: $143,500
    - March: $167,800

    Sales data for Q2 2024:
    - April: $189,200
    - May: $201,450
    - June: $215,600
    """

    query = "Calculate the total sales for Q1 and Q2, and find the percentage increase"

    print(f"\nQuery: {query}")
    print("Context: [sales data]")
    print("\nProcessing with E2B sandbox...\n")

    try:
        rlm = RLM(
            model="gpt-4o-mini",
            sandbox='e2b',  # Use E2B cloud sandbox
            max_iterations=15
        )

        result = rlm.completion(query, context)

        print("=" * 70)
        print("Result:")
        print("=" * 70)
        print(result)
        print("\n" + "=" * 70)
        print(f"Stats: {rlm.stats}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: E2B requires E2B_API_KEY environment variable.")
        print("Get your API key at: https://e2b.dev")


def demo_e2b_advanced():
    """Advanced E2B features: package installation."""
    print("\n" + "=" * 70)
    print("E2B Sandbox - Advanced Features")
    print("=" * 70)

    context = """
    Project timeline:
    Start: 2024-01-15
    Phase 1 complete: 2024-03-22
    Phase 2 complete: 2024-07-10
    Expected completion: 2024-11-30

    The project has 4 phases total.
    """

    query = """
    Calculate:
    1. Days taken for Phase 1
    2. Days taken for Phase 2
    3. Average days per phase
    4. Estimated completion date if remaining phases take the same average time
    """

    print(f"\nQuery: {query}")
    print("\nNote: E2B can install Python packages if needed (e.g., datetime, numpy)")
    print("\nProcessing...\n")

    try:
        rlm = RLM(
            model="gpt-4o-mini",
            sandbox='e2b',
            max_iterations=20  # More iterations for complex calculations
        )

        result = rlm.completion(query, context)

        print("=" * 70)
        print("Result:")
        print("=" * 70)
        print(result)
        print("\n" + "=" * 70)

    except Exception as e:
        print(f"Error: {e}")


def demo_e2b_vs_restricted():
    """Compare E2B vs RestrictedPython sandboxes."""
    print("\n" + "=" * 70)
    print("Sandbox Comparison: E2B vs RestrictedPython")
    print("=" * 70)

    context = "Numbers: 15, 23, 42, 8, 91, 17, 33, 56, 72, 11"
    query = "Find the median of these numbers"

    print(f"\nQuery: {query}")
    print(f"Context: {context}")

    # Test with E2B
    print("\n1. E2B Sandbox:")
    print("-" * 70)
    try:
        rlm_e2b = RLM(
            model="gpt-4o-mini",
            sandbox='e2b',
            max_iterations=10
        )
        result_e2b = rlm_e2b.completion(query, context)
        print(f"Result: {result_e2b}")
        print(f"Iterations: {rlm_e2b.stats['iterations']}")
    except Exception as e:
        print(f"E2B error (may need API key): {e}")

    # Test with RestrictedPython
    print("\n2. RestrictedPython Sandbox:")
    print("-" * 70)
    try:
        rlm_restricted = RLM(
            model="gpt-4o-mini",
            sandbox='restricted',
            max_iterations=10
        )
        result_restricted = rlm_restricted.completion(query, context)
        print(f"Result: {result_restricted}")
        print(f"Iterations: {rlm_restricted.stats['iterations']}")
    except Exception as e:
        print(f"RestrictedPython error: {e}")

    print("\n" + "=" * 70)
    print("Key Differences:")
    print("- E2B: Cloud-based, can install packages, more isolated")
    print("- RestrictedPython: Local, limited imports, faster startup")
    print("=" * 70)


def demo_security():
    """Demonstrate sandbox security features."""
    print("\n" + "=" * 70)
    print("Sandbox Security Demo")
    print("=" * 70)

    print("\nBoth E2B and RestrictedPython prevent:")
    print("  - File system access (except allowed operations)")
    print("  - Network requests (unless explicitly allowed)")
    print("  - System command execution")
    print("  - Import of dangerous modules")

    print("\nE2B provides additional isolation:")
    print("  - Runs in separate cloud container")
    print("  - Complete process isolation")
    print("  - Resource limits (CPU, memory, time)")
    print("  - Can be destroyed after execution")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Check for API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_e2b = bool(os.getenv("E2B_API_KEY"))

    if not has_openai:
        print("❌ Error: OPENAI_API_KEY not found!")
        print("\nPlease set up your OpenAI API key in .env file")
        exit(1)

    if not has_e2b:
        print("⚠️  Warning: E2B_API_KEY not found!")
        print("E2B examples will fail. Get your key at: https://e2b.dev")
        print("\nWill demonstrate with RestrictedPython fallback where possible.\n")

    # Run demos
    demo_e2b_basic()

    # Uncomment to run additional demos:
    # demo_e2b_advanced()
    # demo_e2b_vs_restricted()
    # demo_security()
