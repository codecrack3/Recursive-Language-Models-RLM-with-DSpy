"""Example: Using RLM with OpenRouter models."""

import os
from dotenv import load_dotenv
from rlm import RLM

# Load environment variables
load_dotenv()

# Sample document for testing
sample_document = """
Project Timeline Analysis

Q1 2024:
- January: Project kickoff, team formation
- February: Requirements gathering and initial design
- March: Development phase 1 completed

Q2 2024:
- April: Testing and bug fixes
- May: Feature enhancements based on feedback
- June: Production deployment

Q3 2024 (Projected):
- July: Performance optimization
- August: User onboarding and training
- September: Full-scale rollout

The project has been proceeding smoothly with all Q1 and Q2 milestones met on time.
""" * 3  # Multiply for longer context


def demo_basic_openrouter():
    """Basic OpenRouter usage with Claude."""
    print("=" * 70)
    print("OpenRouter - Basic Usage (Claude 4.5 Sonnet)")
    print("=" * 70)

    query = "What are the completed milestones and what's still projected?"

    print(f"\nQuery: {query}")
    print(f"Context length: {len(sample_document):,} characters")
    print("Model: Anthropic Claude 4.5 Sonnet (via OpenRouter)")
    print("\nProcessing...\n")

    try:
        # Use Claude via OpenRouter
        rlm = RLM(
            model="openrouter/anthropic/claude-sonnet-4.5",
            recursive_model="openrouter/anthropic/claude-haiku-4.5",
            max_iterations=10
        )

        result = rlm.completion(query, sample_document)

        print("=" * 70)
        print("Result:")
        print("=" * 70)
        print(result)
        print("\n" + "=" * 70)
        print(f"Stats: {rlm.stats}")

    except ValueError as e:
        if "OPENROUTER_API_KEY" in str(e):
            print("❌ Error: OPENROUTER_API_KEY not found!")
            print("\nTo use OpenRouter:")
            print("  1. Get API key at https://openrouter.ai/keys")
            print("  2. Add to .env: OPENROUTER_API_KEY=your-key")
            print("  3. Or pass as parameter: RLM(model='...', api_key='...')")
        else:
            raise
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def demo_model_comparison():
    """Compare different models via OpenRouter."""
    print("\n" + "=" * 70)
    print("OpenRouter - Model Comparison")
    print("=" * 70)

    query = "Summarize the project status in one sentence."
    context = sample_document[:1000]  # Use shorter context for quick demo

    models = [
        ("openrouter/anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet"),
        ("openrouter/openai/gpt-4o-mini", "GPT-4o Mini"),
        ("openrouter/google/gemini-pro", "Gemini Pro"),
    ]

    print(f"\nQuery: {query}\n")

    for model_id, model_name in models:
        print(f"{model_name}:")
        print("-" * 70)

        try:
            rlm = RLM(model=model_id, max_iterations=5)
            result = rlm.completion(query, context)
            print(f"Result: {result}")
            print(f"Iterations: {rlm.stats['iterations']}\n")
        except Exception as e:
            print(f"Error: {e}\n")

    print("=" * 70)


def demo_cost_optimization():
    """Demonstrate cost optimization with OpenRouter."""
    print("\n" + "=" * 70)
    print("OpenRouter - Cost Optimization Strategy")
    print("=" * 70)

    print("\nStrategy: Use cheaper models for recursive calls")

    query = "What were the Q2 achievements?"

    try:
        # Use expensive model for root, cheap for recursive
        rlm = RLM(
            model="openrouter/anthropic/claude-3.5-sonnet",  # Premium for root
            recursive_model="openrouter/anthropic/claude-3-haiku",  # Cheap for recursion
            max_iterations=15
        )

        print(f"\nRoot model: Claude 3.5 Sonnet (premium)")
        print(f"Recursive model: Claude 3 Haiku (economical)")
        print(f"\nQuery: {query}")
        print("\nProcessing...\n")

        result = rlm.completion(query, sample_document)

        print("=" * 70)
        print("Result:")
        print("=" * 70)
        print(result)
        print("\n" + "=" * 70)
        print(f"Stats: {rlm.stats}")
        print("\n💡 Cost saved by using cheaper model for recursive calls!")

    except Exception as e:
        print(f"Error: {e}")


def demo_available_models():
    """Show available OpenRouter model categories."""
    print("\n" + "=" * 70)
    print("OpenRouter - Available Model Categories")
    print("=" * 70)

    models_by_category = {
        "🧠 Anthropic (Claude)": [
            "openrouter/anthropic/claude-3.5-sonnet",
            "openrouter/anthropic/claude-3-opus",
            "openrouter/anthropic/claude-3-sonnet",
            "openrouter/anthropic/claude-3-haiku",
        ],
        "🤖 OpenAI (GPT)": [
            "openrouter/openai/gpt-4o",
            "openrouter/openai/gpt-4o-mini",
            "openrouter/openai/gpt-4-turbo",
            "openrouter/openai/gpt-3.5-turbo",
        ],
        "✨ Google (Gemini)": [
            "openrouter/google/gemini-pro-1.5",
            "openrouter/google/gemini-pro",
            "openrouter/google/gemini-flash-1.5",
        ],
        "🦙 Meta (Llama)": [
            "openrouter/meta-llama/llama-3.1-405b-instruct",
            "openrouter/meta-llama/llama-3.1-70b-instruct",
            "openrouter/meta-llama/llama-3.1-8b-instruct",
        ],
        "🌟 Mistral": [
            "openrouter/mistralai/mistral-large",
            "openrouter/mistralai/mistral-medium",
            "openrouter/mistralai/mistral-small",
        ],
    }

    for category, models in models_by_category.items():
        print(f"\n{category}:")
        for model in models:
            print(f"  - {model}")

    print("\n" + "=" * 70)
    print("💡 Full model list: https://openrouter.ai/models")
    print("=" * 70)


if __name__ == "__main__":
    # Check for API key
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))

    if not has_openrouter:
        print("=" * 70)
        print("⚠️  OPENROUTER_API_KEY not found!")
        print("=" * 70)
        print("\nTo use OpenRouter:")
        print("  1. Sign up at https://openrouter.ai")
        print("  2. Get your API key at https://openrouter.ai/keys")
        print("  3. Add to .env file:")
        print("     OPENROUTER_API_KEY=your-key-here")
        print("\n" + "=" * 70)
        print("Showing available models without making API calls:")
        print("=" * 70)
        demo_available_models()
        exit(1)

    # Run demos
    demo_basic_openrouter()

    # Uncomment to run additional demos:
    # demo_model_comparison()
    # demo_cost_optimization()
    # demo_available_models()
