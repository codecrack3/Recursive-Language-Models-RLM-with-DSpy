"""DSPy signatures for Recursive Language Models."""

import dspy

REPL_SYSTEM_PROMPT = """You are a Recursive Language Model that solves tasks by writing Python code inside a sandboxed REPL. You never receive the raw document in your prompt. Instead, the REPL starts with:
- context: str (the complete document to analyze; not shown here)
- query: str (the user request you must answer)
- recursive_llm(sub_query: str, sub_context: str) -> str (use to delegate work on large chunks)
- llm_query(sub_query: str, sub_context: str) -> str (alias for recursive_llm)
- re (the Python regular-expression module)

General rules:
- Use Python statements to inspect and transform `context` (for example, print(context[:200])).
- Do not import modules or access the filesystem, network, or OS. Rely only on the provided variables and built-ins.
- Show intermediate results with print statements or by leaving an expression as the last line so you can see its value.
- Plan your approach, explore the context as needed, and iterate until you can answer `query`.
- When you finish, emit FINAL("answer") or FINAL_VAR(variable_name) as plain text (not inside a code block).

You are told the size of the context through the `context_size` input and the current recursion depth through `depth`. Think step by step and use recursive_llm when it helps manage large analyses.
"""


class RLMSignature(dspy.Signature):
    """
    Signature for RLM code generation.

    The LM receives a query and information about the context (size, depth),
    and generates Python code to explore and answer the query.
    """

    # Inputs
    query = dspy.InputField(
        desc="The question or task to answer using the context"
    )
    context_size = dspy.InputField(
        desc="Size of the context in characters (context is stored as a variable, not in prompt)"
    )
    depth = dspy.InputField(
        desc="Current recursion depth (0 for root call)"
    )
    previous_attempts = dspy.InputField(
        desc="Previous code and results (if any iterations have been performed)",
        default=""
    )

    # Output
    code = dspy.OutputField(
        desc=(
            "Pure Python code (no markdown fences) that inspects the `context` variable, optionally uses "
            "`recursive_llm`, and progresses toward answering the query. Avoid imports. "
            "When finished, you will emit FINAL(\"answer\") or FINAL_VAR(name) as plain text in a later turn."
        )
    )


class RecursiveCallSignature(dspy.Signature):
    """
    Signature for recursive RLM calls.

    Used when the LM makes a recursive call via recursive_llm(sub_query, sub_context).
    """

    # Inputs
    sub_query = dspy.InputField(
        desc="The specific question for this sub-context"
    )
    sub_context = dspy.InputField(
        desc="The sub-portion of the context to analyze"
    )
    depth = dspy.InputField(
        desc="Current recursion depth"
    )

    # Output
    answer = dspy.OutputField(
        desc="The answer extracted from the sub-context"
    )


class RLMInstructionSignature(dspy.Signature):
    """
    Enhanced signature with explicit instructions for the RLM pattern.

    This signature includes detailed instructions about the REPL environment
    and available operations, making it clearer for the LM.
    """

    # System-level instruction
    __doc__ = REPL_SYSTEM_PROMPT

    # Inputs
    query = dspy.InputField(desc="The question or task to answer")
    context_size = dspy.InputField(desc="Size of context in characters")
    depth = dspy.InputField(desc="Current recursion depth")
    previous_output = dspy.InputField(
        desc="Output from previous code execution (if any)",
        default=""
    )

    # Output
    reasoning = dspy.OutputField(
        desc="Brief explanation of your next steps for exploring the context"
    )
    code = dspy.OutputField(
        desc=(
            "Pure Python code (no markdown fences) to execute next in the REPL. "
            "Use the provided `context`, `query`, `recursive_llm`, and `re` module. Avoid imports."
        )
    )
