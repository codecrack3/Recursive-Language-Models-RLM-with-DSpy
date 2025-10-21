# Recursive Language Models - DSpy (RLM)

Using Python and DSpy’s Recursive Language Model implementation to handle unbounded context lengths.

**Based on [the paper](https://alexzhang13.github.io/blog/2025/rlm/) by Alex Zhang and Omar Khattab (MIT, 2025)**

**Based on [the source code](https://github.com/ysz/recursive-llm) by ysz**


## What is RLM?

RLM enables language models to process extremely long contexts (100k+ tokens) by:
- Storing context as a Python variable instead of in the prompt
- Allowing the LM to recursively explore and partition the context
- Avoiding "context rot" (performance degradation with long context)

Instead of this:
```python
llm.complete(prompt="Summarize this", context=huge_document)  # Context rot!
```

RLM does this:
```python
rlm = RLM(model="gpt-5-mini")
result = rlm.completion(
    query="Summarize this",
    context=huge_document  # Stored as variable, not in prompt
)
```

The LM can then peek, search, and recursively process the context adaptively.

## Installation

**Note:** This package is not yet published to PyPI. Install from source:

```bash
# Clone the repository
git clone https://github.com/codecrack3/recursive-llm.git
cd recursive-llm

# Install in editable mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

**Future:** Once published to PyPI, you'll be able to install with `pip install recursive-llm`

## Requirements

- Python 3.9 or higher
- An API key for your chosen LLM provider (OpenAI, Anthropic, etc.)
- Or a local model setup (Ollama, llama.cpp, etc.)

## Quick Start

```python
from rlm import RLM

# Initialize with any LLM (auto-selects best backend)
rlm = RLM(model="gpt-4o-mini")

# Process long context
result = rlm.completion(
    query="What are the main themes in this document?",
    context=long_document
)
print(result)
```

### DSPy + E2B Sandbox

```python
from rlm import RLM

# RLM uses DSPy for LLM orchestration with E2B cloud sandbox
rlm = RLM(
    model="gpt-4o-mini",
    sandbox='e2b'        # E2B cloud sandbox (or 'auto' to auto-detect)
)

result = rlm.completion(query, context)
```

## API Keys Setup

Set your API key via environment variable or pass it directly:

```bash
export OPENAI_API_KEY="sk-..."  # or ANTHROPIC_API_KEY, etc.
```

Or pass directly in code:
```python
rlm = RLM(model="gpt-5-mini", api_key="sk-...")
```

## Supported Models

Works with 100+ LLM providers via DSPy and OpenRouter:

```python
# OpenAI
rlm = RLM(model="gpt-4o-mini")
rlm = RLM(model="gpt-4o")

# Anthropic
rlm = RLM(model="claude-sonnet-4")
rlm = RLM(model="claude-sonnet-4-20250514")

# OpenRouter (100+ models with single API key)
rlm = RLM(model="openrouter/anthropic/claude-3.5-sonnet")
rlm = RLM(model="openrouter/openai/gpt-4o-mini")
rlm = RLM(model="openrouter/google/gemini-pro")
rlm = RLM(model="openrouter/meta-llama/llama-3.1-70b")

# Ollama (local)
rlm = RLM(model="ollama/llama3.2")
rlm = RLM(model="ollama/mistral")

# llama.cpp (local)
rlm = RLM(
    model="openai/local",
    api_base="http://localhost:8000/v1"
)

# Azure OpenAI
rlm = RLM(model="azure/gpt-4-deployment")

# And many more...
```

## Advanced Usage

### Two Models (Optimize Cost)

Use a cheaper model for recursive calls:

```python
rlm = RLM(
    model="gpt-5",              # Root LM (main decisions)
    recursive_model="gpt-5-mini"  # Recursive calls (cheaper)
)
```

### Async API

For better performance with parallel recursive calls:

```python
import asyncio

async def main():
    rlm = RLM(model="gpt-5-mini")
    result = await rlm.acompletion(query, context)
    print(result)

asyncio.run(main())
```

### Configuration

```python
rlm = RLM(
    model="gpt-5-mini",
    max_depth=5,         # Maximum recursion depth
    max_iterations=20,   # Maximum REPL iterations
    temperature=0.7,     # LLM parameters
    timeout=60
)
```

## How It Works

1. **Context is stored as a variable** in a Python REPL environment
2. **Root LM gets only the query** plus instructions
3. **LM can explore context** using Python code:
   ```python
   # Peek at context
   context[:1000]

   # Search with regex
   import re
   re.findall(r'pattern', context)

   # Recursive processing
   recursive_llm("extract dates", context[1000:2000])
   ```
4. **Returns final answer** via `FINAL(answer)` statement

## Graph Tracking & Visualization

Visualize recursive LLM calls with interactive NetworkX graphs:

```python
from rlm import RLM

# Enable graph tracking
rlm = RLM(
    model="gpt-4o-mini",
    enable_graph_tracking=True,
    graph_output_path="./rlm_graph.html"
)

result = rlm.completion(query="Analyze this", context=document)
# Graph automatically saved to ./rlm_graph.html
```

The interactive HTML visualization shows:
- **Hierarchical structure**: See the complete call tree
- **Node details**: Input/output for each recursive call
- **REPL iterations**: Code generated and executed at each step
- **Performance metrics**: Iterations and LLM calls per node
- **Error tracking**: Which nodes encountered issues

![Graph Visualization](docs/graph_example.png)

**Programmatic access:**
```python
import networkx as nx

# Get the graph object
graph = rlm.get_graph()
print(f"Total nodes: {graph.number_of_nodes()}")

# Analyze the graph structure
for node_id, node_data in graph.nodes(data=True):
    print(f"Depth {node_data['depth']}: {node_data['iterations']} iterations")

# Save to different location
rlm.save_graph("./analysis/custom_graph.html")
```

**Learn more:** See [`docs/GRAPH_TRACKING.md`](docs/GRAPH_TRACKING.md) for full documentation.

### 🔍 LLM Call History (Debugging)

Track and inspect all LLM calls for debugging prompts and responses:

```python
# Enable history tracking
rlm = RLM(
    model="gpt-4o-mini",
    enable_history=True,  # Enable LLM call history
    history_output_path="./logs/history.json"  # Auto-save to JSON (optional)
)

result = rlm.completion(query="Your query", context=document)

# Print history summary (shows model, messages, outputs)
rlm.print_history(detailed=False)

# Print detailed history with full prompts/responses
rlm.print_history(detailed=True, max_length=2000)

# Get raw history for programmatic access
history = rlm.get_history()
print(f"Total LLM calls: {len(history)}")

# Save history to JSON manually
rlm.save_history("./my_history.json", pretty=True)

# Clear history for new run
rlm.clear_history()
```

**Use cases:**
- 🐛 **Debug prompts**: See exactly what's being sent to the LLM (shows full messages/inputs)
- 📊 **Analyze responses**: Inspect the raw outputs from each call
- 🔧 **Optimize prompts**: Iterate on prompt engineering
- 📈 **Monitor usage**: Track token usage and costs (exports to JSON)
- 💾 **Export logs**: Auto-save or manually export history as JSON
- 🔄 **Combine with graph tracking**: Visualize + inspect LLM calls

**Tip:** Combine `enable_history=True` and `enable_graph_tracking=True` for comprehensive debugging!

## Examples

See the `examples/` directory for complete working examples:
- `basic_usage.py` - Simple completion with OpenAI
- `dspy_usage.py` - DSPy backend with E2B sandbox
- `openrouter_usage.py` - OpenRouter multi-model access
- `e2b_usage.py` - E2B cloud sandbox features
- `ollama_local.py` - Using Ollama locally
- `two_models.py` - Cost optimization with two models
- `long_document.py` - Processing 50k+ token documents
- `data_extraction.py` - Extract structured data from text
- `multi_file.py` - Process multiple documents
- `custom_config.py` - Advanced configuration
- `graph_tracking.py` - NetworkX visualization of recursive calls

Run an example:
```bash
# Set your API key first
export OPENAI_API_KEY="sk-..."

# Run example
python examples/basic_usage.py
```

## Performance

### Paper Results

On OOLONG benchmark (132k tokens):
- GPT-5: baseline
- RLM(GPT-5-Mini): **33% better than GPT-5** at similar cost

### Our Benchmark Results

Tested with GPT-5-Mini on structured data queries (counting, filtering) across 5 different test cases:

**60k token contexts:**
- **RLM**: 80% accurate (4/5 correct)
- **Direct OpenAI**: 0% accurate (0/5 correct, all returned approximations)

RLM wins on accuracy. Both complete requests, but only RLM gives correct answers.

**150k+ token contexts:**
- **Direct OpenAI**: Fails (rate limit errors)
- **RLM**: Works (processes 1M+ tokens successfully)

**Token efficiency:** RLM uses ~2-3k tokens per query vs 95k+ for direct approach, since context is stored as a variable instead of being sent in prompts.

## Development

```bash
# Clone repository
git clone https://github.com/codecrack3/recursive-llm.git
cd recursive-llm

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=src/rlm --cov-report=term-missing

# Type checking
mypy src/rlm

# Linting
ruff check src/rlm

# Format code
black src/rlm tests examples
```

## Architecture

RLM uses DSPy for LLM orchestration with flexible sandbox options:

```
RLM (DSPy Backend)
├── Custom RLMModule (DSPy module for REPL pattern)
├── E2B Sandbox (cloud code execution)
└── RestrictedPython Fallback (local execution)
```

**Key Features**:
- Programmatic LLM orchestration via DSPy
- Better prompt optimization and composability
- Supports E2B cloud sandboxes for enhanced security
- Custom `RLMModule` optimized for recursive REPL pattern
- RestrictedPython fallback for local execution

## Sandbox Selection

### Automatic Selection (Recommended)

```python
# Auto-selects E2B if API key set, otherwise RestrictedPython
rlm = RLM(model="gpt-4o-mini")
```

### Explicit Sandbox Selection

```python
# Use E2B cloud sandbox (requires E2B_API_KEY)
rlm = RLM(model="gpt-4o-mini", sandbox='e2b')

# Use RestrictedPython (no API key needed, runs locally)
rlm = RLM(model="gpt-4o-mini", sandbox='restricted')
```

### Environment Variables

Configure sandbox preferences via environment variables:

```bash
# Sandbox selection
export RLM_SANDBOX=e2b         # or 'restricted', 'auto'

# E2B API key (get from https://e2b.dev)
export E2B_API_KEY=your-key-here
```

## What's New in v0.2.0

### Simplified Architecture
- **DSPy Only**: Removed legacy LiteLLM backend for simpler codebase
- **Cleaner API**: No more backend selection - DSPy is the only backend
- **Better Maintainability**: Reduced complexity and dependencies

### DSPy Integration
- **Programmatic LLM Orchestration**: Use DSPy for better prompt engineering
- **Custom RLMModule**: Purpose-built DSPy module for recursive REPL pattern
- **Automatic Optimization**: DSPy's optimization capabilities (optional)

### E2B Sandbox
- **Cloud Execution**: Secure sandboxed code execution in isolated containers
- **Enhanced Security**: Better isolation than local RestrictedPython
- **Package Installation**: Install Python packages on-the-fly if needed
- **Auto-Fallback**: Gracefully falls back to RestrictedPython if E2B unavailable

## Migration Guide

### From v0.1.0 to v0.2.0

The LiteLLM backend has been removed in v0.2.0. If you were using the `backend` parameter, simply remove it:

**Before (v0.1.0)**:
```python
from rlm import RLM
rlm = RLM(model="gpt-4o-mini", backend='dspy')
result = rlm.completion(query, context)
```

**After (v0.2.0)**:
```python
from rlm import RLM
rlm = RLM(model="gpt-4o-mini")  # backend parameter removed
result = rlm.completion(query, context)
```

**Breaking Changes**:
- Removed `backend` parameter (only DSPy is supported now)
- Removed `RLMLiteLLM` class
- Removed `Backend` type from exports
- Removed `litellm` dependency

### Setting Up E2B

1. Get API key from https://e2b.dev
2. Add to `.env` file:
   ```bash
   E2B_API_KEY=your-key-here
   ```
3. RLM will automatically use E2B when available

### Setting Up OpenRouter

OpenRouter provides access to 100+ models through a single API key:

1. Sign up at https://openrouter.ai
2. Get your API key at https://openrouter.ai/keys
3. Add to `.env` file:
   ```bash
   OPENROUTER_API_KEY=your-key-here
   ```

**Usage:**
```python
# Anthropic Claude via OpenRouter
rlm = RLM(model="openrouter/anthropic/claude-3.5-sonnet")

# OpenAI GPT via OpenRouter
rlm = RLM(model="openrouter/openai/gpt-4o-mini")

# Google Gemini via OpenRouter
rlm = RLM(model="openrouter/google/gemini-pro")

# Meta Llama via OpenRouter
rlm = RLM(model="openrouter/meta-llama/llama-3.1-70b-instruct")
```

**Benefits:**
- ✅ Access 100+ models with single API key
- ✅ No rate limits on most models
- ✅ Competitive pricing
- ✅ Automatic fallback if model unavailable
- ✅ Easy model switching for testing

**Cost optimization:**
```python
# Use premium model for root, economical for recursion
rlm = RLM(
    model="openrouter/anthropic/claude-3.5-sonnet",
    recursive_model="openrouter/anthropic/claude-3-haiku"
)
```

See full model list: https://openrouter.ai/models

## Limitations

- REPL execution is sequential (no parallel code execution yet)
- No prefix caching (future enhancement)
- Recursion depth is limited (configurable via `max_depth`)
- No streaming support yet
- E2B requires API key for cloud sandboxes (free tier available)

## Troubleshooting

### "Max iterations exceeded"
- Increase `max_iterations` parameter
- Simplify your query
- Check if the model is getting stuck in a loop

### "API key not found"
- Set the appropriate environment variable (e.g., `OPENAI_API_KEY`)
- Or pass `api_key` parameter to RLM constructor

### "Model not found"
- Check model name format for your provider
- See DSPy docs: https://dspy-docs.vercel.app/

### Using Ollama
- Make sure Ollama is running: `ollama serve`
- Pull a model first: `ollama pull llama3.2`
- Use model format: `ollama/model-name`

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass (`pytest tests/`)
5. Follow code style (use `black` and `ruff`)
6. Submit a pull request

## Citation

This implementation is based on the RLM paper by Alex Zhang and Omar Khattab.

**To cite this implementation:**
```bibtex
@software{rlm_python,
  title = {recursive-llm-dspy: Using Python and DSpy’s Recursive Language Model implementation to handle unbounded context lengths},
  author = {codecrack3},
  year = {2025},
  url = {https://github.com/codecrack3/recursive-llm}
}

@software{rlm_python,
  title = {recursive-llm: Python Implementation of Recursive Language Models},
  author = {ysz},
  year = {2025},
  url = {https://github.com/ysz/recursive-llm}
}

@software{rlm_python,
  title = {Recursive Language Models (minimal version)},
  author = {alexzhang13},
  year = {2025},
  url = {https://github.com/alexzhang13/rlm}
}
```

**To cite the original paper:**
```bibtex
@misc{zhang2025rlm,
  title = {Recursive Language Models},
  author = {Zhang, Alex and Khattab, Omar},
  year = {2025},
  month = {October},
  url = {https://alexzhang13.github.io/blog/2025/rlm/}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

Based on the Recursive Language Models paper by Alex Zhang and Omar Khattab from MIT CSAIL.

Built using:
- DSPy for LLM orchestration
- E2B for cloud code execution
- RestrictedPython for safe local code execution

## Links

- **Paper**: https://alexzhang13.github.io/blog/2025/rlm/
- **DSPy Docs**: https://dspy-docs.vercel.app/
- **E2B Docs**: https://e2b.dev/docs
- **Issues**: https://github.com/codecrack3/recursive-llm/issues
