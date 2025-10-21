"""Rich-based console logger for tracking Recursive Language Model runs."""

from __future__ import annotations

from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text
    from rich.rule import Rule
except ImportError as exc:  # pragma: no cover
    Console = None  # type: ignore
    Panel = None  # type: ignore
    Syntax = None  # type: ignore
    Text = None  # type: ignore
    Rule = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class RLMRunLogger:
    """Jupyter-style console logger inspired by REPLEnvLogger."""

    def __init__(
        self,
        enabled: bool = True,
        preview_limit: int = 200,
        console: Optional[Console] = None,
        truncate_text: bool = True,
        truncate_code: bool = True,
    ) -> None:
        self.enabled = enabled
        self.preview_limit = preview_limit
        self.truncate_text = truncate_text
        self.truncate_code = truncate_code
        if self.enabled:
            if _IMPORT_ERROR is not None:
                raise RuntimeError(
                    "rich is required for logging. Install with `pip install rich`."
                ) from _IMPORT_ERROR
            self.console = console or Console()
        else:
            self.console = None

    # Public logging API -------------------------------------------------

    def run_start(self, depth: int, query: str, context_size: int) -> None:
        if not self.enabled:
            return
        self._print_rule(f"RLM depth {depth} start")
        body = Text()
        body.append("Context size: ", style="bold")
        body.append(f"{context_size}\n", style="green")
        body.append("Query: ", style="bold")
        body.append(self._truncate(query) if query else "n/a", style="cyan")
        self.console.print(Panel(body, border_style="green", title="Session"))

    def run_end(self, depth: int, answer: Optional[str]) -> None:
        if not self.enabled:
            return
        text = Text()
        text.append("Answer: ", style="bold")
        text.append(self._truncate(answer), style="green")
        self.console.print(Panel(text, title=f"RLM depth {depth} completed ✅", border_style="green"))
        self._print_rule("")

    def iteration_start(self, iteration: int, depth: int, query: str, context_size: int) -> None:
        if not self.enabled:
            return
        text = Text()
        text.append(f"Iteration {iteration}\n", style="bold cyan")
        text.append(f"Depth: {depth}\n", style="magenta")
        text.append(f"Context size: {context_size}", style="cyan")
        self.console.print(Panel(text, border_style="cyan", title="Iteration Start"))

    def before_llm_call(
        self,
        iteration: int,
        depth: int,
        query: str,
        context_size: int,
        history: Optional[str]
    ) -> None:
        if not self.enabled:
            return
        text = Text()
        text.append(f"Calling LLM (iteration {iteration})\n", style="bold blue")
        if history:
            text.append("History:\n", style="bright_black")
            text.append(self._truncate(history), style="white")
        else:
            text.append("History: <empty>", style="dim")
        self.console.print(Panel(text, border_style="blue", title="LLM Call"))

    def after_llm_call(
        self,
        iteration: int,
        depth: int,
        code: str,
        reasoning: Optional[str]
    ) -> None:
        if not self.enabled:
            return
        code_panel = self._code_panel(code, title=f"LLM Code (iter {iteration})")
        self.console.print(code_panel)
        if reasoning:
            reasoning_text = Text(self._truncate(reasoning), style="yellow")
            self.console.print(Panel(reasoning_text, border_style="yellow", title="Reasoning"))

    def before_execution(
        self,
        iteration: int,
        depth: int,
        code: str,
        blocked_reason: Optional[str]
    ) -> None:
        if not self.enabled:
            return
        if blocked_reason:
            reason_text = Text(self._truncate(blocked_reason), style="red")
            self.console.print(
                Panel(reason_text, border_style="red", title=f"Execution blocked (iter {iteration})")
            )
        else:
            self.console.print(self._code_panel(code, title=f"Executing code (iter {iteration})", style="magenta"))

    def execution_exception(self, iteration: int, depth: int, error: str) -> None:
        if not self.enabled:
            return
        text = Text(self._truncate(error), style="red")
        self.console.print(Panel(text, border_style="red", title=f"Execution exception (iter {iteration})"))

    def after_execution(
        self,
        iteration: int,
        depth: int,
        output: str,
        error: Optional[str]
    ) -> None:
        if not self.enabled:
            return
        style = "red" if error else "green"
        title = f"Execution result (iter {iteration})"
        text = Text(self._truncate(output or "<no output>"), style="white")
        panel = Panel(text, border_style=style, title=title)
        self.console.print(panel)

    def on_duplicate_code(self, iteration: int, depth: int, code: str) -> None:
        if not self.enabled:
            return
        panel = self._code_panel(code, title=f"Duplicate code detected (iter {iteration})", style="yellow")
        self.console.print(panel)

    def on_final(self, iteration: int, depth: int, answer: str) -> None:
        if not self.enabled:
            return
        text = Text(self._truncate(answer), style="green")
        self.console.print(Panel(text, border_style="green", title=f"FINAL at iteration {iteration} 🏁"))

    def on_max_iterations(self, max_iterations: int, depth: int) -> None:
        if not self.enabled:
            return
        text = Text(f"Max iterations ({max_iterations}) reached without FINAL()", style="red")
        self.console.print(Panel(text, border_style="red", title="Max Iterations Reached"))

    def on_error(self, depth: int, error: str) -> None:
        if not self.enabled:
            return
        text = Text(self._truncate(error), style="red")
        self.console.print(Panel(text, border_style="red", title=f"Error at depth {depth}"))

    # Internal helpers -------------------------------------------------

    def _truncate(self, value: Optional[str], limit: Optional[int] = None, *, is_code: bool = False) -> str:
        if not value:
            return ""
        limit = limit or self.preview_limit
        # handle tuple object not attribute strip
        if isinstance(value, tuple):
            value = value[0]
        trimmed = value.strip()
        
        allow_truncate = self.truncate_code if is_code else self.truncate_text
        if not allow_truncate:
            return trimmed
        if len(trimmed) > limit:
            return f"{trimmed[:limit]}... (truncated {len(trimmed) - limit} chars)"
        return trimmed

    def _code_panel(self, code: str, title: str, style: str = "green") -> Panel:
        snippet = self._truncate(code, self.preview_limit, is_code=True)
        if Syntax is not None:
            syntax = Syntax(snippet, "python", theme="monokai", line_numbers=False)
            return Panel(syntax, border_style=style, title=title)
        text = Text(snippet, style="white")
        return Panel(text, border_style=style, title=title)

    def _print_rule(self, title: str) -> None:
        if Rule is None:
            return
        rule_title = title or ""
        self.console.print(Rule(title=rule_title, style="grey50"))
