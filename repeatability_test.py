"""
repeatability_test.py
----------------------
Runs Prompt 1 and Prompt 2 against the configured LLM provider (LLM_PROVIDER
in .env) REPEAT_COUNT times each, and prints a side-by-side table of the
per-run responses so you can eyeball how consistent each prompt is.

Run:  python repeatability_test.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.table import Table

from app import load_prompt
from llm_client import create_llm

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
console = Console()


def _label(prompt_name: str) -> str:
    """'unstructured_prompt' -> 'Unstructured'."""
    return prompt_name.replace("_prompt", "").replace("_", " ").strip().capitalize()


class PromptRepeatabilityTest:
    """Calls Prompt 1 and Prompt 2 repeatedly against one provider and tabulates responses."""

    def __init__(self, provider: Optional[str] = None, repeat_count: Optional[int] = None):
        configured = os.getenv("LLM_PROVIDER", "ollama").split(",")[0].strip()
        self.provider = (provider or configured).lower()
        self.repeat_count = repeat_count or int(os.getenv("REPEAT_COUNT", "10"))

        num_a, num_b = os.getenv("NUM_A", "25"), os.getenv("NUM_B", "17")
        variables = {"NUM_A": num_a, "NUM_B": num_b}
        self.prompt1_name = os.getenv("PROMPT_1_NAME", "unstructured_prompt")
        self.prompt2_name = os.getenv("PROMPT_2_NAME", "structured_prompt")
        self.prompt1_text = load_prompt(self.prompt1_name, variables)
        self.prompt2_text = load_prompt(self.prompt2_name, variables)

    def test_prompt_repeatability(self) -> list[dict]:
        """Runs both prompts `repeat_count` times and returns each run's responses."""
        llm = create_llm(self.provider)
        rows = []
        for i in range(1, self.repeat_count + 1):
            with console.status(f"[cyan]{self.provider}: run {i}/{self.repeat_count}..."):
                response1 = llm.generate(self.prompt1_text).text.strip()
                response2 = llm.generate(self.prompt2_text).text.strip()
            rows.append({"run": i, "prompt1_response": response1, "prompt2_response": response2})
        self._render(rows, llm.model)
        return rows

    def _render(self, rows: list[dict], model: str) -> None:
        table = Table(
            title=f"Prompt Repeatability  |  {self.provider.upper()}  |  {model}  |  "
                  f"{len(rows)} runs",
            box=box.ROUNDED, header_style="bold cyan", show_lines=True,
        )
        table.add_column("Sr. No", justify="right")
        table.add_column(f"Prompt 1 Response ({_label(self.prompt1_name)})", justify="left")
        table.add_column(f"Prompt 2 Response ({_label(self.prompt2_name)})", justify="left")

        for row in rows:
            table.add_row(
                str(row["run"]),
                row["prompt1_response"] or "[dim](empty response)[/dim]",
                row["prompt2_response"] or "[dim](empty response)[/dim]",
            )

        console.print()
        console.print(table)


def main() -> int:
    test = PromptRepeatabilityTest()
    console.print(
        f"[magenta]Running {test.repeat_count} repetitions of Prompt 1 and Prompt 2 "
        f"against provider '{test.provider}'...[/magenta]"
    )
    try:
        test.test_prompt_repeatability()
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
