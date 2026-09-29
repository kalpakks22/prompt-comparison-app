"""
app.py - Prompt Comparison App
------------------------------
Sends the same task with two different prompts (loaded from .md files named in
.env) to one or more LLM providers, then prints a side-by-side comparison of
tokens, latency, cost and answer quality.

Run:  python app.py
"""
from __future__ import annotations

import csv
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_client import LLMResult, create_llm

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
console = Console()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_prompt(name: str, variables: dict) -> str:
    """Load prompts/<name>.md and replace {{PLACEHOLDERS}} with values."""
    prompt_dir = BASE_DIR / os.getenv("PROMPT_DIR", "prompts")
    path = prompt_dir / (name if name.endswith(".md") else f"{name}.md")
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    for key, value in variables.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def env_float(key: str) -> Optional[float]:
    raw = os.getenv(key, "").strip()
    return float(raw) if raw else None


def expected_answer(a: str, b: str) -> str:
    """The correct answer, used for the accuracy check."""
    try:
        total = float(a) + float(b)
    except ValueError:
        return "ERROR"
    return str(int(total)) if total.is_integer() else str(total)


def check_accuracy(output: str, expected: str) -> tuple[str, str]:
    """Returns (is_correct, output_format)."""
    clean = output.strip().strip("`").strip()
    if expected == "ERROR":
        correct = "error" in clean.lower() or "invalid" in clean.lower()
        fmt = "Clean" if clean.upper().startswith("ERROR") and len(clean) < 40 else "Verbose"
    else:
        numbers = [n.replace(",", "") for n in re.findall(r"-?\d[\d,]*(?:\.\d+)?", clean)]
        correct = expected in numbers
        fmt = "Clean" if clean.replace(",", "") == expected else "Verbose"
    return ("Yes" if correct else "No"), fmt


def reduction(p1: float, p2: float) -> str:
    """How much smaller prompt 2 is than prompt 1 (positive = saving)."""
    if not p1:
        return "-"
    pct = (p1 - p2) / p1 * 100
    color = "green" if pct > 0 else "red"
    return f"[{color}]{pct:+.1f}%[/{color}]"


def fmt_cost(value: Optional[float]) -> str:
    return "-" if value is None else f"${value:.6f}"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_comparison(provider: str, model: str, rows: list[dict]) -> None:
    estimated = any(r["result"].estimated for r in rows)
    title = f"Prompt Comparison  |  {provider.upper()}  |  {model}"
    if estimated:
        title += "  (offline estimate)"

    table = Table(title=title, box=box.ROUNDED, header_style="bold cyan", show_lines=True)
    for col in ["Prompt", "Words", "Input Tokens", "Output Tokens", "Reasoning Tokens",
                "Total Tokens", "Latency (ms)", "Est. Cost", "Correct?", "Output Format"]:
        table.add_column(col, justify="left" if col == "Prompt" else "right")

    for r in rows:
        res: LLMResult = r["result"]
        table.add_row(
            r["label"], str(r["words"]), str(res.input_tokens), str(res.output_tokens),
            str(res.reasoning_tokens), str(res.total_tokens), f"{res.latency_ms:.0f}",
            fmt_cost(r["cost"]), r["correct"], r["format"],
        )

    if len(rows) == 2:
        a, b = rows[0], rows[1]
        ra, rb = a["result"], b["result"]
        table.add_row(
            "[bold]Saved by P2[/bold]",
            reduction(a["words"], b["words"]),
            reduction(ra.input_tokens, rb.input_tokens),
            reduction(ra.output_tokens, rb.output_tokens),
            reduction(ra.reasoning_tokens, rb.reasoning_tokens),
            reduction(ra.total_tokens, rb.total_tokens),
            reduction(ra.latency_ms, rb.latency_ms),
            reduction(a["cost"] or 0, b["cost"] or 0) if a["cost"] is not None else "-",
            "", "",
        )

    console.print()
    console.print(table)

    for r in rows:
        console.print(Panel(r["result"].text or "[dim](empty response)[/dim]",
                            title=f"Output - {r['label']} ({r['file']})", border_style="blue"))


def save_csv(records: list[dict]) -> Path:
    out_dir = BASE_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    providers = [p.strip() for p in os.getenv("LLM_PROVIDER", "ollama").split(",") if p.strip()]
    num_a, num_b = os.getenv("NUM_A", "25"), os.getenv("NUM_B", "17")
    variables = {"NUM_A": num_a, "NUM_B": num_b}
    expected = expected_answer(num_a, num_b)

    prompts = [
        ("Prompt 1", os.getenv("PROMPT_1_NAME", "unstructured_prompt")),
        ("Prompt 2", os.getenv("PROMPT_2_NAME", "structured_prompt")),
    ]

    try:
        loaded = [(label, load_prompt(name, variables)) for label, name in prompts]
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return 1

    console.print(Panel(f"Task: add [bold]{num_a}[/bold] and [bold]{num_b}[/bold]   "
                        f"| Expected answer: [bold]{expected}[/bold]   "
                        f"| Providers: {', '.join(providers)}\n"
                        f"Prompt 1 = {prompts[0][1]}.md   |   Prompt 2 = {prompts[1][1]}.md",
                        title="Prompt Comparison App", border_style="magenta"))

    csv_records = []
    for provider in providers:
        try:
            llm = create_llm(provider)
        except Exception as e:  # missing key, unknown provider, missing SDK...
            console.print(f"[red]Skipping {provider}: {e}[/red]")
            continue

        in_price = env_float(f"{provider.upper()}_INPUT_PRICE_PER_1M")
        out_price = env_float(f"{provider.upper()}_OUTPUT_PRICE_PER_1M")

        rows = []
        for (label, prompt_text), (_, name) in zip(loaded, prompts):
            with console.status(f"[cyan]{provider}: running {label}..."):
                try:
                    result = llm.generate(prompt_text)
                except Exception as e:
                    console.print(f"[red]{provider} failed on {label}: {e}[/red]")
                    break
            correct, fmt = check_accuracy(result.text, expected)
            row = {
                "label": label, "file": name, "words": len(prompt_text.split()), "result": result,
                "cost": result.cost(in_price, out_price), "correct": correct, "format": fmt,
            }
            rows.append(row)
            csv_records.append({
                "provider": provider, "model": result.model, "prompt": label,
                "prompt_file": name,
                "prompt_words": row["words"], "prompt_chars": len(prompt_text),
                "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
                "reasoning_tokens": result.reasoning_tokens, "total_tokens": result.total_tokens,
                "latency_ms": round(result.latency_ms, 1), "est_cost_usd": row["cost"],
                "correct": correct, "output_format": fmt, "estimated": result.estimated,
                "output": result.text,
            })

        if rows:
            render_comparison(provider, llm.model, rows)

    if not csv_records:
        console.print("[yellow]No results. Check your .env settings.[/yellow]")
        return 1

    path = save_csv(csv_records)
    console.print(f"\n[green]Results saved to {path.relative_to(BASE_DIR)}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
