# Prompt Comparison App

Compares an **unstructured** prompt against a **structured** prompt for the same task
(adding two numbers) and shows the difference in tokens, latency, cost and accuracy.

## Project structure

```
prompt-comparison-app/
├── llm_client.py            # Unified client: OpenAI (Responses API), Gemini, Anthropic, Ollama
├── app.py                   # Runs both prompts and renders the comparison table
├── repeatability_test.py    # Runs both prompts REPEAT_COUNT times, tables the responses
├── .env                     # Your config (API keys, models, prompt names, inputs)
├── .env.example             # Template of .env
├── prompts/
│   ├── unstructured_prompt.md
│   └── structured_prompt.md
├── requirements.txt
├── LINKEDIN_ARTICLE.md
└── results/                 # CSV of every run (created automatically)
```

## Setup

```bash
python -m venv .venv

.venv\Scripts\Activate.ps1          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Open `.env` and set:

1. `LLM_PROVIDER` to `openai`, `gemini`, `anthropic`, `ollama`, or a comma list like `openai,gemini,ollama`.
2. The matching `*_API_KEY` and `*_MODEL` (check your provider's docs for current model names).
3. Optional: `*_INPUT_PRICE_PER_1M` / `*_OUTPUT_PRICE_PER_1M` (USD) to see estimated cost.

No API key yet? Use `LLM_PROVIDER=ollama` to run against a model hosted locally by
[Ollama](https://ollama.com) — install it, pull a model (e.g. `ollama run qwen2.5-coder:3b`),
then set `OLLAMA_MODEL` (and `OLLAMA_BASE_URL` if it's not on the default `localhost:11434`).

## Run

```bash
python app.py
```

## Table columns

| Column | Meaning |
|---|---|
| Words | Size of the prompt text |
| Input Tokens | Tokens the provider billed for the prompt |
| Output Tokens | Tokens billed for the answer (includes reasoning tokens) |
| Reasoning Tokens | "Thinking" tokens used by reasoning models (subset of output) |
| Total Tokens | Input + output |
| Latency (ms) | Wall-clock time of the API call (varies run to run) |
| Est. Cost | Uses the prices you set in `.env` |
| Correct? | Answer contains the expected sum (or an error for invalid input) |
| Output Format | `Clean` = only the answer, `Verbose` = extra text around it |
| Saved by P2 | % reduction of Prompt 2 vs Prompt 1 (green = saving) |

## Try the early-exit rule

Set `NUM_B=abc` in `.env` and run again. The structured prompt's first rule makes the model
reply `ERROR: invalid input` immediately instead of attempting the calculation.

## Repeatability test

`repeatability_test.py` calls Prompt 1 and Prompt 2 back-to-back, `REPEAT_COUNT` times each
(against whichever provider `LLM_PROVIDER` resolves to — the first one, if it's a comma list),
and prints a table of every run's responses side by side. Useful for eyeballing how consistent
each prompt's output is across repeated calls to the same model.

```bash
python repeatability_test.py
```

Set `REPEAT_COUNT` in `.env` to change how many times it runs (default `10`).

## Use your own prompts

Drop any `.md` file into `prompts/`, use `{{NUM_A}}` / `{{NUM_B}}` placeholders if needed,
and point `PROMPT_1_NAME` / `PROMPT_2_NAME` at the file names (without `.md`).

## Notes

* `TEMPERATURE` is empty by default because some reasoning models reject it. Set `0` for
  non-reasoning models to get more repeatable results.
* If an OpenAI reasoning model returns an empty answer, raise `MAX_OUTPUT_TOKENS` or set
  `OPENAI_REASONING_EFFORT=low`.
* Never commit `.env` with real keys.
