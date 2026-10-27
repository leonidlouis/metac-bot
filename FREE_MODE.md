# Free-only mode ($0 LLM spend)

This fork runs **only on free OpenRouter models** (ids ending in `:free`).

- `free_guard.py` patches litellm/forecasting-tools so any non-`openrouter/*:free`
  model (paid OpenRouter, OpenAI, Anthropic, Metaculus proxy, Exa, Perplexity,
  Responses API) raises `PaidModelBlocked` before a request is sent. OpenRouter
  fallback lists are checked too. Paid-provider env keys are removed at startup.
- Models (see top of `main.py`; first one currently listed as free is used, the
  rest are OpenRouter fallbacks inside the same request):
  - forecaster: `thinkingmachines/inkling:free` -> `nvidia/nemotron-3-ultra-550b-a55b:free` -> `thinkingmachines/inkling-small:free`
  - parser: `thinkingmachines/inkling-small:free` -> `nvidia/nemotron-3.5-super-vl:free` -> `google/gemma-4-31b-it:free`
  - research: AskNews (free bot-maker plan) if `ASKNEWS_CLIENT_ID`/`ASKNEWS_SECRET`
    are set, otherwise Google News RSS headlines (no LLM call).
- Budget: OpenRouter allows 50 free requests/day (20/min) for accounts that never
  bought credits. Each run reads the key's `free_model_daily_requests` counter
  (`GET /api/v1/key`, not itself a model request) and caps its own calls at
  `min(MAX_LLM_CALLS_PER_RUN, remaining_today - DAILY_CALL_RESERVE)`.
  A normal question costs 2 requests (1 forecast + 1 parse); conditionals ~8.
  Questions that don't fit are skipped and retried on a later run.
- 1 forecast per question (no 5x ensemble), no research summary, 1 parse sample.
- Free model ids change often. If a run logs "None of these free models is
  currently listed", update the lists in `main.py` (or set `FORECASTER_MODELS` /
  `PARSER_MODELS` env vars, comma-separated, each `openrouter/<id>:free`).
- Never add credits/auto-top-up to the OpenRouter account unless you decide to:
  with a $0 balance OpenRouter cannot bill even if a paid model were requested.
