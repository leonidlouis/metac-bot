"""
Free-only guard for this bot ($0 LLM spend rule).

What it enforces, process-wide, once `install()` has been called:
  1. Every LLM call must go to an OpenRouter model whose id ends in ':free'
     (e.g. 'openrouter/thinkingmachines/inkling:free'). Anything else -- paid
     OpenRouter models, OpenAI/Anthropic direct, the Metaculus proxy, Exa,
     Perplexity, the Responses API -- raises PaidModelBlocked BEFORE any
     network request is made. OpenRouter fallback lists (extra_body.models)
     are checked the same way.
  2. A hard cap on LLM requests per run, derived from OpenRouter's own
     free-model daily counter (GET /api/v1/key -> free_model_daily_requests,
     which does not itself count as a model request). Once the cap is hit,
     further calls raise CallBudgetExhausted instead of being sent.
  3. Paid-provider API keys are removed from the environment so no library
     default can silently pick them up.

AskNews (free bot-maker plan) is the only non-OpenRouter source allowed, and
only if ASKNEWS_CLIENT_ID/ASKNEWS_SECRET are set.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

# Env vars that could make a library route to a paid provider.
_PAID_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "PERPLEXITY_API_KEY",
    "EXA_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"


class PaidModelBlocked(RuntimeError):
    pass


class CallBudgetExhausted(RuntimeError):
    pass


def is_free_model(model: str) -> bool:
    return (
        isinstance(model, str)
        and model.startswith("openrouter/")
        and model.endswith(":free")
        and model != "openrouter/free"  # random router; we pin models instead
    )


def assert_free_model(model: str) -> None:
    if not is_free_model(model):
        raise PaidModelBlocked(
            f"Blocked non-free model '{model}'. This bot may only call "
            "'openrouter/<id>:free' models ($0 spend rule)."
        )


def _get_json(url: str, api_key: str | None = None, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "metac-bot"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def available_free_model_ids() -> set[str]:
    """OpenRouter model ids (without 'openrouter/' prefix) that are priced at 0."""
    data = _get_json(OPENROUTER_MODELS_URL)["data"]
    free = set()
    for m in data:
        p = m.get("pricing") or {}
        priced_zero = all(
            str(p.get(k, "0")) in ("0", "0.0") for k in ("prompt", "completion", "request")
        )
        if m["id"].endswith(":free") and priced_zero:
            free.add(m["id"])
    return free


def pick_available(candidates: list[str], free_ids: set[str] | None) -> list[str]:
    """Keep only candidates that are free AND currently listed on OpenRouter."""
    for c in candidates:
        assert_free_model(c)
    if free_ids is None:  # couldn't reach the models list; trust the static list
        return list(candidates)
    return [c for c in candidates if c.removeprefix("openrouter/") in free_ids]


class CallBudget:
    def __init__(self, limit: int) -> None:
        self.limit = max(0, int(limit))
        self.used = 0
        self._lock = asyncio.Lock()

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    async def take(self, model: str) -> None:
        async with self._lock:
            if self.used >= self.limit:
                raise CallBudgetExhausted(
                    f"LLM call budget for this run is used up ({self.used}/{self.limit}); "
                    f"not calling {model}."
                )
            self.used += 1
            logger.info(f"[free_guard] LLM call {self.used}/{self.limit} -> {model}")


BUDGET: CallBudget = CallBudget(0)


def compute_run_budget() -> CallBudget:
    """
    Per-run cap = min(MAX_LLM_CALLS_PER_RUN, OpenRouter free requests left today
    - DAILY_CALL_RESERVE). Also honours MAX_LLM_CALLS_PER_DAY as a self-imposed
    daily ceiling (default 50 = OpenRouter's limit for accounts that never bought
    credits).
    """
    per_run = int(os.getenv("MAX_LLM_CALLS_PER_RUN", "24"))
    reserve = int(os.getenv("DAILY_CALL_RESERVE", "3"))
    per_day = int(os.getenv("MAX_LLM_CALLS_PER_DAY", "50"))
    key = os.getenv("OPENROUTER_API_KEY")
    limit = min(per_run, 12)  # conservative if we can't read the counter
    try:
        info = _get_json(OPENROUTER_KEY_URL, api_key=key)["data"]
        fmd = info.get("free_model_daily_requests") or {}
        used_today = int(fmd.get("used", 0))
        remaining_today = int(fmd.get("remaining", fmd.get("limit", 50) - used_today))
        remaining_today = min(remaining_today, per_day - used_today)
        limit = max(0, min(per_run, remaining_today - reserve))
        logger.info(
            f"[free_guard] OpenRouter free requests today: used={used_today}, "
            f"daily_limit={fmd.get('limit')}, remaining={fmd.get('remaining')}; "
            f"self-imposed daily cap={per_day}; this run may use {limit}."
        )
    except Exception as e:  # never print the key
        logger.warning(
            f"[free_guard] Could not read OpenRouter key status ({type(e).__name__}); "
            f"using conservative per-run cap of {limit}."
        )
    global BUDGET
    BUDGET = CallBudget(limit)
    return BUDGET


def _check_request(kwargs: dict) -> str:
    model = kwargs.get("model", "")
    assert_free_model(model)
    base_url = kwargs.get("base_url") or kwargs.get("api_base")
    if base_url and "openrouter.ai" not in str(base_url):
        raise PaidModelBlocked(f"Blocked custom base_url '{base_url}'.")
    extra = kwargs.get("extra_body") or {}
    for fb in extra.get("models", []) or []:
        assert_free_model(fb if fb.startswith("openrouter/") else f"openrouter/{fb}")
    return model


def install() -> None:
    """Patch litellm + forecasting_tools so only free OpenRouter calls can happen."""
    for var in _PAID_ENV_VARS:
        if os.environ.pop(var, None):
            logger.warning(f"[free_guard] Removed {var} from environment ($0 rule).")

    import litellm
    from forecasting_tools.ai_models import general_llm as gl

    original_acompletion = gl.acompletion

    async def guarded_acompletion(*args, **kwargs):
        if args:
            raise PaidModelBlocked("Positional args to acompletion are not allowed.")
        model = _check_request(kwargs)
        await BUDGET.take(model)
        return await original_acompletion(*args, **kwargs)

    async def blocked_async(*args, **kwargs):
        raise PaidModelBlocked("This LLM entry point is disabled ($0 rule).")

    def blocked_sync(*args, **kwargs):
        raise PaidModelBlocked("This LLM entry point is disabled ($0 rule).")

    gl.acompletion = guarded_acompletion
    gl.aresponses = blocked_async
    litellm.acompletion = guarded_acompletion
    litellm.completion = blocked_sync
    litellm.responses = blocked_sync
    litellm.aresponses = blocked_async

    # Block Exa and the Metaculus proxy at the GeneralLlm level too.
    async def blocked_method(self, *args, **kwargs):
        raise PaidModelBlocked(f"Blocked provider for model '{self.model}'.")

    gl.GeneralLlm._call_exa_model = blocked_method

    original_init = gl.GeneralLlm.__init__

    def guarded_init(self, model, *args, **kwargs):
        allowed_asknews = model.startswith("asknews/") and bool(
            os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")
        )
        if not (is_free_model(model) or allowed_asknews):
            raise PaidModelBlocked(f"Refusing to construct LLM for '{model}'.")
        original_init(self, model, *args, **kwargs)

    gl.GeneralLlm.__init__ = guarded_init
    logger.info("[free_guard] Installed: only openrouter/*:free models can be called.")
