"""
Backtest step 2: forecast resolved questions with free OpenRouter models.
Never publishes anything to Metaculus. 1 LLM request per question (local
regex parse; no parser call). Honours free_guard's daily budget so the live
hourly bot keeps DAILY_CALL_RESERVE free requests.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import free_guard  # noqa: E402
from free_guard import CallBudgetExhausted  # noqa: E402
from prompts import baseline_prompt, parse_probability, structured_prompt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backtest")


def dt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def make_prompt(variant: str, q: dict) -> str:
    today = dt(q["open_time"])
    if variant == "baseline":
        return baseline_prompt(q, today)
    if variant == "structured":
        return structured_prompt(q, today, dt(q["close_time"]), dt(q.get("scheduled_resolve_time")))
    raise ValueError(variant)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--models", default="", help="comma list of openrouter/<id>:free")
    ap.add_argument("--questions", default="backtest/data/questions.json")
    ap.add_argument("--skip", default="", help="comma list of post_ids already done")
    ap.add_argument("--ids", default="", help="only these post_ids (comma list)")
    ap.add_argument("--max-questions", type=int, default=100)
    ap.add_argument("--out", default="backtest/out/results.jsonl")
    args = ap.parse_args()

    free_guard.install()
    from forecasting_tools import GeneralLlm  # after install
    from main import FORECASTER_MODELS, build_free_llm

    try:
        free_ids = free_guard.available_free_model_ids()
    except Exception:  # noqa: BLE001
        free_ids = None
    models = [m.strip() for m in args.models.split(",") if m.strip()] or FORECASTER_MODELS
    llm: GeneralLlm = build_free_llm(models, free_ids, temperature=0.3, timeout=300, allowed_tries=1)
    budget = free_guard.compute_run_budget()
    log.info(f"backtest budget this run: {budget.limit} requests")

    qs = json.load(open(args.questions))
    skip = {int(x) for x in args.skip.split(",") if x.strip()}
    only = {int(x) for x in args.ids.split(",") if x.strip()}
    qs = [q for q in qs if q["post_id"] not in skip and (not only or q["post_id"] in only)]
    qs = qs[: args.max_questions]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = 0
    with open(args.out, "a") as f:
        for q in qs:
            if budget.remaining < 1:
                log.info("budget used up; stopping")
                break
            t0 = time.time()
            try:
                text = await llm.invoke(make_prompt(args.variant, q))
            except CallBudgetExhausted:
                break
            except Exception as e:  # noqa: BLE001
                log.warning(f"q{q['post_id']} failed: {type(e).__name__}: {str(e)[:200]}")
                f.write(json.dumps({"post_id": q["post_id"], "variant": args.variant,
                                    "model": llm.model, "error": type(e).__name__}) + "\n")
                f.flush()
                continue
            p = parse_probability(text)
            rec = {"post_id": q["post_id"], "variant": args.variant, "model": llm.model,
                   "p": p, "resolution": q["resolution"], "secs": round(time.time() - t0),
                   "reasoning": text}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done += 1
            log.info(f"q{q['post_id']} p={p} res={q['resolution']} ({done})")
            await asyncio.sleep(3)
    log.info(f"done {done}; LLM calls used {budget.used}/{budget.limit}")


if __name__ == "__main__":
    asyncio.run(main())
