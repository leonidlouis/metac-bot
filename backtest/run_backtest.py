"""
Backtest step 2: forecast resolved questions with free OpenRouter models.
Never publishes anything to Metaculus. Honours free_guard's daily budget so the
live hourly bot keeps DAILY_CALL_RESERVE free requests.

Variants (--variant): baseline, structured, baserate, redteam  -> 1 LLM call/Q
                      ensemble                                   -> 2 LLM calls/Q
  ensemble runs the structured prompt twice on two different free models (see
  --ensemble-models) and averages the parsed probabilities. Budget it as 2x.

All answers parsed locally (parse_probability); no separate parser call.
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
from prompts import build_prompt, parse_probability  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backtest")


def dt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def make_prompt(variant: str, q: dict, research_field: str = "research") -> str:
    q = dict(q, research=q.get(research_field) or "")
    today = dt(q["open_time"])
    return build_prompt(variant, q, today, dt(q["close_time"]), dt(q.get("scheduled_resolve_time")))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--models", default="", help="comma list of openrouter/<id>:free (forecaster)")
    ap.add_argument("--ensemble-models", default="",
                    help="comma list of two openrouter/<id>:free for --variant ensemble")
    ap.add_argument("--questions", default="backtest/data/questions.json")
    ap.add_argument("--skip", default="", help="comma list of post_ids already done")
    ap.add_argument("--ids", default="", help="only these post_ids (comma list)")
    ap.add_argument("--max-questions", type=int, default=100)
    ap.add_argument("--out", default="backtest/out/results.jsonl")
    ap.add_argument("--research", default="research", help="research (v2) or research_v1")
    args = ap.parse_args()

    free_guard.install()
    from forecasting_tools import GeneralLlm  # after install
    from main import FORECASTER_MODELS, build_free_llm

    try:
        free_ids = free_guard.available_free_model_ids()
    except Exception:  # noqa: BLE001
        free_ids = None
    models = [m.strip() for m in args.models.split(",") if m.strip()] or FORECASTER_MODELS

    ensemble = args.variant == "ensemble"
    llms = []
    if ensemble:
        ens = [m.strip() for m in args.ensemble_models.split(",") if m.strip()]
        if len(ens) < 2:
            # default: two distinct cheap free models from FORECASTER_MODELS
            ens = (models + FORECASTER_MODELS)[:2]
        for m in ens[:2]:
            llms.append(build_free_llm([m], free_ids, temperature=0.3, timeout=300, allowed_tries=1))
        log.info(f"ensemble models: {[l.model for l in llms]}")
    else:
        llms = [build_free_llm(models, free_ids, temperature=0.3, timeout=300, allowed_tries=1)]

    budget = free_guard.compute_run_budget()
    calls_per_q = 2 if ensemble else 1
    log.info(f"backtest budget this run: {budget.limit} requests ({calls_per_q}/Q -> "
             f"~{budget.limit // calls_per_q} Qs)")

    qs = json.load(open(args.questions))
    skip = {int(x) for x in args.skip.split(",") if x.strip()}
    only = {int(x) for x in args.ids.split(",") if x.strip()}
    qs = [q for q in qs if q["post_id"] not in skip and (not only or q["post_id"] in only)]
    qs = qs[: args.max_questions]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rsuffix = "" if args.research == "research" else "+r1"
    done = 0
    with open(args.out, "a") as f:
        for q in qs:
            if budget.remaining < calls_per_q:
                log.info("budget would be exceeded; stopping")
                break
            t0 = time.time()
            prompt = make_prompt("structured" if ensemble else args.variant, q, args.research)
            try:
                if ensemble:
                    texts = await asyncio.gather(*[l.invoke(prompt) for l in llms])
                    ps = [parse_probability(t) for t in texts]
                    ps = [p for p in ps if p is not None]
                    p = sum(ps) / len(ps) if ps else None
                    text = "\n\n===ENSEMBLE MEMBER===\n\n".join(texts)
                    model = "+".join(l.model.split("/")[-1] for l in llms)
                else:
                    text = await llms[0].invoke(prompt)
                    p = parse_probability(text)
                    model = llms[0].model
            except CallBudgetExhausted:
                break
            except Exception as e:  # noqa: BLE001
                log.warning(f"q{q['post_id']} failed: {type(e).__name__}: {str(e)[:200]}")
                f.write(json.dumps({"post_id": q["post_id"], "variant": args.variant,
                                    "model": llms[0].model, "error": type(e).__name__}) + "\n")
                f.flush()
                continue
            rec = {"post_id": q["post_id"], "variant": args.variant + rsuffix, "model": model,
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
