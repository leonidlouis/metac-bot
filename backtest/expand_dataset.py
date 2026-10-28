"""
Expand the backtest dataset OFFLINE (no LLM / no paid API).

Pulls more resolved binary Polymarket markets (public Gamma + CLOB APIs, free),
over a wider window than the original questions.json, dedupes against the
existing dataset by post_id, and generates leakage-limited research headlines
(research.py v2 -> Google News RSS, headline fetch only, NO LLM).

Writes the UNION (existing + new) to backtest/data/questions_expanded.json so the
original questions.json is never touched. Report-only prints counts + yes-rate.

Selection mirrors fetch_polymarket.py:
- cleanly resolved Yes/No markets, topical tags only (no sports/esports/crypto
  price ladders / tweet markets);
- forecast time T = end-10d (>= start+2d, horizon >= 3d);
- crowd baseline = CLOB price at T, kept only if in [0.03, 0.97];
- <= per_event markets per event.
Wider window: end_date in [--end-min, --end-max] (default 2026-07-01..2026-10-18),
so outcomes are still post model-release (~Jul 2026) but we get many more Qs.
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fetch_polymarket as fp  # get, pdt, price_at, TAGS, BAD_TAGS, BAD_WORDS, GAMMA
from research import news_headlines  # v2 headlines, no LLM


def fetch_events(end_min, end_max, min_volume, verbose=True):
    events = {}
    for tag in fp.TAGS:
        for off in range(0, 800, 100):
            d = fp.get(fp.GAMMA, {"closed": "true", "tag_slug": tag, "limit": 100, "offset": off,
                                  "end_date_min": end_min, "end_date_max": end_max,
                                  "order": "volume", "ascending": "false"})
            if not d:
                break
            for e in d:
                events[e["id"]] = e
            if len(d) < 100:
                break
        if verbose:
            print(f"tag {tag}: total events so far {len(events)}", flush=True)
    return events


def market_rows(events, min_volume, end_min_dt, end_max_dt, per_event, exclude_ids, rnd):
    rows = []
    for e in events.values():
        tags = {str(t.get("label", "")).lower() for t in e.get("tags", [])} | {
            str(t.get("slug", "")).lower() for t in e.get("tags", [])}
        if tags & fp.BAD_TAGS or any(w in e["title"].lower() for w in fp.BAD_WORDS):
            continue
        cands = []
        for m in e.get("markets", []):
            try:
                outs = json.loads(m.get("outcomes") or "[]")
                prices = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
                toks = json.loads(m.get("clobTokenIds") or "[]")
            except Exception:  # noqa: BLE001
                continue
            if outs != ["Yes", "No"] or sorted(prices) != [0.0, 1.0] or len(toks) != 2:
                continue
            if m.get("umaResolutionStatus") not in (None, "resolved") or float(m.get("volumeNum") or 0) < min_volume:
                continue
            if any(w in (m.get("question") or "").lower() for w in fp.BAD_WORDS):
                continue
            if int(m["id"]) in exclude_ids:
                continue
            end, start = fp.pdt(m.get("endDate")), fp.pdt(m.get("startDate"))
            if not end or not start or not (end_min_dt <= end <= end_max_dt):
                continue
            t = max(end - timedelta(days=10), start + timedelta(days=2))
            if (end - t).days < 3:
                continue
            cands.append((m, t, end, toks[0], 1 if prices[0] == 1.0 else 0))
        rnd.shuffle(cands)
        kept = 0
        for m, t, end, tok, res in cands:
            if kept >= per_event:
                break
            p = fp.price_at(tok, t)
            time.sleep(0.15)
            if p is None or not 0.03 <= p <= 0.97:
                continue
            kept += 1
            rows.append({
                "post_id": int(m["id"]), "tournament": "polymarket", "event": e["title"],
                "title": m["question"],
                "background": (e.get("description") or "")[:3000] if e.get("description") != m.get("description") else "",
                "resolution_criteria": m.get("description") or "",
                "fine_print": "",
                "open_time": t.isoformat(), "close_time": end.isoformat(),
                "scheduled_resolve_time": end.isoformat(),
                "resolution": res,
                "cp_at_open_plus": {"source": "polymarket_price_at_T", "p": p},
                "volume": float(m.get("volumeNum") or 0),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing", default="backtest/data/questions.json")
    ap.add_argument("--out", default="backtest/data/questions_expanded.json")
    ap.add_argument("--end-min", default="2026-07-01T00:00:00Z")
    ap.add_argument("--end-max", default="2026-10-18T00:00:00Z")
    ap.add_argument("--min-volume", type=float, default=10000)
    ap.add_argument("--per-event", type=int, default=3)
    ap.add_argument("--target-new", type=int, default=440)
    ap.add_argument("--research-sleep", type=float, default=0.5)
    args = ap.parse_args()

    rnd = random.Random(42)
    existing = json.load(open(args.existing))
    exclude_ids = {int(q["post_id"]) for q in existing}
    print(f"existing dataset: {len(existing)} Qs", flush=True)

    end_min_dt = fp.pdt(args.end_min)
    end_max_dt = fp.pdt(args.end_max)
    events = fetch_events(args.end_min, args.end_max, args.min_volume)
    print(f"candidate events: {len(events)}", flush=True)

    rows = market_rows(events, args.min_volume, end_min_dt, end_max_dt,
                       args.per_event, exclude_ids, rnd)
    # dedupe new rows by post_id and title (defensive)
    seen_pid, seen_title, uniq = set(), set(), []
    for r in rows:
        tk = r["title"].strip().lower()
        if r["post_id"] in seen_pid or tk in seen_title:
            continue
        seen_pid.add(r["post_id"])
        seen_title.add(tk)
        uniq.append(r)
    rnd.shuffle(uniq)
    uniq = uniq[: args.target_new]
    print(f"new unique markets kept: {len(uniq)} from "
          f"{len({r['event'] for r in uniq})} events; "
          f"yes-rate {sum(r['resolution'] for r in uniq)/max(1,len(uniq)):.3f}", flush=True)

    # Resume/idempotent: reuse research already computed in a prior out file so a
    # re-run doesn't redo finished headline lookups.
    prior = {}
    if os.path.exists(args.out):
        try:
            for r in json.load(open(args.out)):
                if r.get("post_id") in {u["post_id"] for u in uniq} and r.get("research"):
                    prior[r["post_id"]] = r["research"]
            print(f"resume: reusing research for {len(prior)} already-done new rows", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"resume: could not read prior out ({type(e).__name__}); recomputing all", flush=True)

    # research (headlines only, NO LLM) for the new rows
    got = 0
    for i, r in enumerate(uniq):
        if r["post_id"] in prior:
            r["research"] = prior[r["post_id"]]
            r["research_v1"] = r["research"]
            got += 1
            continue
        t = datetime.fromisoformat(r["open_time"])
        try:
            r["research"] = news_headlines(r["title"], before=t, after=t - timedelta(days=45))
        except Exception as e:  # noqa: BLE001
            print(f"  research err {r['post_id']}: {type(e).__name__}", flush=True)
            r["research"] = ""
        r["research_v1"] = r.get("research", "")
        got += bool(r["research"])
        time.sleep(args.research_sleep)
        if i % 25 == 0:
            print(f"  research {i}/{len(uniq)} (with headlines: {got})", flush=True)
    print(f"new rows with headlines: {got}/{len(uniq)}", flush=True)

    combined = existing + uniq
    with open(args.out, "w") as f:
        json.dump(combined, f, indent=1)
    print(f"wrote {args.out}: {len(combined)} total "
          f"(existing {len(existing)} + new {len(uniq)}); "
          f"combined yes-rate {sum(r['resolution'] for r in combined)/len(combined):.3f}", flush=True)


if __name__ == "__main__":
    main()
