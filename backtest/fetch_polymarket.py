"""
Backtest data source (no auth, no LLM): resolved Polymarket binary markets.

Why: Metaculus's API redacts resolutions/criteria of resolved questions for
bot accounts, so we can't pastcast on Metaculus questions with the bot token.
Polymarket's public API gives resolved yes/no questions with full rules text,
plus the market price at forecast time as a strong "crowd" baseline.

Selection (all decided with information available at forecast time T):
- event ended 2026-09-01..2026-10-27 (after the free models were released
  ~July 2026, so outcomes shouldn't be in training data), cleanly resolved;
- topical tags only (politics/geopolitics/economy/finance/tech/AI/science/...),
  no sports, esports, crypto price ladders, tweet counts;
- forecast time T = 10 days before the market end date (or 2 days after it
  opened if later; horizon >= 3 days); market price at T within [0.03, 0.97];
- at most 2 markets per event (seeded random), to limit correlated siblings.
Research = Google News headlines published before T (same as live bot).
Writes backtest/data/questions.json in the same schema as fetch_questions.py.
"""
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fetch_questions import headlines  # noqa: E402

GAMMA = "https://gamma-api.polymarket.com/events"
CLOB = "https://clob.polymarket.com/prices-history"
TAGS = ["politics", "geopolitics", "economy", "finance", "tech", "ai", "science",
        "world", "elections", "business", "fed-rates", "economic-policy", "world-elections",
        "middle-east", "trade-war", "courts", "culture"]
BAD_TAGS = {"sports", "esports", "crypto", "games", "bitcoin", "ethereum", "solana", "nfl", "nba",
            "soccer", "tennis", "mlb", "tweet markets", "mention markets", "crypto prices"}
BAD_WORDS = ["up or down", "tweets", "o/u", "spread", " vs. ", "bitcoin", "ethereum", "solana", "xrp",
             "dogecoin", "say \"", "says \"", "mention"]


def get(url, params=None):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for a in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            print("  retry", a, type(e).__name__)
            time.sleep(3 * (a + 1))
    return None


def pdt(s):
    if not s:
        return None
    s = s.replace("Z", "+00:00").replace(" ", "T")
    if s.endswith("+00"):
        s += ":00"
    return datetime.fromisoformat(s)


def price_at(token, t: datetime):
    h = get(CLOB, {"market": token, "startTs": int((t - timedelta(days=2)).timestamp()),
                   "endTs": int(t.timestamp()), "fidelity": 60})
    pts = (h or {}).get("history") or []
    pts = [p for p in pts if p["t"] <= t.timestamp()]
    return pts[-1]["p"] if pts else None


def main(out="backtest/data/questions.json", per_event=2, max_q=160):
    rnd = random.Random(42)
    events = {}
    for tag in TAGS:
        for off in range(0, 600, 100):
            d = get(GAMMA, {"closed": "true", "tag_slug": tag, "limit": 100, "offset": off,
                            "end_date_min": "2026-09-01T00:00:00Z", "end_date_max": "2026-10-27T00:00:00Z",
                            "order": "volume", "ascending": "false"})
            if not d:
                break
            for e in d:
                events[e["id"]] = e
            if len(d) < 100:
                break
        print(f"tag {tag}: total events so far {len(events)}")
    rows = []
    for e in events.values():
        tags = {str(t.get("label", "")).lower() for t in e.get("tags", [])} | {
            str(t.get("slug", "")).lower() for t in e.get("tags", [])}
        if tags & BAD_TAGS or any(w in e["title"].lower() for w in BAD_WORDS):
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
            if m.get("umaResolutionStatus") not in (None, "resolved") or float(m.get("volumeNum") or 0) < 20000:
                continue
            if any(w in (m.get("question") or "").lower() for w in BAD_WORDS):
                continue
            end, start = pdt(m.get("endDate")), pdt(m.get("startDate"))
            if not end or not start or not (datetime(2026, 9, 1, tzinfo=timezone.utc) <= end <= datetime(2026, 10, 27, tzinfo=timezone.utc)):
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
            p = price_at(tok, t)
            time.sleep(0.2)
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
    rnd.shuffle(rows)
    rows = rows[:max_q]
    print(f"kept {len(rows)} markets from {len({r['event'] for r in rows})} events; yes-rate "
          f"{sum(r['resolution'] for r in rows)/max(1,len(rows)):.2f}")
    for i, r in enumerate(rows):
        t = datetime.fromisoformat(r["open_time"])
        r["research"] = headlines(r["title"], before=t, after=t - timedelta(days=45))
        time.sleep(1.0)
        if i % 20 == 0:
            print("  research", i)
    json.dump(rows, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
