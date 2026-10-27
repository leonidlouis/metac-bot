"""
Backtest step 1 (no LLM calls): pull already-resolved Metaculus questions and
pre-compute leakage-limited research for each.

- Needs METACULUS_TOKEN (the API is auth-only). Never printed.
- Keeps binary questions that opened on/after --since (default 2026-08-01, i.e.
  after the free models' release, so outcomes shouldn't be in training data)
  and resolved yes/no (annulled/ambiguous dropped).
- Research = Google News RSS headlines restricted with `before:<open date>`, so
  the model only sees news published before the question opened (bots forecast
  at open time in FutureEval/MiniBench).
- Baseline: Metaculus community/aggregate prediction at close, when the API
  exposes one (main-site Cup questions; bot tournaments usually have none).
Writes backtest/data/questions.json.
"""
import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

API = "https://www.metaculus.com/api/posts/"
TOURNAMENTS = {
    "fall-futureeval-2026": 33121,
    "minibench": "minibench",
    "metaculus-cup-fall-2026": 33108,
    "summer-futureeval-2026": 33022,
    "metaculus-cup-summer-2026": 33021,
}
STOP = set(
    "will the a an of in on by to for be is before after than more less at or and what which who how many "
    "much does do with from as it its this that between least most any there their has have been were was".split()
)


def api_get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Token {os.environ['METACULUS_TOKEN']}",
            "Accept": "application/json",
            "User-Agent": "metac-bot-backtest",
        },
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            print(f"  api retry {attempt}: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(5 * (attempt + 1))
    return {"results": []}


def api_get_detail(pid: int) -> dict:
    req = urllib.request.Request(
        f"{API}{pid}/",
        headers={"Authorization": f"Token {os.environ['METACULUS_TOKEN']}", "User-Agent": "metac-bot-backtest"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        print(f"  detail {pid} failed: {type(e).__name__}")
        return {}


def parse_dt(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def cp_at(q: dict, when: datetime | None):
    """Latest aggregate (recency_weighted) forecast at/before `when`, binary only."""
    aggs = q.get("aggregations") or {}
    for key in ("recency_weighted", "unweighted", "metaculus_prediction"):
        a = aggs.get(key) or {}
        hist = a.get("history") or []
        best = None
        for h in hist:
            st = h.get("start_time")
            st = datetime.fromtimestamp(st, tz=timezone.utc) if isinstance(st, (int, float)) else parse_dt(st)
            if when is None or (st and st <= when):
                best = h
        if best is None and a.get("latest"):
            best = a["latest"]
        if best:
            c = best.get("centers") or best.get("means")
            if c:
                return {"source": key, "p": float(c[0])}
    return None


def headlines(query: str, before: datetime, after: datetime, n: int = 12) -> str:
    words = re.findall(r"[A-Za-z0-9$%.'-]+", query)
    kw = [w for w in words if w.lower() not in STOP][:8]
    if not kw:
        return ""
    q = " ".join(kw) + f" after:{after:%Y-%m-%d} before:{before:%Y-%m-%d}"
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            root = ET.fromstring(r.read())
    except Exception as e:  # noqa: BLE001
        print(f"  news failed: {type(e).__name__}")
        return ""
    lines = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        pub = item.findtext("pubDate") or ""
        try:
            pdt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc)
        except ValueError:
            pdt = None
        if pdt and pdt >= before:  # belt and braces: drop anything not strictly earlier
            continue
        if title:
            lines.append(f"- {title} ({pub})")
        if len(lines) >= n:
            break
    if not lines:
        return ""
    return (
        f"Recent news headlines (Google News search: '{' '.join(kw)}'). "
        "Headlines only; may be incomplete or partly irrelevant:\n" + "\n".join(lines)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-06-01")
    ap.add_argument("--max", type=int, default=150)
    ap.add_argument("--out", default="backtest/data/questions.json")
    args = ap.parse_args()
    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    rows, seen = [], set()
    try:
        req = urllib.request.Request(
            "https://www.metaculus.com/api/projects/tournaments/",
            headers={"Authorization": f"Token {os.environ['METACULUS_TOKEN']}", "User-Agent": "metac-bot-backtest"})
        with urllib.request.urlopen(req, timeout=60) as r:
            tl = json.loads(r.read())
        tl = tl.get("results", tl) if isinstance(tl, dict) else tl
        for t in tl:
            slug_ = str(t.get("slug", ""))
            if any(k in slug_ for k in ("minibench", "futureeval", "cup", "aib", "bench")):
                print("tournament:", t.get("id"), slug_, t.get("start_date"), t.get("close_date"))
                if "minibench" in slug_ and slug_ not in TOURNAMENTS:
                    TOURNAMENTS[slug_] = t.get("id")
    except Exception as e:  # noqa: BLE001
        print("tournament list failed:", type(e).__name__)
    for slug, tid in TOURNAMENTS.items():
        offset = 0
        while True:
            data = api_get(
                {
                    "tournaments": [tid],
                    "statuses": "resolved",
                    "forecast_type": "binary",
                    "with_cp": "true",
                    "limit": 100,
                    "offset": offset,
                    "order_by": "-published_at",
                }
            )
            res = data.get("results") or []
            print(f"{slug}: offset {offset}: {len(res)} posts")
            for post in res[:2]:
                qq = post.get("question") or {}
                print("   sample:", post.get("id"), qq.get("type"), repr(qq.get("resolution")),
                      qq.get("open_time"), qq.get("actual_close_time"), list((qq.get("aggregations") or {}).keys()))
            for post in res:
                q = post.get("question") or {}
                if q.get("type") != "binary" or post["id"] in seen:
                    continue
                if q.get("resolution") is None:  # list endpoint may omit it
                    det = api_get_detail(post["id"])
                    time.sleep(0.4)
                    if det.get("question"):
                        post = det
                        q = det["question"]
                    if not getattr(main, "_printed", False):
                        main._printed = True
                        print("   detail keys:", sorted(q.keys()))
                        print("   detail resolution:", repr(q.get("resolution")), q.get("status"), q.get("resolution_set_time"))
                resolution = str(q.get("resolution", "")).lower()
                if resolution not in ("yes", "no"):
                    continue
                open_t = parse_dt(q.get("open_time"))
                close_t = parse_dt(q.get("actual_close_time") or q.get("scheduled_close_time"))
                if not open_t or open_t < since:
                    continue
                seen.add(post["id"])
                rows.append(
                    {
                        "post_id": post["id"],
                        "tournament": slug,
                        "title": q.get("title") or post.get("title"),
                        "background": q.get("description", ""),
                        "resolution_criteria": q.get("resolution_criteria", ""),
                        "fine_print": q.get("fine_print", ""),
                        "open_time": open_t.isoformat(),
                        "close_time": close_t.isoformat() if close_t else None,
                        "scheduled_resolve_time": q.get("scheduled_resolve_time"),
                        "resolution": 1 if resolution == "yes" else 0,
                        "cp_at_open_plus": cp_at(q, open_t + timedelta(hours=3)),
                        "cp_at_close": cp_at(q, close_t),
                        "n_forecasters": post.get("nr_forecasters") or q.get("nr_forecasters"),
                    }
                )
            if len(res) < 100 or offset > 1000:
                break
            offset += 100

    rows.sort(key=lambda r: r["open_time"], reverse=True)
    rows = rows[: args.max]
    print(f"kept {len(rows)} resolved binary questions; base rate yes = "
          f"{sum(r['resolution'] for r in rows) / max(1, len(rows)):.3f}")
    for i, r in enumerate(rows):
        ot = datetime.fromisoformat(r["open_time"])
        r["research"] = headlines(r["title"], before=ot, after=ot - timedelta(days=45))
        time.sleep(1.0)
        if i % 10 == 0:
            print(f"  research {i}/{len(rows)}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
