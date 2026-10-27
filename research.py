"""Free, no-LLM research: Google News RSS headlines (v2).

v2 vs the template version: Unicode-aware keywords (e.g. 'Grüne', 'Kršiak'),
a fallback to a shorter query when the long one finds little, de-duplication,
and an optional `before`/`after` date window (used by the backtest so the model
only sees news published before the forecast time).
"""
import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

STOP = set(
    "will the a an of in on by to for be is before after than more less at or and what which who how many "
    "much does do with from as it its this that between least most any there their has have been were was "
    "next win wins winning get gets reach hit end new according".split()
)
_WORD = re.compile(r"[^\W_][\w$%.'’-]*", re.UNICODE)


def _keywords(text: str) -> list[str]:
    words = [w.strip(".'’-") for w in _WORD.findall(text or "")]
    out = []
    for w in words:
        if not w or w.lower() in STOP or re.fullmatch(r"\d{1,2}", w):
            continue
        if w.lower() not in (x.lower() for x in out):
            out.append(w)
    return out


def _fetch(q: str) -> list[tuple[str, str, datetime | None]]:
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            root = ET.fromstring(r.read())
    except Exception as e:  # noqa: BLE001
        logger.warning(f"News lookup failed: {type(e).__name__}")
        return []
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        pub = (it.findtext("pubDate") or "").strip()
        try:
            dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc)
        except ValueError:
            dt = None
        if title:
            items.append((title, pub, dt))
    return items


def news_headlines(question: str, before: datetime | None = None, after: datetime | None = None,
                   max_items: int = 15) -> str:
    kw = _keywords(question)
    if not kw:
        return ""
    window = ""
    if after:
        window += f" after:{after:%Y-%m-%d}"
    if before:
        window += f" before:{before:%Y-%m-%d}"
    queries = [" ".join(kw[:8]), " ".join(kw[:4]), " ".join(kw[:2])]
    seen, lines, used = set(), [], []
    for q in dict.fromkeys(queries):
        if len(lines) >= max_items * 2 // 3:
            break
        got = _fetch(q + window)
        used.append(q)
        for title, pub, dt in got:
            if before and dt and dt >= before:
                continue
            key = title.lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            lines.append((dt or datetime.min.replace(tzinfo=timezone.utc), f"- {title} ({pub})"))
    if not lines:
        return ""
    lines.sort(key=lambda x: x[0], reverse=True)  # newest first
    return (
        f"News headlines (Google News searches: {', '.join(repr(u) for u in used)}), newest first. "
        "Headlines only; may be incomplete, partly irrelevant, or about a different but similar "
        "entity/instrument, so check they match the question exactly:\n"
        + "\n".join(l for _, l in lines[:max_items])
    )
