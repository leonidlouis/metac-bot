"""Binary forecasting prompt variants used by the backtest (and by main.py)."""
import re
from datetime import datetime


def _days_between(a: datetime, b: datetime | None) -> str:
    if not b:
        return "unknown"
    return str(max(0, round((b - a).total_seconds() / 86400)))


def baseline_prompt(q: dict, today: datetime) -> str:
    """Exact template prompt the live bot used before tuning."""
    return f"""
You are a professional forecaster interviewing for a job.

Your interview question is:
{q['title']}

Question background:
{q['background']}


This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{q['resolution_criteria']}

{q['fine_print']}


Your research assistant says:
{q['research']}

Today is {today.strftime("%Y-%m-%d")}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A brief description of a scenario that results in a No outcome.
(d) A brief description of a scenario that results in a Yes outcome.

You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
""".strip()


def structured_prompt(q: dict, today: datetime, close: datetime | None, resolve: datetime | None) -> str:
    """Outside view -> inside view -> explicit time-to-resolution reasoning."""
    return f"""
You are an experienced, well-calibrated superforecaster. You are scored with a proper scoring rule (log score), so confident wrong answers are punished hard and needless hedging also costs points.

QUESTION:
{q['title']}

BACKGROUND:
{q['background']}

RESOLUTION CRITERIA (not yet satisfied as of today):
{q['resolution_criteria']}

FINE PRINT:
{q['fine_print']}

NEWS HEADLINES (published before today; may be incomplete or partly irrelevant):
{q['research'] or 'No headlines found.'}

Today is {today.strftime("%Y-%m-%d")}. Forecasting closes {close.strftime("%Y-%m-%d") if close else 'soon'}; the outcome is known around {resolve.strftime("%Y-%m-%d") if resolve else 'unknown'} (about {_days_between(today, resolve)} days from now).

Work through these steps briefly:
1. Resolution mechanics: exactly what must happen, by when, and per which source, for YES. Note thresholds and whether the current value is above or below them.
2. Outside view: name a reference class and estimate a base rate for this kind of event in a window of this length. Remember that tournament questions are often written about a change that may or may not happen; the status quo usually persists over short windows.
3. Inside view: list the specific evidence above that moves you away from the base rate, and how much. Discount weak or speculative headlines.
4. Time check: given the days left, how much can realistically change? Short windows strongly favour the status quo.
5. Final calibration: combine the views. If the outcome is already nearly determined by current facts, be decisive (e.g. 3% or 97%); otherwise avoid extremes.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
""".strip()


_PROB_RE = re.compile(r"Probability\s*[:=]\s*\**\s*([0-9]+(?:\.[0-9]+)?)\s*%", re.I)


def parse_probability(text: str) -> float | None:
    """Return the last 'Probability: ZZ%' in text as a decimal clamped to [0.01, 0.99]."""
    matches = _PROB_RE.findall(text or "")
    if not matches:
        return None
    p = float(matches[-1]) / 100.0
    if not 0.0 <= p <= 1.0:
        return None
    return max(0.01, min(0.99, p))
