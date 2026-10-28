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

NEWS HEADLINES (published before today; may be incomplete, partly irrelevant, or about a similar but different entity, instrument or date range):
{q['research'] or 'No headlines found.'}

Today is {today.strftime("%Y-%m-%d")}. Forecasting closes {close.strftime("%Y-%m-%d") if close else 'soon'}; the outcome is known around {resolve.strftime("%Y-%m-%d") if resolve else 'unknown'} (about {_days_between(today, resolve)} days from now).

Work through these steps briefly:
1. Resolution mechanics: exactly what must happen, by when, and per which source, for YES. Note thresholds and whether the current value is above or below them.
2. Outside view: name a reference class and estimate a base rate for this kind of event in a window of this length. Remember that tournament questions are often written about a change that may or may not happen; the status quo usually persists over short windows.
3. Inside view: list the specific evidence above that moves you away from the base rate, and how much. Check each headline really matches the question (e.g. Brent vs WTI, a different election round, region or date) before relying on it; discount weak or speculative headlines. If there is little relevant news, stay close to the base rate rather than guessing a story.
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


def baserate_prompt(q: dict, today: datetime, close: datetime | None, resolve: datetime | None) -> str:
    """Arm (a): explicit base-rate anchoring + inside/outside view with a forced
    numeric anchor before adjustment, to fight the near-uninformative raw forecasts."""
    return f"""
You are a well-calibrated superforecaster scored by a proper scoring rule (log score). Your single biggest failure mode is jumping to an inside-view story and ignoring base rates, so you ALWAYS anchor on an explicit base rate first and adjust only for strong, well-matched evidence.

QUESTION:
{q['title']}

BACKGROUND:
{q['background']}

RESOLUTION CRITERIA (not yet satisfied as of today):
{q['resolution_criteria']}

FINE PRINT:
{q['fine_print']}

NEWS HEADLINES (published before today; may be incomplete, partly irrelevant, or about a similar but different entity/instrument/date):
{q['research'] or 'No headlines found.'}

Today is {today.strftime("%Y-%m-%d")}. Forecasting closes {close.strftime("%Y-%m-%d") if close else 'soon'}; the outcome is known around {resolve.strftime("%Y-%m-%d") if resolve else 'unknown'} (about {_days_between(today, resolve)} days from now).

Work through these steps and WRITE each number down:
1. OUTSIDE VIEW FIRST. Define a reference class ("how often does an event like this resolve YES in a window this long?"). State an explicit base-rate number P0 (a percentage) BEFORE looking at the specifics. For "will X change / happen by a deadline" questions the base rate is usually LOW because the status quo persists over short windows.
2. Resolution mechanics: exactly what must be true, by when, per which source, for YES. Is the current value already above/below the threshold?
3. INSIDE VIEW adjustments. List only the evidence that genuinely matches THIS question (check entity, instrument, region, date range, round). For each, say whether it pushes P up or down and by roughly how many points. Discount speculative or weakly-matched headlines; if there is little relevant news, barely move from P0.
4. Time check: given days left, how much can realistically change? Short windows favour the status quo strongly.
5. FINAL: start from P0, apply your adjustments, and state the result. If the outcome is nearly determined by current facts, be decisive (e.g. 3% or 97%); otherwise do not overshoot your evidence.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
""".strip()


def redteam_prompt(q: dict, today: datetime, close: datetime | None, resolve: datetime | None) -> str:
    """Arm (b): produce a first forecast, then red-team it in the SAME completion
    and give a corrected final number. One LLM call, self-critique built in."""
    return f"""
You are a well-calibrated superforecaster scored by a proper scoring rule (log score). You will forecast, then red-team your own forecast, then give a corrected answer -- all in this one response.

QUESTION:
{q['title']}

BACKGROUND:
{q['background']}

RESOLUTION CRITERIA (not yet satisfied as of today):
{q['resolution_criteria']}

FINE PRINT:
{q['fine_print']}

NEWS HEADLINES (published before today; may be incomplete, partly irrelevant, or about a similar but different entity/instrument/date):
{q['research'] or 'No headlines found.'}

Today is {today.strftime("%Y-%m-%d")}. Forecasting closes {close.strftime("%Y-%m-%d") if close else 'soon'}; the outcome is known around {resolve.strftime("%Y-%m-%d") if resolve else 'unknown'} (about {_days_between(today, resolve)} days from now).

PART 1 - FIRST PASS. Briefly: resolution mechanics; an explicit base rate; the key evidence; a first probability P1.

PART 2 - RED TEAM. Now attack P1 as if you were paid to prove it wrong:
- Did you misread the resolution criteria or the threshold/direction?
- Did you treat a headline as relevant when it is about a different entity, instrument, region, round or date?
- Are you over-reacting to vivid but speculative news, or under-weighting the status-quo/base rate?
- Are you too confident given how little can change in the time left, OR too hedged when the facts already settle it?
List the 2-3 strongest objections.

PART 3 - CORRECTED FORECAST. Revise P1 in light of the red team. If the objections hold, move toward the base rate / status quo; if they don't, keep (or sharpen) your number.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
""".strip()


# Variant registry so runners (backtest + main) can select by name.
# "ensemble" uses the structured prompt but is run as 2 cheap free-model calls
# and averaged by the runner (see run_backtest.py --variant ensemble).
PROMPT_BUILDERS = {
    "baseline": lambda q, today, close, resolve: baseline_prompt(q, today),
    "structured": structured_prompt,
    "baserate": baserate_prompt,
    "redteam": redteam_prompt,
    "ensemble": structured_prompt,
}


def build_prompt(variant: str, q: dict, today: datetime,
                 close: datetime | None = None, resolve: datetime | None = None) -> str:
    fn = PROMPT_BUILDERS.get(variant)
    if fn is None:
        raise ValueError(f"unknown prompt variant: {variant}")
    return fn(q, today, close, resolve)
