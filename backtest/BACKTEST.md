# Backtest results

Dataset: resolved Polymarket binaries (`backtest/data/questions_expanded.json`, 600 Qs),
forecast time T = end − 10 days, crowd price at T = market baseline. Model:
`openrouter/thinkingmachines/inkling:free`. Scoring: `backtest/score.py` (bootstrap 95% CIs).

## 2026-10-28 — arm comparison, round 1 (paired 12 ids)

Runs: structured [40611844867], baserate [40613139559]; baseline from [40555881044].
Common set n = 12 (PAIRED-12 in tuning plan).

| arm        | Brier  | Brier 95% CI     | log     |
|------------|--------|------------------|---------|
| baseline   | 0.2017 | [0.104, 0.303]   | -0.564  |
| baserate   | 0.2290 | [0.111, 0.357]   | -0.612  |
| structured | 0.2433 | [0.107, 0.418]   | -0.844  |
| market     | 0.1623 | (same 12 ids)    | -0.483  |

Paired vs baseline: baserate P(better)=0.34 (Brier), structured P(better)=0.34.
Neither clears the winner rule → **no production change**; live bot keeps its current prompt.
Calibration: not fitted (n < 25; LOO-Platt does not beat raw).
Caveat: the baseline arm used research v1 headlines, the new arms research v2.

Next: redteam (+ more paired ids for baserate) at the next free-quota reset; ensemble after.
