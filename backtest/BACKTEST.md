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

## 2026-10-29 — arm comparison, round 2

Runs: redteam on PAIRED-12 [40796362881]; baserate on paired ids 13–24 [40798694486]
(baserate now n = 24, ids 1–24). 24 backtest calls, reserve 15 kept for the live bot.

Common set (PAIRED-12, all arms):

| arm        | Brier  | Brier 95% CI     | log     |
|------------|--------|------------------|---------|
| baseline   | 0.2017 | [0.104, 0.303]   | -0.564  |
| redteam    | 0.2030 | [0.094, 0.331]   | -0.574  |
| baserate   | 0.2290 | [0.111, 0.357]   | -0.612  |
| structured | 0.2433 | [0.107, 0.418]   | -0.844  |
| market     | 0.1623 | (same 12 ids)    | -0.483  |

Paired vs baseline: redteam P(better)=0.48 (Brier), ≈ tie.

Baserate vs baseline on its 24 paired ids: Brier 0.188 vs 0.264, log -0.517 vs -0.767;
paired −Brier diff 95% CI [-0.033, +0.196], P(better)=0.91 → promising but CI overlaps 0.
Market on the same 24: Brier 0.140, log -0.431 (still better than every arm).
Baserate ECE 0.22 (poorly calibrated); Platt fit (0.68, -0.30) from n=24 — not shipped (< 25).

Winner rule not met → **no production change**. Local main.py default (`structured`) still not pushed.

Next: ensemble (2 free models, 12 ids = 24 calls) at the next reset; keep extending baserate
and redteam on ids 25+ to reach N≈40, then re-score (baserate is the lead candidate).
