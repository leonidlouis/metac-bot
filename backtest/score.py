"""
Backtest step 3 (offline, no LLM, no network): score results against resolutions.

Usage:
  python backtest/score.py backtest/out/*.jsonl
  BACKTEST_QFILE=backtest/data/questions_expanded.json python backtest/score.py ... 
  python backtest/score.py --questions backtest/data/questions_expanded.json backtest/out/*.jsonl

Prints, for each variant/model:
- Brier, log score (natural log; higher = better), Metaculus-style baseline
  score 100*(1+log2 p_true), each with a bootstrap 95% CI;
- constant-base-rate and community-prediction (crowd price at T) baselines;
- a reliability / calibration table (predicted-prob buckets vs realized freq);
- leave-one-out Platt-calibrated version (when n>=10).
On the common question set it adds paired bootstrap CIs of each arm vs the first
arm AND vs the market (crowd) baseline, which is robust to small N.
"""
import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict

QFILE = os.getenv("BACKTEST_QFILE", "backtest/data/questions.json")


def load(paths):
    recs = defaultdict(dict)  # key -> post_id -> p
    for path in paths:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("p") is None:
                continue
            key = f"{r['variant']}|{r['model'].split('/')[-1]}"
            recs[key][r["post_id"]] = r["p"]
    return recs


def clamp(p, lo=0.01, hi=0.99):
    return max(lo, min(hi, p))


def brier(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ys)


def logs(ps, ys):
    return sum(math.log(p if y else 1 - p) for p, y in zip(ps, ys)) / len(ys)


def baseline_score(ps, ys):
    return sum(100 * (1 + math.log2(p if y else 1 - p)) for p, y in zip(ps, ys)) / len(ys)


def logit(p):
    return math.log(p / (1 - p))


def sig(x):
    return 1 / (1 + math.exp(-x))


def fit_platt(ps, ys, iters=300, lr=0.1):
    a, b = 1.0, 0.0
    xs = [logit(clamp(p)) for p in ps]
    for _ in range(iters):
        ga = gb = 0.0
        for x, y in zip(xs, ys):
            q = sig(a * x + b)
            ga += (q - y) * x
            gb += (q - y)
        a -= lr * ga / len(xs)
        b -= lr * gb / len(xs)
    return a, b


def loo_platt(ps, ys):
    out = []
    for i in range(len(ps)):
        a, b = fit_platt(ps[:i] + ps[i + 1:], ys[:i] + ys[i + 1:])
        out.append(clamp(sig(a * logit(clamp(ps[i])) + b)))
    return out


def boot_metric(ps, ys, fn=brier, n=4000, seed=0):
    """Bootstrap 95% CI of a single metric over resampled questions."""
    rnd = random.Random(seed)
    idx = list(range(len(ys)))
    vals = []
    for _ in range(n):
        s = [rnd.choice(idx) for _ in idx]
        vals.append(fn([ps[i] for i in s], [ys[i] for i in s]))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def boot_diff(pa, pb, ys, fn=logs, n=4000, seed=0):
    """Paired bootstrap of fn(a) - fn(b): 95% CI and P(a better than b)."""
    rnd = random.Random(seed)
    idx = list(range(len(ys)))
    diffs = []
    for _ in range(n):
        s = [rnd.choice(idx) for _ in idx]
        diffs.append(fn([pa[i] for i in s], [ys[i] for i in s]) - fn([pb[i] for i in s], [ys[i] for i in s]))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)], sum(d > 0 for d in diffs) / n


def report(name, ps, ys, ci=False):
    extra = ""
    if ci:
        bl, bh = boot_metric(ps, ys, brier)
        ll, lh = boot_metric(ps, ys, logs)
        extra = f"  Brier95%[{bl:.4f},{bh:.4f}]  log95%[{ll:+.4f},{lh:+.4f}]"
    print(f"  {name:38s} n={len(ys):3d}  Brier={brier(ps, ys):.4f}  log={logs(ps, ys):+.4f}  "
          f"baseline_score={baseline_score(ps, ys):+.1f}{extra}")


def reliability_table(ps, ys, nbins=5):
    """Predicted-prob buckets vs realized frequency (calibration)."""
    edges = [i / nbins for i in range(nbins + 1)]
    print(f"  reliability ({nbins} buckets): pred-range | n | mean_pred | realized_freq | gap")
    ece = 0.0
    for b in range(nbins):
        lo, hi = edges[b], edges[b + 1]
        sel = [(p, y) for p, y in zip(ps, ys) if (lo <= p < hi or (b == nbins - 1 and p == hi))]
        if not sel:
            print(f"    [{lo:.2f},{hi:.2f})   0        -         -           -")
            continue
        mp = sum(p for p, _ in sel) / len(sel)
        rf = sum(y for _, y in sel) / len(sel)
        ece += len(sel) / len(ys) * abs(mp - rf)
        print(f"    [{lo:.2f},{hi:.2f})  {len(sel):3d}    {mp:.3f}     {rf:.3f}       {rf - mp:+.3f}")
    print(f"    ECE (expected calibration error): {ece:.4f}")


def main(paths, qfile=None):
    qfile = qfile or QFILE
    qs = {q["post_id"]: q for q in json.load(open(qfile))}
    recs = load(paths)
    keys = sorted(recs)
    print(f"questions file: {qfile} ({len(qs)} Qs)")
    print("variants:", {k: len(v) for k, v in recs.items()})
    for k in keys:
        ids = sorted(recs[k])
        ys = [qs[i]["resolution"] for i in ids]
        ps = [recs[k][i] for i in ids]
        print(f"\n== {k} (own set) ==  yes-rate={sum(ys)/len(ys):.2f}")
        report("raw", ps, ys, ci=True)
        report("constant 0.35 (generic prior)", [0.35] * len(ys), ys)
        reliability_table(ps, ys)
        if len(ys) >= 10:
            report("LOO Platt-calibrated", loo_platt(ps, ys), ys, ci=True)
            print("  Platt fit on all (a=slope on logit, b=shift):", tuple(round(v, 2) for v in fit_platt(ps, ys)))
    # paired comparison on common questions
    if len(keys) >= 2:
        common = sorted(set.intersection(*[set(recs[k]) for k in keys]))
        print(f"\n== common set: {len(common)} questions ==")
        ys = [qs[i]["resolution"] for i in common]
        for k in keys:
            report(k, [recs[k][i] for i in common], ys, ci=True)
        base = keys[0]
        for k in keys[1:]:
            lo, hi, pr = boot_diff([recs[k][i] for i in common], [recs[base][i] for i in common], ys)
            print(f"  log-score diff {k} - {base}: 95% CI [{lo:+.3f}, {hi:+.3f}], P(better)={pr:.2f}")
            lo, hi, pr = boot_diff([recs[k][i] for i in common], [recs[base][i] for i in common], ys,
                                   fn=lambda p, y: -brier(p, y))
            print(f"  -Brier diff  {k} - {base}: 95% CI [{lo:+.4f}, {hi:+.4f}], P(better)={pr:.2f}")

    # market (crowd) comparison per arm on its OWN set, robust to small N via paired bootstrap
    print("\n== model vs market (crowd price at T), paired per arm ==")
    for k in keys:
        ids = [i for i in sorted(recs[k]) if qs[i].get("cp_at_open_plus")]
        if not ids:
            continue
        ys = [qs[i]["resolution"] for i in ids]
        pm = [clamp(qs[i]["cp_at_open_plus"]["p"]) for i in ids]
        pk = [recs[k][i] for i in ids]
        report(f"{k} (model)", pk, ys)
        report("market (crowd at T)", pm, ys)
        lo, hi, pr = boot_diff(pk, pm, ys)
        print(f"  log-score diff model-market: 95% CI [{lo:+.3f}, {hi:+.3f}], P(model better)={pr:.2f}")
        lo, hi, pr = boot_diff(pk, pm, ys, fn=lambda p, y: -brier(p, y))
        print(f"  -Brier diff  model-market: 95% CI [{lo:+.4f}, {hi:+.4f}], P(model better)={pr:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=None)
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    main(a.paths, a.questions)
