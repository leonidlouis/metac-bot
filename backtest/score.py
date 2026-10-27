"""
Backtest step 3 (offline, no LLM): score results against resolutions.
Usage: python backtest/score.py backtest/out/*.jsonl
Prints Brier, log score (natural log, higher = better), Metaculus-style
baseline score (100*(1+log2 p_true)), for each variant/model, on the common
set of questions, plus constant-base-rate and community-prediction baselines,
leave-one-out-calibrated versions, and bootstrap CIs on paired differences.
"""
import json
import math
import random
import sys
from collections import defaultdict

QFILE = "backtest/data/questions.json"


def load(paths):
    recs = defaultdict(dict)  # key -> post_id -> p
    for path in paths:
        for line in open(path):
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


def boot_diff(pa, pb, ys, fn=logs, n=4000, seed=0):
    rnd = random.Random(seed)
    idx = list(range(len(ys)))
    diffs = []
    for _ in range(n):
        s = [rnd.choice(idx) for _ in idx]
        diffs.append(fn([pa[i] for i in s], [ys[i] for i in s]) - fn([pb[i] for i in s], [ys[i] for i in s]))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)], sum(d > 0 for d in diffs) / n


def report(name, ps, ys):
    print(f"  {name:38s} n={len(ys):3d}  Brier={brier(ps, ys):.4f}  log={logs(ps, ys):+.4f}  "
          f"baseline_score={baseline_score(ps, ys):+.1f}")


def main(paths):
    qs = {q["post_id"]: q for q in json.load(open(QFILE))}
    recs = load(paths)
    keys = sorted(recs)
    print("variants:", {k: len(v) for k, v in recs.items()})
    for k in keys:
        ids = sorted(recs[k])
        ys = [qs[i]["resolution"] for i in ids]
        ps = [recs[k][i] for i in ids]
        print(f"\n== {k} (own set) ==  yes-rate={sum(ys)/len(ys):.2f}")
        report("raw", ps, ys)
        report("constant 0.35 (generic prior)", [0.35] * len(ys), ys)
        if len(ys) >= 10:
            report("LOO Platt-calibrated", loo_platt(ps, ys), ys)
            print("  Platt fit on all (a=slope on logit, b=shift):", tuple(round(v, 2) for v in fit_platt(ps, ys)))
    # paired comparison on common questions
    if len(keys) >= 2:
        common = sorted(set.intersection(*[set(recs[k]) for k in keys]))
        print(f"\n== common set: {len(common)} questions ==")
        ys = [qs[i]["resolution"] for i in common]
        for k in keys:
            report(k, [recs[k][i] for i in common], ys)
        cp_ids = [i for i in common if qs[i].get("cp_at_open_plus")]
        if cp_ids:
            print(f"  community/aggregate baseline available on {len(cp_ids)} of them:")
            ys2 = [qs[i]["resolution"] for i in cp_ids]
            report("community (3h after open)", [clamp(qs[i]["cp_at_open_plus"]["p"]) for i in cp_ids], ys2)
            for k in keys:
                report(f"  {k} on same", [recs[k][i] for i in cp_ids], ys2)
        base = keys[0]
        for k in keys[1:]:
            lo, hi, pr = boot_diff([recs[k][i] for i in common], [recs[base][i] for i in common], ys)
            print(f"  log-score diff {k} - {base}: 95% CI [{lo:+.3f}, {hi:+.3f}], P(better)={pr:.2f}")
            lo, hi, pr = boot_diff([recs[k][i] for i in common], [recs[base][i] for i in common], ys,
                                   fn=lambda p, y: -brier(p, y))
            print(f"  -Brier diff  {k} - {base}: 95% CI [{lo:+.4f}, {hi:+.4f}], P(better)={pr:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
