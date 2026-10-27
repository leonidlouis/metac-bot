"""Re-run research (v2, research.py) for the dataset, keeping v1 as research_v1."""
import json, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research import news_headlines
from datetime import datetime, timedelta
P = "backtest/data/questions.json"
qs = json.load(open(P))
n = 0
for i, q in enumerate(qs):
    t = datetime.fromisoformat(q["open_time"])
    q.setdefault("research_v1", q["research"])
    q["research"] = news_headlines(q["title"], before=t, after=t - timedelta(days=45))
    n += bool(q["research"])
    time.sleep(0.7)
    if i % 20 == 0:
        print(i, n, flush=True)
print("with research v2:", n, "/", len(qs))
json.dump(qs, open(P, "w"), indent=1)
