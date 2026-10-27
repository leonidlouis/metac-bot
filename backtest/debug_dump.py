import json, os, urllib.request, time
H = {"Authorization": f"Token {os.environ['METACULUS_TOKEN']}", "User-Agent": "metac-bot-backtest", "Accept": "application/json"}
os.makedirs("backtest/data", exist_ok=True)
for pid in [46069, 45182, 44919, 45015]:
    for url in [f"https://www.metaculus.com/api/posts/{pid}/", f"https://www.metaculus.com/api2/questions/{pid}/"]:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=60) as r:
                d = json.loads(r.read())
            d.pop("user_permission", None)
            json.dump(d, open(f"backtest/data/raw_{pid}_{'v2' if 'api2' in url else 'v1'}.json", "w"), indent=1)
            q = d.get("question") or d
            print(pid, url, "resolution=", repr(q.get("resolution")), "status=", q.get("status"))
        except Exception as e:
            print(pid, url, type(e).__name__, getattr(e, "code", ""), getattr(e, "headers", {}).get("Retry-After") if hasattr(e, "headers") else "")
        time.sleep(2)
