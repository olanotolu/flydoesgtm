import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8090"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=120) as r:
        return json.loads(r.read())


def post(path, body):
    req = urllib.request.Request(BASE + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


x = get("/live?query=select%20from%20companies%20where%20industry%20%3D%20"
        "%22Software%20Development%22&limit=6")
print("source:", x["source"])
print("distribution:", x["distribution"])
for r in x["rows"]:
    print("  %-24s %-9s -> %-8s (%s)" % (r["company"], r["size"],
                                          r["decision"], r["confidence"]))

signals = {"funding": 0.9, "hiring": 0.8, "intent": 0.9,
           "job_change": 0.7, "negative": 0.1, "trigger": 0.8}
pre = post("/decide", {"signals": signals, "policy": "pre"})
post_ = post("/decide", {"signals": signals, "policy": "post"})
print("pre-policy decision:", pre["decision"], pre["confidence"])
print("post-policy decision:", post_["decision"], post_["confidence"])
print("pools:", post_["pools"])
print("signals:", post_["signals"])
print("costs:", post_["costs"])
print("channel_activity keys:", len(post_["channel_activity"]))
