"""Tables for the check-set experiment: means over seeds 4-6. Writes results/runs/summary.json and prints markdown."""

import collections
import glob
import json
import statistics as st

R = collections.defaultdict(list)
FA = collections.defaultdict(list)
for f in glob.glob("results/runs/*_s*_r*.json"):
    d = json.load(open(f))
    rate = f.rsplit("_r", 1)[1][:-5]
    if d["seed"] in (4, 5, 6):
        R[(d["task"], d["teacher"], d["guard"], rate)].append(d)
    if d["teacher"] in ("honest", "noisy") and d["guard"] == "check":
        FA[(d["task"], d["teacher"], rate)].append(d)
m = lambda v: st.mean(v) if v else None
pc = lambda x: "-" if x is None else "%.1f" % (100 * x)
TASKS = ["massive", "cb", "saas"]
S = {}


def cell(task, k, g, rate="0.25"):
    rs = R.get((task, k, g, rate), [])
    if not rs:
        return None
    al = [r for r in rs if r["alarms"]]
    return {
        "n": len(rs),
        "acc": m([r["final"]["acc"] for r in rs]),
        "acc0": m([r["curve"][0]["acc"] for r in rs]),
        "right": m([r["final"]["target_right"] for r in rs if r["final"]["target_right"] is not None]),
        "bad": m([r["final"]["target_bad"] for r in rs if r["final"]["target_bad"] is not None]),
        "right0": m([r["curve"][0]["target_right"] for r in rs if r["curve"][0]["target_right"] is not None]),
        "taught": m([r["sabotaged_taught"] for r in rs]),
        "left": m([r["sabotaged_left"] for r in rs]),
        "alarms": len(al),
        "rules": sorted(collections.Counter(r["alarms"][0]["rule"] for r in al).items()),
        "alarm_at": [r["alarm_at"] for r in al],
        "onset": rs[0]["onset"],
        "delay": m([r["alarm_at"] - (r["onset"] or 0) for r in al]) if al else None,
        "forgot": m([r["alarms"][0]["forgotten"] for r in al]) if al else None,
        "kept": m([r["alarms"][0].get("kept", 0) for r in al]) if al else None,
        "calls": m([r["teacher_calls"] for r in rs]),
        "checks": m([r["checks"] for r in rs]),
        "person": m([r["needs_person"] for r in rs]),
        "share": m([r["teacher_share"] for r in rs]),
        "honest_acc": m([r["honest_acc"] for r in rs]),
        "target_share": m([r["target_share"] for r in rs]),
        "secs": m([r["seconds"] for r in rs]),
    }


print("## A. No check: what vega learns (final, mean of 3 seeds)\n")
print(
    "| task | teacher | vega alone | targeted right | targeted answered as the saboteur wants | bad answers taught | teacher share |\n|---|---|---|---|---|---|---|"
)
for t in TASKS:
    for k in ["honest", "target", "target_half", "target_late", "rand5", "rand5_late"]:
        c = cell(t, k, "none")
        S["%s/%s/none" % (t, k)] = c
        if c:
            print(
                "| %s | %s | %s | %s | %s | %.0f | %s |"
                % (t, k, pc(c["acc"]), pc(c["right"]), pc(c["bad"]), c["taught"], pc(c["share"]))
            )
print("\n## B. With the check set (200 records, rate 0.25)\n")
print(
    "| task | teacher | alarms (of 3) | rule | records after onset until alarm | bad answers taught | left after rollback | targeted right | vega alone | extra teacher calls |\n|---|---|---|---|---|---|---|---|---|---|"
)
for t in TASKS:
    for k in ["honest", "noisy", "target", "target_half", "target_late", "rand5", "rand5_late"]:
        c = cell(t, k, "check")
        S["%s/%s/check" % (t, k)] = c
        if c:
            print(
                "| %s | %s | %d | %s | %s | %.0f | %.1f | %s | %s | %.0f |"
                % (
                    t,
                    k,
                    c["alarms"],
                    ", ".join("%s x%d" % x for x in c["rules"]) or "-",
                    "-" if c["delay"] is None else "%.0f" % c["delay"],
                    c["taught"],
                    c["left"],
                    pc(c["right"]),
                    pc(c["acc"]),
                    c["checks"],
                )
            )
print("\n## C. With a floor three points under the honest teacher's score\n")
print("| task | teacher | alarms (of 3) | rule | targeted right | vega alone |\n|---|---|---|---|---|---|")
for t in TASKS:
    for k in ["honest", "noisy", "target", "target_half", "rand5"]:
        c = cell(t, k, "floor")
        S["%s/%s/floor" % (t, k)] = c
        if c:
            print(
                "| %s | %s | %d | %s | %s | %s |"
                % (
                    t,
                    k,
                    c["alarms"],
                    ", ".join("%s x%d" % x for x in c["rules"]) or "-",
                    pc(c["right"]),
                    pc(c["acc"]),
                )
            )
print("\n## D. Check rate against delay (teacher turns at the middle of the stream)\n")
print(
    "| task | teacher | rate | alarms (of 3) | records until alarm | bad taught | left | extra calls |\n|---|---|---|---|---|---|---|---|"
)
for t in ["massive", "cb"]:
    for k in ["target_late", "rand5_late"]:
        for r in ["0.25", "0.5", "1.0"]:
            c = cell(t, k, "check", r)
            S["%s/%s/check/%s" % (t, k, r)] = c
            if c:
                print(
                    "| %s | %s | %s | %d | %s | %.0f | %.1f | %.0f |"
                    % (
                        t,
                        k,
                        r,
                        c["alarms"],
                        "-" if c["delay"] is None else "%.0f" % c["delay"],
                        c["taught"],
                        c["left"],
                        c["checks"],
                    )
                )
print("\n## E. False alarms: an honest teacher under the check\n")
print("| task | teacher | rate | runs | runs with an alarm | checks per run |\n|---|---|---|---|---|---|")
for (t, k, r), v in sorted(FA.items()):
    S["fa/%s/%s/%s" % (t, k, r)] = {
        "runs": len(v),
        "alarms": sum(bool(x["alarms"]) for x in v),
        "checks": m([x["checks"] for x in v]),
        "rules": [x["alarms"][0]["rule"] for x in v if x["alarms"]],
    }
    print(
        "| %s | %s | %s | %d | %d | %.0f |"
        % (t, k, r, len(v), sum(bool(x["alarms"]) for x in v), m([x["checks"] for x in v]))
    )
json.dump(S, open("results/summary.json", "w"), indent=1)
print("\nruns:", sum(len(v) for v in R.values()))
