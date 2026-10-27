# The escalation band, and keeping it fitted

Two things a pilot has to be able to do: send the uncertain middle to a human on a line it chose,
and refit that line itself when the traffic moves. Neither depends on us being in the room.

## Why not the abstain flag

Every engine answer carries `abstain`, and on the published calibration it fires only when the
conformal prediction set comes out empty. On a question with `k` options that needs `qhat < 1 -
1/k`. Across the six corpora we published it never fired once, and on two of them it could not
have: coverage was loose enough that every set held at least one option. A flag no operator can
trigger is not an escalation policy.

A TTT head has it worse. A head answer carries no conformal set and no abstain flag at all, so a
deployment that fits heads has no escalation signal from the model whatsoever.

The band fixes both by reading confidence and nothing else.

## A line you set

No labels needed:

```python
import vegaml

v = vegaml.load().set_band({"payment_change": 0.72, "team": 0.6})
out = v.decide(record, questions)
out["answers"]["payment_change"]["route"]            # "auto" or "review"
out["answers"]["payment_change"]["route_reason"]     # "confidence 0.610 is below the 0.720 review line"
out["band"]                                          # {"auto": 1, "review": 1, "unbanded": [], ...}
```

`vegaml.Band(default=0.6)` covers every question with one line; `Band.from_thresholds({...},
default=0.6)` mixes the two. A question with no line of its own and no default comes back
`route="unbanded"` and is named in the summary, because a question nobody configured should not be
silently auto-approved under a policy it was never measured against.

One call at a time: `v.decide(record, questions, review_below=0.6)`. From the shell: `vegaml
"..." --boolean "urgent=..." --review-below 0.6`.

The band applies to every readout, heads included, which is the only escalation a head answer
gets.

## A line fitted to a precision target

With labels, you can ask for the thing you actually want, which is not a threshold but a standard:

```python
from vegaml.calibrate import calibrate

cal = calibrate(v, examples, questions, target_precision=0.97)
cal.report["payment_change"]["band"]
# {"threshold": 0.83, "review_rate": 0.31, "precision": 0.983, "precision_lb": 0.971,
#  "reachable": True, "held_out": {"precision": 0.975, "review_rate": 0.29, ...}}
out = cal.apply(v.decide(record, questions, mode="engine"))
```

The widest automatic band whose precision holds at the target, so review load is as small as the
target allows. Three details that matter more than the threshold:

**It is fitted on a lower bound, not on the score.** The threshold is chosen on the same rows that
then measure it, and picking the best-looking cut on a few hundred rows is how a 97% target ships
as a 91% pilot. `precision_lb` is a one-sided 95% Wilson lower bound, so a narrow band pays for
being narrow: 12 correct out of 12 certifies 0.816, not 1.0.

**`held_out` is the number to quote.** The same fitting rule, refit on each four fifths of the rows
and scored on the fifth it did not see. It costs no data and it is out of sample.

**An unreachable target is refused, not approximated.** When no line holds it, `reachable` is
False, every answer routes to review, and `best_achievable` names the target this much data can
support. Lower the target, widen `max_review_rate`, or label more rows.

A review budget works the other way round, for a pilot with a fixed number of reviewers:
`calibrate(..., review_rate=0.2)` sends a fifth to review and reports the precision that buys.

## Recalibration in one command

```
vegaml-calibrate fit --labels recent.jsonl --questions @questions.json \
    --target-precision 0.97 --out calibration.json
```

`recent.jsonl` is one object per line, `{"state": ..., "labels": {"churn": "true"}}`, with an
optional per-row `"questions"`. The command refits the temperature, the conformal threshold and the
band, writes one file, and prints what changed: ECE before and after, the answer mix, the review
line, what the automatic share scored, and what it scored held out. It exits non-zero if any
question could not be calibrated. `--threads 1` and `--quantize int8` apply, as in
[CPU serving](cpu.md).

Fitting is cheap. On 240 rows the whole fit and calibrate took 26.4 seconds, of which 26.3 was
reading the rows through the frozen reader once; the temperature itself is 6 ms and the head is 43
ms. The cost of recalibrating is the cost of labelling, which is yours, and that is the honest
position: there is nothing here we have to do for you.

## Knowing when to run it

```
vegaml-calibrate check --answers last-week.jsonl --calibration calibration.json
```

No labels. It reads answers your deployment already produced and compares the mix of them against
the mix the calibration was fitted on, which is recorded automatically by `fit`. It exits 1 at
`recalibrate`, so a cron entry and a pager are the whole alerting setup. `--fail-on watch` pages
earlier, `--state watch.json` accumulates counters across daily runs, `--json` prints the report.

What it watches, and why those three:

* **The answer mix**, by population stability index, with the conventional bands: under 0.1 stable,
  0.1 to 0.25 watch, 0.25 and over refit. Our own drift runs are the reason. A covariate shift, the
  same decisions in different words, left calibration intact and in fact improved it, ECE 0.116 to
  0.055 with conformal coverage holding at 90.0% against a 90% target. A prior shift, the same
  words carrying a different mix of answers, broke it on all five corpora we tried. So the signal
  worth watching is not how the text looks, which is hard to measure and did not matter, but the
  mix of answers coming out, which is one counter per option and did.
* **Mean confidence**, because a reader that has started hedging on everything moves no mix at all.
* **Review load**, because a band whose review rate has doubled is costing a pilot money whatever
  the statistics say.
* **An option the calibration never produced**, which is not drift in degree but a different
  question, and always a refit.

Under 50 answers for a question it reports the numbers and refuses a verdict, because a mix drawn
from a handful of answers moves on its own and paging on that is how a drift alert gets switched
off for good.

In process, rather than from a file:

```python
from vegaml.monitor import Monitor

mon = Monitor.from_calibration(cal)
for result in stream:
    mon.observe(result)
mon.report()["status"]           # "ok", "watch", "recalibrate" or "insufficient"
```

16 labelled rows were enough to restore calibration after the prior shift in our runs, and 64 for
it to settle. That is the size of the job this command exists to make routine.
