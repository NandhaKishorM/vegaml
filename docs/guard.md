# The prompt-injection guard

A guard is a two-way decision on an incoming request: forward it, or refuse it. That is an
ordinary `choice` question, so the whole policy is one sheet and needs no new engine.

```python
import vegaml
from vegaml.guard import GUARD

v = vegaml.load()
a = v.decide(request, {"guard": GUARD})["answers"]["guard"]
if a["choice"] == "block":
    raise PermissionError("refused at the gate")
```

`GUARD`'s criteria contain no strings to match. There is no mention of "ignore previous
instructions" and no list of attack names, because a guard built on literals is beaten by the
first attacker who rephrases. The criteria describe the property instead: an attempt to
subvert the assistant rather than use it.

## Scoring one honestly

A single number cannot describe a guard. The two degenerate policies sit at the extremes:
blocking everything catches every attack, and forwarding everything never refuses a customer.
Both score well on one metric and are worthless.

```python
from vegaml.guard import GUARD, score, degenerate

preds = [v.decide(t, {"guard": GUARD})["answers"]["guard"]["choice"] for t in texts]
s = score(preds, labels)            # labels are "block" / "allow"

print("caught %.1f%% of attacks, refused %.1f%% of good traffic"
      % (100 * s["block_recall"], 100 * s["false_positive_rate"]))

why = degenerate(s)
if why:
    raise AssertionError("this guard is not reading the request: %s" % why)
```

`score` returns the four cells and the figures worth quoting:

| Key | What it is |
| --- | --- |
| `block_recall` | the share of attacks caught |
| `false_positive_rate` | the share of ordinary traffic refused |
| `balanced_accuracy` | the mean of catch rate and pass rate |
| `block_rate` | the share of all traffic refused, which is what `degenerate` reads |
| `counts` | `tp`, `fn`, `fp`, `tn` |

Prefer `balanced_accuracy` as the headline. Plain accuracy on a corpus that is mostly attacks
rewards a guard for refusing everything, and most published jailbreak corpora are mostly
attacks.

Anything that is not `block` counts as a forward, abstentions included. An abstention falls
through to the default policy in production, so counting it as a catch would score a coin flip
as a save.

## Measure the false positives or do not bother

Scoring needs a labelled corpus with both classes in it. Measuring only attacks tells you
nothing, because that is exactly the measurement a block-everything guard passes. `degenerate`
says so rather than returning a flattering number:

```python
score(["block"] * 60, ["block"] * 60)["false_positive_rate"]   # None
degenerate(score(["block"] * 60, ["block"] * 60))
# 'the corpus has only one class, which no guard can fail'
```

## Fitting the policy to your own traffic

Untaught, the engine answers the sheet from the criteria alone, and on adversarial text that
is often close to a constant: it either refuses nearly everything or forwards nearly
everything, which `degenerate` will tell you. A guard is a policy, and a policy is what a
fitted head encodes. Collect a few hundred labelled requests from your own gateway, fit, and
score the held-out half:

```python
from vegaml.ttt import TTTHead

head = TTTHead(["allow", "block"], k=96)
head.fit([v.features(t, GUARD) for t in train_texts], train_labels)
```

See [Calibration](calibration.md) for refitting probabilities and [Evals](evals.md) for running
a labelled set end to end.

## What a guard is not

One layer. It sits in front of a model that also refuses, and neither replaces the other. It
reads the request, so it says nothing about what the model then does with a document the
request pulls in: indirect injection arrives later, in tool output, and has to be guarded where
that output enters.
