# Trained policies

*Added in 0.16.0.*

A policy is the trained layer on top of the frozen reader: one small head per question, the
calibration those heads were measured with (review band included), the questions they answer,
and the off-topic limit for the routing question. Load one, and every answer comes back marked
`auto` or `review` with the reason in a sentence.

```python
import vegaml

policy = vegaml.load_policy("acme-policy.zip")    # Embedding Studio's download, or a saved folder
policy.verified          # {"status": "signed", "signed_by": "Embedding Space (key 589c...)", "engine": "matches"}

out = policy.decide("I was charged twice for my order")
out["route"]                                # "auto" or "review", for the whole record
out["answers"]["queue"]["choice"]           # the answer
out["answers"]["queue"]["route_reason"]     # why it was routed that way
out["outbound"]                             # {"residual", "limit", "question", "outside"}
```

With no `model=`, `load_policy` loads the reader the policy was trained on (the manifest names
it). Pass `model=vegaml.load(...)` to reuse one you already have, and `device=` like any load.

## Verified before anything is installed

1. **Hashes.** Every file the manifest names must have exactly the SHA-256 it states. A changed
   or swapped file is refused, naming the file.
2. **Signature.** A manifest that names a signing key must carry an Ed25519 signature that
   verifies against a key this release trusts (`vegaml.policy.TRUSTED_KEYS`). Embedding Space
   signs every policy Studio exports; its public key is also served at
   `GET /v1/policy-signing-key` on the Studio API, so anyone can check one by hand.
3. **Engine.** The loaded engine must be the one the heads were trained on: heads fitted on one
   reader's features are meaningless on another's.

Any failure raises `vegaml.policy.PolicyError`. `verify="signed"` also refuses an unsigned
policy, which is what `vegaml-serve --policy` uses by default. `verify=False` checks nothing.

## The route

Four things send an answer to a person. None of them is a confidence number someone had to
guess:

| Reason | What it means |
| --- | --- |
| outbound | The record lies outside what the policy learned: the routing head's projection holds too little of it. Every answer of that record goes to review. |
| not calibrated | The calibration does not cover this question, or it was asked with an option the policy was never trained on (`calibrated: False`). |
| abstain | The conformal set is empty (the confidence is under the bar the held-out rows set, shown as `auto_at`) or holds more than one option. |
| band | The policy's review line for that question, placed where precision held on its calibration rows. |

The record is `auto` only when every answer is. Pass `review_below=0.8` to `decide` to draw your
own line instead: your line and the outbound check then decide, and `abstain` and `calibrated`
are reported without routing.

**Gate on `route`, never on `confidence`.** A fitted head can return 1.000 on its own mistakes;
the route comes from the conformal set, the band and the outbound check rather than from the top
probability.

Pictures, and clips decided from their frames, are answered by the vision tower's zero-shot
readout, which a text-trained policy does not cover: those answers are `calibrated: False` and
go to review unless you give `review_below`.

## Fit one yourself

```python
v = vegaml.load()
policy = vegaml.fit_policy(v, "labels.csv", questions)    # or a list of {"state", "gold"} rows
policy.fit_report["serving"]       # the questions whose head beat the engine
policy.save("my-policy/")          # heads.npz, calibration.json, questions.json, manifest.json
```

`labels.csv` has a `state` column and one column per question id holding that row's answer.
A quarter of the rows, taken evenly across the first question's labels, is held back to
calibrate on, so the calibration is measured on rows the heads never saw (or pass
`calibration_rows=`). `vegaml.policy.min_rows(questions)` says how many you need: a five-way
choice needs 35, a yes/no question 20. Those are floors.

The off-topic limit is the largest residual of the routing head over the calibration rows.
Studio also raises it to the residual of a bank of generic off-topic messages, which a few
hundred of your own rows cannot measure.

## Serving

```
pip install "vegaml[serve]"
vegaml-serve --policy acme-policy.zip --port 8000
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{"state": "my invoice is wrong"}'
```

The policy is verified before the port opens, and a request may leave `questions` out to ask the
policy's own. `--unsigned-ok` serves a policy you saved yourself. See [HTTP API](http-api.md).

## Studio's download

Studio's `.zip` holds the policy in one top-level folder and spells yes/no questions `noul` (the
spelling laya uses); `load_policy` reads both spellings and needs nothing unpacked.
