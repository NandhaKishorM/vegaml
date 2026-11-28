# Quick start

```bash
pip install vegaml
```

```python
import vegaml

v = vegaml.load()            # vega-U with the 0.8B reader: the default
# v = vegaml.load("4b")      # vega-U with the Qwen3.5-4B reader, same engine, same interface

out = v.decide(
    {"from": "billing@acme.com", "subject": "Invoice overdue", "body": "Third notice. Pay now."},
    {"team":  {"type": "choice", "instructions": "Which team should handle this?",
               "criteria": {"billing": "payments and invoices",
                            "technical": "product faults",
                            "sales": "new business"}},
     "churn": {"type": "boolean", "instructions": "Is this customer at risk of leaving?",
               "criteria": {"true": "shows intent to cancel", "false": "no such signal"}}})

print(out["answers"]["team"]["choice"])        # 'billing'
print(out["answers"]["team"]["probs"])         # calibrated, sums to 1
print(out["answers"]["churn"]["abstain"])      # False -> the engine stands behind it
```

Both questions are answered from one read of the state. Nothing is generated: the answer is a
value your code can branch on.

## What comes back

Every engine answer carries four things besides the answer itself.

| Field | What it tells you |
| --- | --- |
| `probs` | A calibrated probability for every option. 0.9 means right about nine times in ten. |
| `prediction_set` | The conformal set: every option the answer could still be. One option means it is sure. |
| `abstain` | The engine declines to commit. Send this one to a person or a larger model. |
| `unbound` | The input sits far from every answer, so it looks unlike anything the engine knows. |

[How it works](how-it-works.md) explains where these come from, and [Questions](questions.md)
covers the three answer types (choice, score, boolean) and how to write their criteria.

## Images, audio and video

```python
v.decide_image("invoice.png", questions)
v.decide_audio("call.wav", questions)       # pip install "vegaml[audio]"
v.decide_video("clip.mp4", questions)       # pip install "vegaml[video]"
```

The questions and the answers have the same shape for every kind of input.

## Fit it to your own labels

With labelled examples, a small head is fitted over the same frozen features in seconds. The
reader and the engine are not changed.

```python
v.fit(examples, questions)      # examples: [(state, {question: answer}), ...]
v.decide(state, questions)      # "auto" uses a head only where it beats the engine
```

Every answer says which one produced it, in `readout` (`"ttt"` for a head, `"engine"` for the
engine). A head is sharper on labels it has seen and carries no abstain flag; the engine answers
questions it has never seen. To package the questions, heads and calibration as one signed file,
see [Policies](policies.md).

## Keep learning

The vegaml-learn package adds a learner that answers from day one and learns from every answer you confirm,
and a teacher loop that asks a person or an LLM only when vega is unsure. It is a separate package,
free for noncommercial use: see [Continuous learning](learn.md) and [Licensing](licensing.md).

## Where next

- Serve it: [CLI](cli.md), [HTTP API](http-api.md), [MCP](mcp.md), [Docker](docker.md)
- No GPU: [CPU serving](cpu.md), [ONNX](onnx.md), [MLX](mlx.md)
- Measure it on your data: [Evals](evals.md), [Calibration](calibration.md), [Escalation](escalation.md)
- Hosted: the vega Proxy at [platform.embedding.in](https://platform.embedding.in)
