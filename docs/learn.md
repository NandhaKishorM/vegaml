# Continuous learning

`vegaml.Learner` answers your questions from day one and keeps learning from every answer you
confirm, on the machine it runs on. No fine-tuning, no GPU needed to learn, and no record leaves
that machine.

```python
from vegaml import Learner

questions = {
    "team": {"type": "choice", "instructions": "Which team handles this ticket?",
             "criteria": {"billing": "payments, refunds, invoices",
                          "technical": "bugs, errors, outages",
                          "sales": "pricing, plans, new accounts"}},
    "urgent": {"type": "boolean", "instructions": "Does this need a reply today?"},
}

learner = Learner(questions, workdir="./vega-state")       # loads vega; resumes from workdir

out = learner.decide("I was charged twice this month")
out["answers"]["team"]["answer"], out["answers"]["team"]["confidence"]

# whenever the right answer is known (your team fixed it, the customer confirmed it):
learner.feedback("I was charged twice this month", {"team": "billing", "urgent": False})
```

It learns every 50 confirmed answers, in a background thread, and switches to the new version
between requests. Nothing waits for it.

## How it learns

- Every question gets a small head: a classifier over vega's frozen features. vega itself never
  changes, so learning is a few seconds of numpy, not a fine-tune.
- A head answers only when its cross-validated accuracy beats vega's own answer on the same rows.
  Until then vega answers that question, with its calibrated probabilities and abstain flag. Every
  answer says which one answered, in `readout` ("head" or "engine").
- Your answers replace the starting set one for one, so after a few hundred it is fitted on your
  decisions alone.
- A batch the current heads agree with far less than they agree with your answers so far (a broken
  feed, or a real change of policy) is held, not learned. You decide: `learner.held()`,
  `learner.accept(question)` or `learner.discard(question)`.
- `learner.undo()` goes back one step.

## A starting set

Give it labelled examples and it starts with a fitted head instead of vega's zero-shot answer:

```python
rows = [("I was charged twice", {"team": "billing", "urgent": False}),
        ("The app crashes on login", {"team": "technical", "urgent": True}),
        ...]
learner = Learner(questions, examples=rows, workdir="./vega-state")
learner.report        # per question: rows used, cross-validated accuracy, vega alone on the same rows
```

Labels are read the way people write them: an option name, `True`/`"yes"`/`"no"`, or a level.

Measured across 14 decision templates, starting from about 600 generated examples each and then
learning from confirmed answers on 60 held-out rows: **85% right on day one, 92% after 40
confirmed answers, 96% after 120**.

## Saved state, more machines

- `workdir` keeps the last three versions and is picked up on the next start, as long as the
  questions, the starting set and the vega checkpoint are the same. A resumed learner does not
  reread its starting set. Call `learner.flush()` before shutting down to learn what is buffered.
- One writer learns; any number of read-only copies answer: `Learner(questions, workdir=shared,
  follow=30)` reloads the writer's newest state every 30 seconds and refuses feedback.
- `learner.save_bundle("acme.zip")` writes what it has learned as a file another machine starts
  from: `Learner(bundle="acme.zip")`.

## REST API with JWT

```bash
pip install "vegaml[learn]"
export VEGAML_LEARN_JWT_SECRET="$(openssl rand -hex 32)"
python -m vegaml.learn serve --questions questions.json --examples examples.csv \
       --workdir ./vega-state --host 0.0.0.0 --port 8000
python -m vegaml.learn token --subject helpdesk --scope decide feedback --days 30
```

`examples.csv` has a `text` column (or `--text-column`) and one column per question. Tokens are
HS256 with the shared secret, or verified with your identity provider's public key
(`--jwt-public-key key.pem`, RS256, ES256 or EdDSA, with `--jwt-audience` and `--jwt-issuer`).
Every token needs `exp`, and its scopes in `scope` or `scopes`:

| scope | routes |
| --- | --- |
| `decide` | `POST /v1/decide`, `POST /v1/decide/batch`, `GET /v1/questions` |
| `feedback` | `POST /v1/feedback` |
| `admin` | `GET /v1/status`, `POST /v1/learn`, `GET /v1/held`, `POST /v1/held/{q}/accept`, `POST /v1/held/{q}/discard`, `POST /v1/undo` |

`GET /healthz` needs no token.

## What it costs

Measured on 8 CPU threads with the 0.8B reader: 0.72 s an answer, 6.4 GB of RAM at peak, and 34 s
to learn from 300 answers. On a GPU, about 40 ms an answer in 2 GB.

## Limits

- The heads are only as good as the answers you confirm. A wrong feed is caught when it disagrees
  sharply with what was learned, not when it is slightly wrong.
- A head knows the options it has seen confirmed; an option nobody has confirmed yet is answered by
  vega until it has examples.
- Heads carry no calibration of their own: `confidence` from a head is its probability, not a
  calibrated one. Gate on `readout == "engine"` answers if you need vega's calibrated abstain flag.
