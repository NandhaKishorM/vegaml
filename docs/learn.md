# Continuous learning

> **Separate package, noncommercial licence.** Continuous learning, the teacher loop and the vega Proxy
> are in [vegaml-learn](https://github.com/NandhaKishorM/vegaml/blob/main/learn/README.md): `pip install vegaml-learn`. It is free for noncommercial use
> under the [PolyForm Noncommercial License 1.0.0](https://github.com/NandhaKishorM/vegaml/blob/main/learn/LICENSE); commercial use needs a paid licence:
> [embedding.in/licence](https://embedding.in/licence), quote from nandakishor@embedding.in. vegaml itself stays Apache 2.0, and the learner that shipped in
> vegaml 0.17 to 0.19 keeps the Apache 2.0 licence it was released under. Imports written for 0.19
> (`from vegaml import Learner`, `vegaml_learn`) keep working once vegaml-learn is installed.

`vegaml_learn.Learner` answers your questions from day one and keeps learning from every answer you
confirm, on the machine it runs on. No fine-tuning, no GPU needed to learn, and no record leaves
that machine.

```python
from vegaml_learn import Learner

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
- A head answers only when its cross-validated accuracy, and its balanced accuracy (mean recall per
  answer), beat vega's own answer on the same rows. The balanced bar is what stops a head that always
  gives the common answer from taking over a question where the other answer is rare.
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

## A teacher for what it is unsure of

When nobody confirms answers by hand, give the learner a teacher: a person's review queue, a rules
engine, or a large model. `Escalator` sends the teacher only what vega is unsure of, serves the
teacher's answer, and learns from it, so the share that needs the teacher falls as vega learns.

```python
from vegaml_learn import Escalator, Learner, claude_teacher

learner = Learner(questions, workdir="./vega-state")
desk = Escalator(learner, teacher=claude_teacher())   # pip install "vegaml-learn[claude]"; ANTHROPIC_API_KEY
out = desk.decide("I was charged twice this month")
out["answers"]["team"]["source"]        # "teacher" while vega is learning, "vega" once it is sure
out["answers"]["team"]["escalated"]     # why: "abstain", "unbound", "below line" or "audit"
desk.report()["teacher_share_recent"]   # the share of the last 100 records the teacher still saw
```

Any function works as the teacher: `teacher(text, {question id: question})` returns `{question id:
answer}`, written the way you would write feedback. A question it cannot answer can be left out; a
teacher that raises or returns something unusable is skipped, so vega's own answer is served and
nothing is learned from that record.

Who goes to the teacher:

- an engine answer that abstains or is unbound (`on_engine=("abstain", "unbound")`; drop `"unbound"`
  where it fires on answers that are already right, as it did on spam below);
- a head answer below its question's line (`line=0.8`), because a head carries no abstain flag;
- a random share of everything else as an audit (`audit=0.05`).

Where the line sits is measured. Every head answer the teacher also answered (those below the line,
and the audits above it) is kept with its confidence and whether the two agreed, and the line is the
lowest confidence above which vega agreed with the teacher at least `trust` (0.95) of the time, over
at least `min_checked` (30) answers. As vega learns, the line comes down by itself; if the policy
changes, audits catch the disagreement and it goes back up. Checks never stop, so a line that went up
can always come back down. A head is never trusted on its confidence alone, since a head's confidence
is its probability, not a calibrated one. The learner holds a batch that sharply disagrees with what it
learned for someone to decide; with a teacher, the teacher is that someone, so held batches are learned
(`accept_held=True`; False leaves them for a person).

`vllm_teacher(base_url="http://localhost:8000/v1")` asks a model you serve yourself, with vLLM or any
OpenAI-compatible server (SGLang, llama.cpp server, Ollama, LM Studio). Nothing leaves your machines and
there is no cost per call:

```python
# vllm serve Qwen/Qwen3.8-27B --served-model-name teacher --max-model-len 8192
from vegaml_learn import Escalator, Learner, vllm_teacher

desk = Escalator(Learner(questions), teacher=vllm_teacher("http://localhost:8000/v1"))
```

The answer is constrained by a JSON schema that only admits the questions' own options. `thinking=True`
lets a thinking model reason first; for servers other than vLLM pass `extra_body={}`. Pick a teacher
clearly better than vega on the question: vega learns up to its teacher, not past it, and audits measure
agreement with the teacher, so they cannot catch a teacher that is wrong (measured below).

`claude_teacher(model="claude-opus-5")` asks Claude once per record, every escalated question at once,
with a fixed system prompt and a JSON schema that only admits the questions' own options. The record
and the escalated questions go to Anthropic's API and nothing else does. A refused record gets no
answer; with `fallbacks=True` (the default) Anthropic re-runs a declined record on its recommended
fallback model first.

### Measured: Claude Opus 5 as the teacher

12 public decision tasks of the kind companies automate, each as a stream of 400 records in arrival
order (not balanced by label) and about 150 held-out records the learner never saw. vega with the
0.8B reader, no starting set, starting `line=0.8`, `audit=0.05`, `trust=0.95`; Claude Opus 5 at low effort.
Gold labels only score: neither vega nor Claude sees them. Balanced accuracy (mean recall per answer),
start and after 400 records; teacher share over the first and the last 100 records of the stream.

| task | vega alone | vega + teacher | sent to the teacher |
| --- | --- | --- | --- |
| contract clause type, 100 options (LEDGAR) | 23.3 to **55.1** | 77.7 to 78.1 | 99% to 75% |
| assistant intent, 60 (MASSIVE) | 26.9 to **57.3** | 89.9 to 80.2 | 99% to 60% |
| banking intent, 77 (Banking77) | 33.3 to **47.7** | 86.1 to 84.4 | 100% to 94% |
| unfair terms of service (LexGLUE) | 48.0 to **75.5** | 73.7 to **83.4** | 59% to 78% |
| prompt injection (deepset) | 77.8 to **88.9** | 77.8 to **94.4** | 43% to 13% |
| unsafe prompt (Aegis 2.0) | 54.7 to **66.3** | 61.6 to **76.2** | 47% to 57% |
| offensive tweet (TweetEval) | 63.5 to **69.3** | 66.9 to **76.7** | 14% to 91% |
| review stars (Yelp) | 58.1 to 58.1 | 68.3 to 68.3 | 79% to 75% |
| news section (AG News) | 91.6 to 91.6 | 94.9 to 94.9 | 7% to 17% |
| spam (Enron) | 100.0 to 100.0 | 100.0 to 100.0 | 47% to 54% |
| jailbreak (ToxicChat) | 94.9 to 94.9 | 97.6 to 97.6 | 63% to 74% |
| toxic chat (ToxicChat) | 71.0 to 71.0 | 81.8 to 81.8 | 68% to 78% |

What it shows: vega learned the teacher's policy on 7 of the 12 tasks with no labels from anyone. Where
its confident answers came to agree with the teacher, the line came down and the teacher was asked less
(prompt injections from 43% to 13% of records, while the combined answer rose from 77.8 to 94.4). Where
they did not agree yet (unfair terms, offensive tweets), the line stayed up and more went to the
teacher, so the combined answer improved instead. The line is set where vega agrees with the teacher,
not where vega is right, so handing records to vega can cost accuracy while it learns: on assistant
intents the combined answer fell from 89.9 to 80.2 at 400 records (raise `trust` to hand over later).
On the two rare-answer tasks (jailbreaks are 2% of ToxicChat, toxic messages 7%) no head beat vega's
own engine on balanced accuracy, so the engine kept answering and the teacher kept being asked. Without
that bar, an earlier run of the same loop learned to answer "no" to every record, scored 97% accuracy on
jailbreaks while catching none, and stopped asking the teacher. All 12 streams together cost
$19 of Claude Opus 5 calls.

The same assistant-intent stream, run for 3,000 records:

| records seen | 500 | 1,000 | 1,500 | 2,000 | 2,500 | 3,000 |
| --- | --- | --- | --- | --- | --- | --- |
| sent to the teacher (last 250) | 64% | 54% | 52% | 56% | 50% | 43% |
| vega alone right (held out) | 72% | 75% | 82% | 85% | 84% | 85% |
| vega + teacher right | 87% | 81% | 86% | 84% | 86% | 86% |

vega alone went from 35% to 85%, about level with the combined answer, and the share sent to the
teacher fell from 100% to 43% and was still falling. It did not reach zero in 3,000 records. $14 of
Opus calls.

### Which teacher

The same 3,000-record stream, vega with the 0.8B reader unless noted, with different teachers. The
teacher alone is its own accuracy on the held-out records; vega learns up to it, not past it.

| teacher | teacher alone | vega alone at the end | vega + teacher | sent to the teacher at the end | API cost |
| --- | --- | --- | --- | --- | --- |
| Claude Opus 5 (`claude_teacher()`) | about 87% | 85% | 86% | 43% | $14.09 |
| Claude Haiku 5.5 (`claude_teacher("claude-haiku-5-5")`) | about 85% | 83% | 86% | 47% | $0.33 |
| Qwen3.8-27B on your own GPU (`vllm_teacher()`) | 83% | 79% | 83% | 40% | none |
| Qwen3.5-4B on your own GPU, vega with the 4B reader | 74% | 72% | 73% | 45% | none |
| Qwen3.5-0.8B on your own GPU | 3% | 3% | 3% | 33% | none |

A served 27B model teaches nearly as well as Claude with nothing leaving your machines. A teacher worse
than vega teaches it to be wrong: the 0.8B model, right 3% of the time on 60 intents, took vega from 35%
down to 3%, and audits could not catch it because vega agreed with its teacher.

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
pip install "vegaml-learn[serve]"
export VEGAML_LEARN_JWT_SECRET="$(openssl rand -hex 32)"
python -m vegaml_learn serve --questions questions.json --examples examples.csv \
       --workdir ./vega-state --host 0.0.0.0 --port 8000
python -m vegaml_learn token --subject helpdesk --scope decide feedback --days 30
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
