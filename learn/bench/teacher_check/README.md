# A sabotaging teacher against the check set

What happens to a continuously learning vega when the teacher it learns from gives wrong answers on purpose,
and what `Escalator(check=...)` does about it. The numbers in [Checking the teacher](../../../docs/learn.md#checking-the-teacher)
come from here.

- `run.py`: one run. A task, a teacher (honest, or one of six ways of being wrong), a guard (none, the check
  set, the check set with a floor) and a seed. No model is called: the reads are cached and the honest teacher
  is a file of cached answers or the dataset label.
- `report.py`: the tables, as means over seeds 4 to 6, into `results/tables.md` and `results/summary.json`.
- `results/runs/`: one file per run, 226 of them.
- `opus_massive_answers.jsonl`: Claude Opus 5's answer to each MASSIVE record (the honest teacher on that task).

```bash
python run.py --task massive --teacher target_late --guard check --seed 4 --out results/runs/x.json
python report.py
```

## Tasks

| task | records | reader | honest teacher | targeted sabotage |
| --- | --- | --- | --- | --- |
| `massive` | MASSIVE assistant intents, 59 options | vega 0.8B | cached Claude Opus 5 answers, 88.8% right | six remove/off intents answered with their opposite (7% of traffic) |
| `cb` | CraigslistBargains, what is being sold, 6 options | vega 4B | the dataset label | every phone called electronics (11%) |
| `saas` | SaaS sales conversations, will it convert | vega 4B | the dataset label | in two of sixteen sales patterns, every conversion called a non-conversion (6%) |

Each run draws 200 check records, a starting set (400 for `massive`, 200 for the others), 600 held-out records
and a stream (1,950, 2,000 and 1,000) from one shuffle of the task. Seeds 0 and 1 were used while the rules
were being written; the tables use seeds 4 to 6, and the false-alarm table seeds 4 to 13.

## Data layout

The cached reads are not in the repository (a few gigabytes). `TEACHER_CHECK_DATA` points at a folder with:

```
massive.json            {"massive_intent_en": {"question", "stream": [{"id", "text", "gold"}], "test": [...]}}
massive.npz             X: one row of reader features per record, in stream then test order
massive_engine.json     vega's own answer per record, same order
data/cb.jsonl, data/cb.questions.json, feats/cb_4b/chunk_*.npz        (keys, X[record, question, feature])
data/saas.jsonl, data/saas.questions.json, feats/saas_4b/chunk_*.npz
```

For `cb` and `saas` vega's own engine answers were not kept, so a stand-in engine that always abstains is used
and the heads fitted on the starting set answer from the first record.

## Limits

The saboteur is simulated by changing a real teacher's answers by rule. It cannot tell check records from
traffic. The check labels are dataset labels, so the reviewer is perfect. Three tasks, three seeds.
