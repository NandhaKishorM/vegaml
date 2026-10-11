# Decision Index

[`decision-index`](https://github.com/apolinario/decision-index) is the reproduction kit for the
Decision Index, a community board for typed decision engines. It rebuilds the public suite from its
original sources, drives any engine through it and scores it with the board's own scorers.

`decision_index_engine.py` here is Vega as one of those engines.

## Running it

The suite is not redistributable, so it is built once from the upstream sources: about 7 GB of
downloads, 17 GB of working space, a Hugging Face login, and the HLE terms accepted on the Hub.

```sh
pip install "decision-index[transformers,rebuild] @ git+https://github.com/apolinario/decision-index"
pip install vegaml
export HF_HUB_DISABLE_XET=1

python -m decision_index suite rebuild --work work
python -m decision_index suite import \
    --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
    --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz \
    --gsm8k-rows work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz
```

Then, from a checkout of this repository so that `bench` is importable:

```sh
python -m decision_index pipeline \
    --engine bench.decision_index_engine:VegaEngine \
    --option model=0.8b --out runs/vega-08b
```

`--option model=4b` runs the 4B. `--option device=cuda|mps|cpu` picks the device, `--option views=2`
averages a second reading with the options reversed (steadier, about twice the cost; every published
figure is `views=1`). A short sample first is cheaper than finding out at request 90,000:

```sh
python -m decision_index suite sample --n 200 --out sample-200.jsonl.gz
python -m decision_index run --engine bench.decision_index_engine:VegaEngine \
    --option model=0.8b --rows sample-200.jsonl.gz --out runs/vega-sample
python -m decision_index score --results runs/vega-sample/results.jsonl
```

## What the adapter does, and what it refuses

The kit's rules are deliberately strict, and three of them bite here:

- **No truncation.** Vega's prompt builder quietly shortens a state that will not fit, an
  instruction past 256 tokens, and an option past whatever `opt_max_tokens` the checkpoint config
  carries (32 for the published checkpoints). The adapter reads all three budgets and raises
  `Unsupported` instead, which the suite counts as wrong. A refusal is an honest zero; a truncated
  prompt is a score for a question the model was never shown.
- **Every option gets a probability.** Vega already returns a full distribution, so this costs
  nothing. The probabilities are renormalised for float error only, never reordered.
- **Unanswered is wrong, and so is abstaining.** Vega's `abstain` flag is reported in
  `raw_output` and not acted on. It marks low confidence next to an answer that exists; there is
  nothing to withhold, and withholding it would score zero.

The kit calls the yes/no primitive `noul`, the name this answer type carried here before 0.5.0
renamed it to `boolean`. The two names meet in three places, all in the adapter: `to_vega` going
in, `to_kit_answers` coming out, and `_ttt_answer` for booleans a fitted head answers.

## What a run reports

`raw_output` carries `latency_ms`, `views`, and a per-question `readout`. A head still reads the
frozen backbone — `Vega.features()` is one pooled forward pass over the same prompt — so its
questions are timed and counted in `usage.input_tokens` like any other. A request answered wholly by
heads therefore reports a real latency and a non-zero token count, not `None` and `0`.

The provenance block's `mode` and `runtime()["mode"]` always agree, so a run's `environment.json`
cannot claim one readout while its results claim another.

One state is read once for every question in a request, so the state has to fit the tightest
question in it. That is why the budget check runs over all of them before any decision is made.

## The TTT readout

`--option mode=ttt` answers from a head fitted here, over the same frozen features, instead of from
the physics engine. It is the readout for label spaces the engine has never seen a shape of, and the
wide classification sets are exactly that case.

```sh
python -m decision_index run --engine bench.decision_index_engine:VegaEngine \
    --option model=0.8b --option mode=ttt --option examples=train-rows.jsonl.gz \
    --rows rows.jsonl.gz --out runs/vega-08b-ttt
```

Four rules keep it honest:

- **Examples come from you, from train splits.** The file holds rows in the suite's own format
  (`state`, `questions`, `expected`). The suite itself is evaluation data under 43 upstream
  licences; a head fitted on it would be grading its own homework.
- **One head per answer space, not per question key.** A space is the answer type plus the option
  texts, so all 3,080 BANKING77 rows share one head, and a different option set gets its own or
  none at all.
- **Heads are fitted before the run, never inside a timed request.**
- **A head at or below chance is discarded** and the engine answers that space instead. The package
  warns about this in `fit`; here it is enforced rather than printed.

Every answer records which readout produced it in `raw_output.readout`, and `environment.json`
carries the examples path, its sha256, and the per-space example count and cross-validated
accuracy. A submission that mixes readouts has to say so, and those two files are how a reviewer
checks it.

## Submitting

The board wants the public suite run in full, then a pull request against the kit's
`submissions/README.md` with the model repo, a Hub dataset holding the run directory, the engine and
commit, the hardware and the exact settings. `scores.json` has to say `"complete": true`, and the
maintainers re-score the file and re-run the model themselves on the two private parts, which are
50% and 30% of the Full score against the public suite's 20%.

Two things to know before spending the compute:

- The maintainers measure latency themselves on one RTX PRO 6000, single process, one request at a
  time. A model whose median, mean **or** 80th-percentile latency there is over 1,000 ms per request
  is not added to the board at all. Vega is comfortably inside that on short requests and the wide
  label sets are the ones to watch, so measure the tail before submitting, not after.
- The suite is evaluation data under 43 upstream licences. Do not train on it.
