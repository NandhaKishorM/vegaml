<p align="left">
  <img src="assets/vega-icon.svg" width="68" height="68" alt="Vega: a particle at rest in the deeper of two wells" />
</p>

# vegaml

Typed decisions from a frozen language model and a small physics engine. You give it a **state** and a
**question whose answer type is fixed in advance**; it returns a value your code can use directly: a
choice, a 0 to 1 score, or the probability that a statement is true. Each carries a calibrated
probability, a conformal answer set and an explicit abstain flag.

**vega-U: one 14.3M engine, any frozen reader, from 0.8B to 27B · 73,728-token context · images ·
runs on your own hardware.** No text is generated anywhere in the path.

![vega-U: four readers, one byte-identical engine](assets/vega_u_benchmarks.png)

<sub>The default model. Four frozen readers behind one byte-identical 14.3M engine; the readers
marked "attached cold" were never part of the engine's training and scored highest anyway. The
full per-source board, the method and the caveats are in [docs/vega-u.md](docs/vega-u.md); the
full per-source board is in the [Benchmarks](#benchmarks) section below.</sub>

```bash
pip install vegaml
```

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZxS_Ee4-oyfo8ePKiAgbehv5I1Rh3F7O?usp=sharing)

Inference on a single T4, end to end, in the browser.


<sub>Releases are cut by tagging `vX.Y.Z`: CI builds, checks the version matches the tag, and
publishes through PyPI Trusted Publishing, so no API token exists in this repository.</sub>

## Quick start

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

Both questions are answered from **one** read of the state.

### Attach your own model, without retraining anything

From 0.12, a base model the checkpoint ships no port for is something you attach yourself,
locally, with `vegaml.attach`. The engine is never retrained, the reader is never
fine-tuned, and none of the original training data is needed: the alignment is
self-supervised. The checkpoint's own root reader generates the shared-space target for any
text, so the port (what a port is: the vega-U section, next) is fitted from running both
readers over the same prompts, on text you bring.

**The structure of the text file**, and yes, the content can be almost anything:

| rule | what it means |
| --- | --- |
| one document per line | a line is one text: an email, a ticket, a review, a log, a contract, a chat transcript |
| plain text only | UTF-8, no labels, no headers, no JSON, no markup needed; `attach` never reads answers here |
| long documents stay on one line | replace newlines inside a document with spaces; blank lines are skipped |
| any length | a sentence to tens of thousands of tokens; include genuinely long lines if you will read long records |
| content: your domain | whatever your model will actually read in production; the languages your readers handle are fine |
| diverse beats big | 1,000 varied lines beat 10,000 near-duplicates; do not repeat one template |

"Can it be anything?" Almost: it needs no labels, no questions and no structure, because the
alignment is self-supervised, the anchor reader generates the target for every line. The one
thing that matters is that the lines look like your production traffic, since the port tunes
to the corpus it is fitted on. The shipped starter corpus looks like this:

```text
I was charged 320.00 twice for the booking system all week. Please refund the duplicate and confirm by email, or I will have to cancel the account.
Review: disappointed with the router. It looked fine out of the box but failed with an authentication loop on the second day, and support has not replied this morning.
Incident report, the north plant: at 06:40 the sensor on line 2 reported a blank screen. The shift lead paused the belt, cleared the jam, and production resumed at 07:05. No injuries.
Full incident timeline, for the record. <...a several-thousand-token document, still one line...>
```

```python
import vegaml

# a synthetic starter corpus ships in the repo; swap in strings from YOUR domain
# (on Colab, fetch it first:
#  !curl -sO https://raw.githubusercontent.com/NandhaKishorM/vegaml/main/examples/sample_attach_texts.txt
#  and open "sample_attach_texts.txt")
texts = [t for t in open("examples/sample_attach_texts.txt") if t.strip()]

report = vegaml.attach(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",   # any instruction-tuned causal LM
    texts=texts,
    out_dir="vega-u-smollm2-1.7b",
)
print(report["ridge_cosines"])               # how faithfully the port landed

v = vegaml.load("vega-u-smollm2-1.7b")       # same engine, byte for byte, new eyes
```

Three steps happen under that call:

1. **Anchor pass.** The 0.8B root reader reads every text under a fixed set of probe
   questions; its features ARE the shared-space coordinates, so this writes the targets.
   The anchor is then freed before your reader loads.
2. **Port fit.** Your reader reads the same prompts, its features are standardized, and one
   linear map per tapped layer is fitted in closed form (ridge, lambda picked on a holdout),
   then refined for a few hundred steps so the frozen engine decides the same way through
   the new port as through the anchor. Both steps are label-free.
3. **A servable dir** is written: the unchanged engine, the new port, the extraction
   statistics and a config `load()` reads like any shipped reader.

What to bring, and what it costs:

* **Texts**: 1,000+ strings from your domain is a good corpus, 100 is the floor; diverse
  beats big. Include long records if you will read long records. More text raises the
  reported holdout cosines, and well under ~0.8 usually means too little or too narrow.
* **GPU memory**: enough for your reader itself in bf16, which is the whole bill. Roughly
  2 GB per billion parameters plus a few GB of headroom: a 1B reader attaches on any 8 GB
  card, a 7B to 9B on 24 GB, a 27B wants 80 GB or more. The anchor (0.8B) and the engine
  (14.3M) are noise next to that. CPU and Apple silicon (MPS) work too, just slower.
* **Time**: dominated by reading the texts through your reader, 6 probe prompts per text.
  A 7B reader over 1,000 short texts is tens of minutes on a 24 GB GPU; the ridge and the
  refine are minutes. `max_len=` caps prompt length if your texts are long and you want a
  faster first fit.
* **Labels, only for calibration.** The attached dir answers immediately but its
  probabilities are raw until you calibrate on your own labelled examples, which beat any
  data we could ship because they match your distribution:

```python
import json

# a labelled starter set ships next to the corpus: the question sheet plus
# 600 [state, labels] pairs, generated with ground-truth labels
# (on Colab:
#  !curl -sO https://raw.githubusercontent.com/NandhaKishorM/vegaml/main/examples/sample_calibrate.json)
cal = json.load(open("examples/sample_calibrate.json"))

vegaml.calibrate("vega-u-smollm2-1.7b", cal["examples"], cal["questions"])
```

And then the attached, calibrated model answers exactly like the Quick start one:

```python
v = vegaml.load("vega-u-smollm2-1.7b")

out = v.decide(
    "The dishwasher stopped working this morning and now shows error E-114. "
    "I already restarted it twice. Send a technician this week or refund me.",
    cal["questions"])                                # or any questions of your own

print(out["answers"]["complaint"]["boolean"])        # p(yes), high here
print(out["answers"]["category"]["choice"])          # 'technical'
print(out["answers"]["category"]["probs"])           # calibrated on YOUR examples, sums to 1
print(out["answers"]["urgency"]["score"])            # 0 to 4
print(out["answers"]["complaint"]["abstain"])        # False -> the engine stands behind it
```

`sample_calibrate.json` is also the format to copy for your own data: `questions` maps each
id to its definition, and every example is `[state, {question_id: label}]`, the same shapes
`Vega.fit` takes:

```json
{
 "questions": {
  "complaint": {"type": "boolean", "instructions": "Does the text describe a problem, fault or complaint?"},
  "category":  {"type": "choice",  "instructions": "Which area does this text belong to?",
                "criteria": {"billing": "money, invoices, charges or subscriptions",
                             "technical": "devices, software or technical faults",
                             "operations": "logistics, facilities, incidents or travel",
                             "other": "anything else"}},
  "urgency":   {"type": "score",   "instructions": "How urgent is the text?",
                "criteria": ["no urgency at all", "slightly time-sensitive", "moderately urgent",
                             "clearly urgent", "critical, immediate action needed"]}
 },
 "examples": [
  ["I was charged 49.99 twice for my laptop over the weekend. Please refund the duplicate and confirm by email, or I will have to cancel the account.",
   {"complaint": true, "category": "billing", "urgency": 3}]
 ]
}
```

**The structure of the calibration JSON**, piece by piece:

| piece | what goes there |
| --- | --- |
| `questions` | the decisions your business actually makes, each with an id, a `type` and `instructions` written as you would brief a person |
| `type: "boolean"` | a yes or no decision; labels are `true` / `false` |
| `type: "choice"` | one of your own categories; `criteria` maps each label to a one-line description, and the label in an example is the key, like `"billing"` |
| `type: "score"` | an ordered scale; `criteria` lists the levels from lowest to highest and the label is the level's index, `0` to `4` here |
| `examples` | `[state, {question_id: label}]` pairs from your history; the state is the text (or a JSON object, like the Quick start email) and the labels are what actually happened |
| partial labels | fine: an example may label only some of the questions |

This is where your own business plugs in, whatever it is, because the labels are just your
historical outcomes. A finance team would write it like this, straight from records it
already keeps:

```json
{
 "questions": {
  "chargeback": {"type": "boolean", "instructions": "Will this transaction be disputed by the cardholder?"},
  "action":     {"type": "choice",  "instructions": "What should happen to this payment?",
                 "criteria": {"approve": "clear it automatically",
                              "review": "route to a human analyst",
                              "escalate": "freeze and escalate to risk"}},
  "risk":       {"type": "score",   "instructions": "How risky is this transaction?",
                 "criteria": ["routine", "slightly unusual", "noteworthy", "suspicious", "almost certainly fraud"]}
 },
 "examples": [
  [{"merchant": "QUICKCASH LTD", "amount": "2,450.00", "country": "new for this card",
    "note": "third high-value charge tonight, customer usually spends under 60"},
   {"chargeback": true, "action": "escalate", "risk": 4}],
  [{"merchant": "GROCERY MART", "amount": "54.20", "country": "home",
    "note": "weekly pattern, same store"},
   {"chargeback": false, "action": "approve", "risk": 0}]
 ]
}
```

A support desk would label which team took the ticket, an insurer which claims were
escalated, a lender which applications defaulted: the ids, the categories and the scale are
yours to define, and the engine answers THOSE questions with probabilities calibrated on
THOSE outcomes.

None of this needs `attach`. The same call recalibrates the shipped readers, so the shortest
path to vega-U on your own business is the default Qwen3.5-0.8B plus your JSON:

```python
import json
import vegaml

# (on Colab, same starter file:
#  !curl -sO https://raw.githubusercontent.com/NandhaKishorM/vegaml/main/examples/sample_calibrate.json)
cal = json.load(open("examples/sample_calibrate.json"))   # or your own file, same shape

v = vegaml.load()                                         # the shipped Qwen3.5-0.8B reader
report = vegaml.calibrate(v, cal["examples"], cal["questions"])
print(report["ece_before"], "->", report["ece_after"])

out = v.decide("The dishwasher stopped working this morning and now shows error E-114.",
               cal["questions"])
print(out["answers"]["category"]["choice"], out["answers"]["urgency"]["score"])
```

The refit lands on the loaded model immediately and is written into your local copy of the
checkpoint, so later `vegaml.load()` calls on this machine keep it. The Hub copy is never
touched: a fresh download restores the shipped calibration.

Roughly 500 labelled examples per question type is usable and 1,000 to 2,000 is where the
numbers settle. The refit covers the temperature (physics-conditioned where it beats a plain
one on a holdout), the conformal prediction-set thresholds, the abstain thresholds and the
unbound threshold, written into the dir and applied in place. One honest caveat: a port
fitted on your corpus is tuned to your domain, and the published benchmark numbers only
describe the ports we ship.

### Devices

With no `device` argument, loading prefers CUDA, then Apple silicon MPS when PyTorch reports it
available, then CPU. Override it with `vegaml.load(device="mps")` or `vegaml.load(device="cpu")`.
An explicit MPS request raises a clear error when MPS is unavailable.

The text backbone uses float32 on MPS and CPU; CUDA keeps its capability-based bf16/fp16 policy.
The backbone and decision engine stay on the selected device. MPS is experimental: start with the
0.8B model and short text, and verify decisions on your own Mac before relying on them. The full
runtime uses more memory than the downloaded weights; long contexts and the 4B model need more.

### Two readouts, one set of features

`mode` picks how the pooled features are turned into an answer.

| mode | needs examples | carries | use it when |
|---|---|---|---|
| `"auto"` *(default)* | no, uses them if present | whatever the chosen readout carries, plus `readout` | the normal case |
| `"engine"` | no, zero-shot | calibration, conformal sets, abstain | reproducing every published figure below |
| `"ttt"` | yes, 6 minimum | nothing, and it answers even when it should not | you have labels for exactly this question |
| `"both"` | yes | both, side by side | deciding which to trust |

`"auto"` answers each question with its head where it has one worth using, and with the engine
everywhere else. Fit nothing and it **is** the engine, flags and all, which is why it is safe as the
default and why it still reproduces the figures below. Every answer says which readout produced it:

```python
examples = [({"body": "cancel my account"}, {"churn": "true"}),
            ({"body": "how do I export?"},  {"churn": "false"})]            # ... 20 or so

report = v.fit(examples, questions)
print(report["churn"]["cv_accuracy"], report["churn"]["used_by_auto"])

out = v.decide(state, questions)                 # auto
print(out["answers"]["churn"]["readout"])        # 'ttt' here, 'engine' for anything unfitted
```

`fit` cross-validates every head and **discards any that does not beat chance**, because such a head
has learned nothing and its probabilities are noise. `"auto"` is stricter still: it uses only heads
whose cross-validated accuracy beats both chance and the engine's own accuracy on the same examples,
so a head that loses to the engine or was fitted on too few examples to cross-validate is left to
the engine.

If what distinguishes your examples sits in a question's `instructions` rather than in the state, as
it does in most intent classification, pass the question per example or every example gets an
identical prompt and the head is fitted on noise:

```python
examples = [(state, labels, {"intent": dict(question, instructions="...this row's text...")}), ...]
```

**Changed in 0.8.0.** The image path works. `decide_image` reached for a module that ships only in
the training tree, so it raised `ModuleNotFoundError` in every installed copy; the `vision` extra now
also carries `torchvision`, which the backbone's processor loads on the way in. Each adapter is gated
on its own threshold rather than on the first adapter's. A head at or below chance is no longer
served, `fit` takes a per-example question, and the TTT readout names what went wrong instead of
raising `max() arg is an empty sequence`. `label_ids` and `MODES` are exported.

**Changed in 0.7.0.** `mode="auto"` is the new default. `"engine"` is unchanged and still what every
published figure measures, so pass it explicitly when reproducing them.

### Images

```python
from PIL import Image

v.decide_image(Image.open("invoice.png"),
               {"kind": {"type": "choice", "instructions": "What kind of document is this?",
                         "criteria": {"invoice": "a bill", "receipt": "proof of payment",
                                      "form": "a document to be filled in", "other": "none of these"}}})
```

The backbone is multimodal and its vision encoder is frozen with the rest, so a picture takes the place
of the state text in the same prompt and the same spans are pooled. The benchmark figures below are
text; the image path is functional but not covered by them.

`image` may be a PIL image, a path, or an `HxWx3` numpy array. Answers come back in the same shape as
`decide`, calibration and all: `prediction_set`, `abstain`, `unbound`, `settle_steps` and `energy` are
present exactly as on the text path. `decide_image` always runs the engine and marks every answer
`readout: "engine"`, because a TTT head reads text features and has nothing to say about pixels. To
measure what the picture contributed, `decide_image_text` answers the same questions through the same
path with the image withheld.

### Audio

*Added in 0.15.0.*

```python
answer = v.decide_audio("call.wav",
    {"act": {"type": "choice", "instructions": "What is the caller asking for?",
             "criteria": {"refund": "money back", "status": "where is my order",
                          "cancel": "stop the subscription", "other": "none of these"}}})
answer["transcript"]   # the words the decision was made about
```

Speech goes through a small local transcriber and the transcript rides the ordinary text path, so
everything above applies unchanged: fitted heads, calibration, conformal sets, the escalation band.
That is the difference from the image path, where pixels bypass the heads. The transcript and the
checkpoint that produced it come back in the response, because a decision made on words you never
saw cannot be audited.

The default transcriber is `distil-whisper/distil-small.en` (166M, English), downloaded once and
loaded on first use, so a text-only caller never pays for it. Name any Hugging Face ASR checkpoint
instead: `asr="openai/whisper-tiny"` (39M) for multilingual audio with `language="de"` and friends,
`asr="UsefulSensors/moonshine-tiny"` (27M) for the smallest English front end. Everything runs on
your machine; no audio leaves the process.

`audio` may be a path, raw audio-file bytes, or a float waveform with `sampling_rate=`. Decoding a
file needs the extra:

```
pip install "vegaml[audio]"
```

### 73,728-token context

`max_len` is 73,728 tokens, and the architecture is built for it rather than merely permitting it:

* **Read-once prefix caching.** A state of at least 4,096 tokens is encoded **once**; every question
  about that state continues from a copy of that prefix. Twelve questions about one 73k-token contract
  cost roughly one read, not twelve.
* **A memory reader.** Long states are cut into 16 pieces, mean-pooled at the deepest feature layer and
  fed to the engine alongside the spans, so evidence late in a long document still reaches the decision.
* **No silent truncation.** An input that will not fit is refused, never quietly shortened.

### Answer types

Three, named in every question's `type` field:

| type | returns |
|---|---|
| `choice` | one option from a set you define |
| `score` | an integer level against ordered criteria, plus the expected value |
| `boolean` | the probability that a statement holds |

**Changed in 0.5.0.** `boolean` was renamed from a coined word that told a reader nothing. A
question using the old name is rejected with the valid types listed, rather than silently
reinterpreted. Checkpoints published before the change key their conformal thresholds by the old
name; the loader remaps them, so an older checkpoint still loads with its prediction sets intact.

## vega-U: one engine, any reader

**vega-U is the default model**: `vegaml.load()` with no argument loads it with the 0.8B
reader, and the size names ("4b", "gemma-1b") pick other readers of the same engine. The
universal line decouples the engine from the backbone. A vega-U checkpoint directory is a
normal vega model dir plus two files: the backbone's **port** (`port.safetensors`, one small
linear map per tapped layer into a shared 1024-d space) and the raw feature statistics of its
extraction (`stats.npz`). The engine lives in the shared space and is byte-identical across
every backbone; only the port and the calibration are per reader.

Loading one is the same call as always, and **changing the base model is naming it**,
nothing else:

```python
v = vegaml.load()                                             # the default reader
v = vegaml.load(backbone="Qwen/Qwen3.5-4B")                   # same engine, the 4B reads
v = vegaml.load(backbone="google/gemma-3-1b-it")              # gated reader: set HF_TOKEN
v = vegaml.load("outputs/universal/per_backbone/Qwen3.5-4B")  # or any dir on disk
```

On a universal checkpoint, `backbone=` names WHICH reader: the dir carrying that reader's
port and calibration is selected by its name (repo root or subfolder), and the reader's own
weights come from their original repository. The token question belongs to the reader, not to
vega-U: Qwen loads with no token, Gemma 3 is gated so `HF_TOKEN` must be set and the licence
accepted. Asking for a reader the checkpoint has no port for fails with the list of readers
it ships.

The package recognises the `port` key in `vega_config.json` and inserts the
standardize-then-port step in front of the engine automatically; `decide()` and everything
downstream are unchanged. A new reader is attached by fitting its port (closed-form fit, a
short refine, a calibration refit), never by retraining the engine or fine-tuning the
reader, and from 0.12 you can do that yourself with `vegaml.attach`, above.

Measured across the October 2026 runs, every cold-attached reader scored above the readers
the engine trained with, on an engine that is checksum-identical behind all four:

| reader | how | held-out acc | zero-shot acc | ECE | AURC | acc @ 50% cov |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | attached cold | **0.811** | **0.629** | 0.021 | **0.053** | **0.976** |
| Qwen3.5-4B | attached cold | 0.791 | 0.568 | 0.017 | 0.063 | 0.965 |
| Qwen3.5-0.8B | trained jointly | 0.765 | 0.534 | 0.010 | 0.073 | 0.956 |
| gemma-3-1b-it | trained jointly | 0.738 | 0.504 | 0.007 | 0.093 | 0.927 |

The gains land where reading matters (long context 0.831 to 0.941 from the 0.8B to the 27B,
ARC 0.647 to 0.922) and saturated tasks stay flat, which is the design's claim made
measurable. The full per-source board, the checkpoint layout, the caveats and the version
history are in [docs/vega-u.md](docs/vega-u.md).

```python
vegaml.load()                  # vega-U, 0.8B reader (the repo root)
vegaml.load("4b")              # vega-U, Qwen3.5-4B reader
vegaml.load("gemma-1b")        # vega-U, Gemma 3 1B reader
vegaml.load("owner/some-repo") # anything else on the Hub, including the earlier
                               # per-size checkpoints (nandakishorm/vega-08b-public-intents)
vegaml.load("/path/on/disk")   # or a local directory
```

A private or gated repository needs a Hub token, and without one the Hub answers 401, which it
reports identically to "no such repository". `fetch` turns that into one sentence naming which of
the two it is and where it looked for a token. `load()` picks one up from `HF_TOKEN` in the
environment, so nothing has to be threaded through the call; pass `token=...` to override it, or
`token=False` to ignore the environment and fetch anonymously. It stays optional: the published
checkpoints are public and load with no token set.

```python
vegaml.load("owner/private-repo")               # uses $HF_TOKEN if it is set
vegaml.load("owner/private-repo", token=tok)    # or pass it explicitly
```

## Beyond the Python API

Each surface is one extra; none changes the core, and [docs/](docs/index.md) covers them all.

| Surface | Install | One line |
| --- | --- | --- |
| [CLI](docs/cli.md) | ships with `vegaml` | `vegaml "state" --boolean "churn=Threatens to leave?"`, JSONL batches on one load |
| [HTTP](docs/http-api.md) | `vegaml[serve]` | `vegaml-serve`: POST /v1/decide returns exactly what decide() returns |
| [MCP](docs/mcp.md) | `vegaml[mcp]` | `vegaml-mcp-server`: decide() as a tool an agent can gate on |
| [Structured](docs/structured.md) | `vegaml[structured]` | a pydantic model as the question sheet and the validated answer |
| [ONNX](docs/onnx.md) | `vegaml[onnx]` | export the engine, run it anywhere onnxruntime does |
| [MLX](docs/mlx.md) | `vegaml[mlx]` | the engine on Apple silicon without torch in the process |
| [Docker](docs/docker.md) | repo | `docker compose up`: the HTTP server as a container |

Runnable starts for each live in [examples/](examples/).

## The earlier checkpoints: classic vega in two sizes

Before the universal line, each size was its own checkpoint married to its backbone. They remain
fully supported and loadable by their repository id
(`vegaml.load("nandakishorm/vega-08b-public-intents")`) and take the same code path.

| | 0.8B | 4B |
|---|---|---|
| backbone, frozen | Qwen3.5-0.8B | Qwen3.5-4B |
| engine, trained | 14.3M params, 57 MB | 30.9M params, 124 MB |
| task adapter | 0.84M params | 1.43M params |
| per decision | **22 ms** | 28 ms |

The practical trade-off is the backbone, not the engine: `Qwen/Qwen3.5-4B` is 9.34 GB against
1.77 GB for the 0.8B and takes 61 seconds to load against 27. Pick the 4B when the machine has
the room and the task has more than a handful of labels; pick the 0.8B when it does not.

## Fine-tuning

[`notebooks/vega_finetune_2xT4_kaggle.ipynb`](notebooks/vega_finetune_2xT4_kaggle.ipynb) trains a
gated adapter end to end on Kaggle's 2×T4, and
[`notebooks/vega_finetune.py`](notebooks/vega_finetune.py) is the script it writes to disk and runs,
so the same three commands work on any machine with a GPU.

The language model stays frozen. What is trained is a rank-32 adapter on seven engine projections
plus its gate, about **0.84M parameters**, on `dair-ai/emotion` as a typed `choice` question, with
gate negatives drawn from `ag_news` so the adapter learns when *not* to fire. An adapter that always
fires is not routable. Thermal calibration and the conformal thresholds are refitted afterwards on a
validation split that training never sees, and travel with the adapter.

### The three stages

```bash
pip install vegaml datasets

# 1. frozen features, sharded across both GPUs, cached to disk. The long part.
torchrun --nproc_per_node=2 vega_finetune.py extract --batch 32

# 2. the adapter, on those cached features. Minutes.
python vega_finetune.py train --epochs 3 --batch 64 --rank 32 --lr 3e-4

# 3. base engine against the adapter on the held-out split
python vega_finetune.py eval --batch 64
```

Only stage 1 uses both GPUs, because only stage 1 benefits. The frozen read happens exactly once per
row and is the expensive part; training then runs on cached features, where an epoch is seconds on
one GPU. Launching stages 2 and 3 under `torchrun` would buy nothing, so they are not.

### Settings

| variable | default | what it does |
|---|---|---|
| `VEGA_SIZE` | `0.8b` | which size to tune, `0.8b` or `4b` |
| `HF_TOKEN` | unset | needed for a private or gated checkpoint, and to push |
| `VEGA_MODEL_REPO` | the size name | a different checkpoint repository or a local path |
| `VEGA_BACKBONE` | per size | a different frozen backbone |
| `VEGA_WORK` | `/kaggle/working` | where features, the adapter and results are written |

The two backbones pool to different widths, so each size keeps its own feature cache
(`features-0.8b`, `features-4b`) and its own output directory. Switching `VEGA_SIZE` never mixes the
two or silently reads the wrong cache.

In the notebook these live in one settings cell near the top, which also reads `HF_TOKEN` from
Kaggle Secrets when you have added one and carries on when you have not.

### Pushing what you trained

```bash
export HF_TOKEN=...
python vega_finetune.py eval --batch 64 --push-to your-name/vega-emotion-adapter
```

Only the adapter and its metadata go up, a few hundred kilobytes: no backbone and no engine. The
repository is created **private** unless you add `--public`. With no token the push is skipped with
a logged line rather than failing the run. `VEGA_PUSH_TO` sets the destination if you would rather
not pass the flag.

### Using the result

`adapter-<size>/` holds `emotion.safetensors` and `emotion.json` in the layout the loader already
reads. Drop both into a checkpoint's `adapters/` directory and they load automatically. The gate
decides per question whether they apply, and the adapter's own calibration travels with it.

```python
import vegaml

v = vegaml.load("your-copy-of-the-checkpoint")   # adapters/ picked up automatically
v.decide("I can't believe they remembered my birthday",
         {"emotion": {"type": "choice",
                      "instructions": "Which emotion does this message express?",
                      "criteria": {"sadness": "...", "joy": "...", "love": "...",
                                   "anger": "...", "fear": "...", "surprise": "..."}}})
```

The adapter records the SHA-256 of the base engine it was trained against, and the loader refuses it
against a different one rather than silently producing nonsense.

### What has actually been run

All three stages were run end to end before this was committed, but on a laptop and a **96-row
slice**, not on two T4s: extract → train → eval completed, two-rank sharding was checked to
recombine each split exactly once, and the adapter moved test accuracy 0.313 → 0.542 on 48 held-out
items. Those numbers are a smoke test, not a result. The full 16k run on Kaggle is the real one.

## Benchmarks

The current numbers are the universal (vega-U) runs: four readers behind one byte-identical
engine, summarized in the figure at the top of this page and measured in full below.

![the full vega-U benchmark board](assets/vega_u_full_board.png)

The 27B's held-out split is capped at 400 records per source (n=21,008); the smaller readers are
scored on the full split (n=50,676). Method, caveats and version history:
[docs/vega-u.md](docs/vega-u.md). Each published reader dir ships its raw per-source detail as
`eval_report.json`. The harness notebook
[`notebooks/vega_benchmark_2xT4.ipynb`](notebooks/vega_benchmark_2xT4.ipynb) reruns the live
comparison suite on a free Kaggle session.

## How it works

One prompt, one forward pass, then physics.

**1. One formatted prompt.** The state, the question and every candidate answer are written into a
single sequence:

```
Observation: {state} Measurement ({type}): {instructions} Possible outcomes: * {option 1} * {option 2} Outcome:
```

The option text is genuinely part of the model's input, not metadata kept outside it. No system prompt,
no chat template, no demonstrations, and the gold answer never appears. During training it exists only as a label.

**2. One forward pass, five pooled vectors.** The frozen language model runs **once** over that
sequence. Vega then pools different token spans from the *same* hidden states: the situation span, the
question span, one span per answer option, and the final token. Three questions about one state mean
three prompts and three passes, except for the long-input case above, where the prefix is shared.

**3. Pooled vectors become initial conditions.** The situation vector projects to a world latent and
gets a bounded nonlinear nudge; the question and the final token project to a probe and an impulse.
Together they place a particle at position `z₀` with momentum `p₀` in a 64-dimensional decision space.

**4. Every candidate answer becomes a valley.** Each option's own features produce a Gaussian well with a centre `c_k`, a depth `a_k` and a width `σ_k`:

```
U(z) = ½κ‖z‖² − Σ_k a_k · exp( −‖z − c_k‖² / 2σ_k² )
```

The quadratic term keeps the particle bounded; each well pulls it toward one answer. Score levels sit on
a one-dimensional rail, so ordinal neighbours are physical neighbours.

**5. The particle rolls and settles.** Damped Hamiltonian dynamics (symplectic Euler with friction and
a learned state-space thermostat) for a fixed, small step budget, with early exit once it has settled.
A decision is a short simulation with constant cost, not a sampling loop, so it is deterministic.

**6. Where it settles is the answer.**

```
E_k = ‖z_T − c_k‖² / 2σ_k² − log a_k            P = softmax(−E / τ)
```

### What is new here

**The architecture.** The language model is frozen and never trained. It is perception only, read at
two intermediate layers and cut off above the deepest one, so a decision never pays for the layers above
it and the LM head never runs. Everything learned lives in a 55 MB engine whose state is a *physical*
one: a position and a momentum, not a logit vector.

**The training method.** The engine is trained on the settling behaviour, not on next-token likelihood:
a counterfactual objective pairs items that share an answer space, and a one-step world-dynamics block
is trained to imagine the next state from the current one, so the latent carries what happens next
rather than only what was said. Two low-rank adapters (rank 32) attach to seven engine projections, and
a per-question sigmoid gate decides **for each question independently** whether an adapter contributes.
The gate value and the chosen adapter are returned with every answer, so routing is auditable rather
than implicit.

**The calibration method.** The readout temperature is not a constant. It is predicted per decision from
the physical state the particle ended in:

```
log τ = b + w · [ log(1 + residual kinetic energy),
                  log(1 + distance to the nearest well bottom),
                  fraction of the step budget used,
                  log (number of options) ]
```

A particle still in motion, or stopped far from every well, is an uncertain decision and gets a hotter
temperature. Each adapter route carries its own calibration vector. On top of that, every answer has a
split-conformal set at a chosen risk level, an **abstain** flag below a fitted confidence floor, and an
**unbound** flag when the particle settled far from every well, which is the model's own way of saying the
question is outside what it knows.

### Related work

A frozen language model feeding a trainable module described as a differentiable physics engine in a
learned latent space was proposed for physical reasoning in **CWMI**, [Inducing Causal World Models in
LLMs for Zero-Shot Physical Reasoning](https://arxiv.org/abs/2507.19855) (Sharma et al., 2025). The
separation of a frozen reader from a trained dynamics module is the same idea, and the resemblance is
worth stating plainly.

What differs is what the dynamics are *for*. CWMI learns physics as subject matter: its latent state
is meant to carry position, velocity, mass and material, it is supervised against ground-truth final
states from video and a simulator, and it answers physical-reasoning questions (PIQA, 89.4%). Its
module is a 12-layer Transformer decoder of roughly 256M trainable parameters trained on 8 H100s.

Vega uses dynamics as the inference mechanism rather than as a model of the world. Its 64-dimensional
space has no physical referent; the wells are built from the candidate answers of whatever question
was asked, so the landscape is rebuilt per question rather than learned once. The engine is 14.3M
parameters with no attention anywhere, and the motion is an explicit damped Hamiltonian integrated
with symplectic Euler against a written-down potential, not a learned state transition. The answer is
the lowest-energy well, and the residual kinetic energy at the end is what sets the temperature, which
is the part that makes the probabilities mean something. CWMI reports accuracy; it has no calibration,
conformal or abstain machinery, which is most of what is claimed here.

## Security

A checkpoint is data from a third party, so the inference code lives in this package and is reviewed
with it. `vegaml.load()` downloads **only** `vega_config.json`, `engine.safetensors`, the adapter
files and `config.json`, which exists so the Hub can count downloads and which nothing reads;
nothing fetched at runtime is imported or executed, and weights load through safetensors, never
pickle. Tests enforce that allow-list, and CI fails on any `pickle`, `torch.load`, `eval`, `exec`,
`subprocess` or `sys.path` insertion reaching the shipped package.

## Licence

Apache 2.0.
