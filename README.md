<p align="left">
  <img src="assets/vega-icon.svg" width="68" height="68" alt="Vega: a particle at rest in the deeper of two wells" />
</p>

# vegaml

Typed decisions from a frozen language model and a small physics engine. You give it a **state** and a
**question whose answer type is fixed in advance**; it returns a value your code can use directly: a
choice, a 0 to 1 score, or the probability that a statement is true. Each carries a calibrated
probability, a conformal answer set and an explicit abstain flag.

**800M or 4B parameters · 73,728-token context · images · runs on your own hardware.** No text is
generated anywhere in the path.

![Where this model beats a hosted reference](assets/vega_wins.png)

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

v = vegaml.load()            # the 0.8B, mode="auto": both are the defaults
# v = vegaml.load("4b")      # the 4B, same repository, same interface

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
whose accuracy is *known* to beat chance, so one fitted on too few examples to cross-validate is
left to the engine rather than trusted unmeasured.

If what distinguishes your examples sits in a question's `instructions` rather than in the state, as
it does in most intent classification, pass the question per example or every example gets an
identical prompt and the head is fitted on noise:

```python
examples = [(state, labels, {"intent": dict(question, instructions="...this row's text...")}), ...]
```

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

## Two sizes

Both live in the same checkpoint repository and take the same code path. `load()` with no argument
gives you the 0.8B, which is the default and the model every Jev comparison on this page measures.

| | 0.8B | 4B |
|---|---|---|
| backbone, frozen | Qwen3.5-0.8B | Qwen3.5-4B |
| engine, trained | 14.3M params, 57 MB | 30.9M params, 124 MB |
| task adapter | 0.84M params | 1.43M params |
| with the task adapter | 0.763 | **0.803** |
| soft accuracy | 0.680 | **0.725** |
| calibration error | 0.026 | **0.019** |
| per decision | **22 ms** | 28 ms |

Measured on the `LocalLLaMA/typed-decisions` test split, 2,050 decisions, each checkpoint's own run
of the same harness. Per workflow with the adapter, the 4B leads on five of six: apibank 96.5 against
89.5, when2call 94.6 against 90.4, appliances 89.1 against 85.1, banking77 74.9 against 71.4, esci
64.4 against 56.4. The 0.8B wins clinc150, 88.0 against 86.7.

So the 4B is the better model and the 0.8B is the default anyway, because 22 ms against 28 ms and 57
MB against 124 MB decides more deployments than four points of accuracy does.

```python
vegaml.load()                  # 0.8B
vegaml.load("4b")              # 4B
vegaml.load("owner/some-repo") # anything else on the Hub
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

The figure at the top of this page is the table below. Item-paired against a live Jev 1.13.0 over
identical items, no tuning on the evaluation data. These are the areas where this model leads; the
reference leads on most others, particularly reranking and multi-step reasoning.

Every figure here is `mode="engine"`, zero-shot, with no head fitted. That is also what `"auto"`
gives you until you call `fit`, so these numbers are what the default produces out of the box.

| task | this model | Jev 1.13.0 |
|---|---|---|
| Phishing screening, 800 emails | **75.4** acc · **252/400** caught · recall **0.63** | 61.9 · 99/400 · 0.25 |
| Dates and quantities (temporal_numeric) | **46.7** | 20.0 |
| Spam detection (enron-spam) | **1.000** | 0.920 |
| News topic (ag-news) | **0.955** | 0.806 |
| Typed scores (typed-decisions) | **0.438** | 0.395 |
| Calibration error, product relevance | **6.0** | 22.0 |
| Calibration error, overall (5,096 paired) | 9.5 | 9.3 |
| Median latency per decision | **267 ms** (T4 fp16) · 280 ms (Apple M, fp32) | 591 ms (hosted) |

2.5× the phishing caught at McNemar p = 3e-11, and 2.1× faster with nothing leaving the machine.

### Images

The hosted baseline takes no image input, so there is no like-for-like comparison. On document
pages it was given Apple Vision OCR of the same images and read that text, which makes its row a
two-model pipeline rather than a model that sees.

**RVL-CDIP-N**, 1,002 document pages, the 12 of 16 RVL-CDIP categories the set contains:

| system | reads | accuracy | ECE | median latency |
|---|---|---|---|---|
| Apple Vision OCR + Jev 1.13.0 | text | **0.896** | **0.045** | **1,049 ms** |
| **Vega 0.8B**, zero-shot | the image | 0.793 | 0.207 | 1,989 ms |
| DiT, best published on this set | the image | 0.786 | n/a | n/a |

DiT was trained on RVL-CDIP and is the strongest published result on this out-of-distribution set.
Vega has never seen the dataset and lands level with it. Not quite like-for-like: Vega chose among
the 12 categories present, while a classifier trained on RVL-CDIP chooses among all 16. The
pipeline latency includes the 649 ms Apple Vision takes per page, without which the API call alone
is 400 ms.

**MMMU-Pro**, 300 items over 30 strata, chance 0.120:

| system | reads | accuracy | ECE |
|---|---|---|---|
| Gemini 3.1 Pro, best reported Oct 2026 | the image | **0.839** | n/a |
| Jev 1.13.0 | text only, no image input | 0.337 | 0.160 |
| **Vega 0.8B** | the image | 0.240 | **0.052** |
| Vega 0.8B, same questions | image withheld | 0.157 | 0.062 |

An 800M model is nowhere near a frontier one on university-level reasoning. The result worth
keeping is the last two rows: withholding the image costs 8.3 points, so the vision path carries
real signal rather than the text answering alone. Calibration error is three times better than the
hosted model's on the same items.

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

## Download counts

The repository carries a `config.json` so the Hub can count downloads; it counts nothing for a
repository with no file it recognises. It holds no settings, nothing reads it, and the runtime
still reads `vega_config.json`. `fetch` asks for it from the repository root on every load,
including the 4B, which otherwise reads only from its own subfolder: a counter file that is never
requested counts nothing.

## Security

A checkpoint is data from a third party, so the inference code lives in this package and is reviewed
with it. `vegaml.load()` downloads **only** `vega_config.json`, `engine.safetensors` and the adapter
files; nothing fetched at runtime is imported or executed, and weights load through safetensors, never
pickle. Tests enforce that allow-list, and CI fails on any `pickle`, `torch.load`, `eval`, `exec`,
`subprocess` or `sys.path` insertion reaching the shipped package.

## Repository settings worth knowing

Two things this repository cannot enforce on a private repo without a paid plan, documented here so
nobody mistakes a gap for a gate:

* **Branch protection.** Both classic protection and rulesets return `403 Upgrade to GitHub Pro or
  make this repository public`, so the server enforces nothing: a force-push to `main`, a deletion,
  or a merge over a red check are all possible. The workflows still run on every push and pull
  request; only the enforcement is missing.

  The nearest available substitute is a local hook, `.githooks/pre-push`, which refuses a force-push
  or a deletion of `main`. Enable it in each clone with `git config core.hooksPath .githooks`. It
  stops the accident from a configured clone and **nothing else**: not another machine, not the web
  UI, not `--no-verify`. Treat it as a seatbelt, not a lock.
* **CodeQL.** Code scanning needs GitHub Advanced Security on a private repository
  (`422 Advanced security has not been purchased`), so the CodeQL job reports why it skipped instead
  of failing forever. Secret scanning, the dependency CVE audit and the supply-chain gate do run.

Making the repository public, or upgrading, turns both on with no change to the workflows.

## Licence

Apache 2.0.
