# vegaml-learn

Continuous learning for [vega](https://github.com/NandhaKishorM/vegaml), **free for noncommercial use only** (see [Licence](#licence-noncommercial-only)): a `Learner` that answers from day one and learns from every answer you confirm, an `Escalator` that sends only what vega is unsure of to a teacher and learns from the reply, and the vega Proxy that sits in front of an LLM API and learns your decisions from it.

> **Cite this work.** M, N.; M, A. From Laya to Vega: Open Non-Autoregressive Decision Models with Calibrated Probabilities. *Preprints* **2026**, 2026110781. [https://doi.org/10.20944/preprints202611.0781.v1](https://doi.org/10.20944/preprints202611.0781.v1)

<details>
<summary>BibTeX</summary>

```bibtex
@article{m2026laya_vega,
  title   = {From Laya to Vega: Open Non-Autoregressive Decision Models with Calibrated Probabilities},
  author  = {M, N. and M, A.},
  journal = {Preprints},
  year    = {2026},
  pages   = {2026110781},
  doi     = {10.20944/preprints202611.0781.v1},
  url     = {https://doi.org/10.20944/preprints202611.0781.v1}
}
```

</details>

## Licence: noncommercial only

> **Read this before you build on it.** vegaml-learn is **not** open source in the Apache or MIT sense. It is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) ([LICENSE](https://github.com/NandhaKishorM/vegaml/blob/main/learn/LICENSE)).

**Free:** personal projects, research, study and experiments, and use by charities, schools, public research bodies and government.

**Needs a paid commercial licence:** any use by or for a business. That includes running it inside a product or a paid service, using it in a company's own operations (support routing, sales, risk, back office), and offering it to customers as a hosted service. It covers everything in this package: `Learner`, landscape learning, `Escalator` and the teachers, the REST server and the vega Proxy, and `laya.Learner`, which loads this package.

Terms, and a quote for your company: [embedding.in/licence](https://embedding.in/licence). For a quote, write to [nandakishor@embedding.in](mailto:nandakishor@embedding.in).

vegaml itself, the decision engine this builds on, stays Apache 2.0. The learner that shipped inside vegaml 0.17.0 to 0.19.0 keeps the Apache 2.0 licence it was released under.

## Landscape learning

New in 0.20, opt-in with `VEGA_LEARN_HEAD=landscape`. Each answer gets a well in vega's energy landscape, placed from its confirmed examples; the answer is the well the input settles in, with a temperature fitted on held-out examples. Nothing is iterated and no fit grows with the examples stored.

Measured against the default head on the same features: level on questions with a few answers (12 public tasks and model routing, within about a point from 100 examples on) and on ten sales questions with the vega 4B; behind by 2 to 9 points on questions with 60 to 100 answers until it has several hundred examples; faster to fit only past about a thousand examples (0.6 s against 4.8 s at 3,000). Use it for long-running learners on questions with few answers. What a learner has stored works with both, so switching back loses nothing.

## Install

```bash
pip install vegaml-learn             # Learner, Escalator, vllm_teacher
pip install "vegaml-learn[serve]"    # REST server with JWT: python -m vegaml_learn serve
pip install "vegaml-learn[proxy]"    # vega Proxy:            python -m vegaml_learn proxy
pip install "vegaml-learn[claude]"   # claude_teacher
```

## Use

```python
from vegaml_learn import Learner, Escalator, vllm_teacher

learner = Learner(questions, examples=rows, workdir="./vega-state")   # rows: [(text, {question: answer})]
learner.decide("I was charged twice this month")
learner.feedback("I was charged twice this month", {"team": "billing"})   # learns every 50, in the background

desk = Escalator(learner, teacher=vllm_teacher())   # your own model; or claude_teacher(), or any function
desk.decide("I was charged twice")["answers"]["team"]["source"]   # "teacher" until vega has learned
```

Code written for vegaml 0.19 keeps working: once vegaml-learn is installed, `from vegaml import Learner`, `from vegaml.learn import Escalator` and `python -m vegaml.learn proxy` all resolve to this package.

Full guide: [nandhakishorm.github.io/vegaml/learn](https://nandhakishorm.github.io/vegaml/learn/), also at [embedding.in/docs/continuous-learning](https://embedding.in/docs/continuous-learning). API reference: [Learner and Escalator](https://nandhakishorm.github.io/vegaml/reference/learner/).
