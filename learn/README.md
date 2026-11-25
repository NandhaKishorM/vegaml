# vegaml-learn

Continuous learning for [vega](https://github.com/NandhaKishorM/vegaml): a `Learner` that answers from day one and learns from every answer you confirm, an `Escalator` that sends only what vega is unsure of to a teacher and learns from the reply, and the vega Proxy that sits in front of an LLM API and learns your decisions from it.

## Licence

**Free for noncommercial use.** vegaml-learn is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) (see [LICENSE](https://github.com/NandhaKishorM/vegaml/blob/main/learn/LICENSE)). Personal projects, research, study, and use by charities, schools, public research bodies and government are all covered.

**Commercial use needs a licence** from Embedding Space: [embedding.in](https://embedding.in). That includes using it inside a product, a paid service or a company's own operations.

vegaml itself, the decision engine this builds on, stays Apache 2.0. Versions of the learner that shipped inside vegaml 0.17.0 to 0.19.0 keep the Apache 2.0 licence they were released under.

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

Full guide: [docs/learn.md](https://github.com/NandhaKishorM/vegaml/blob/main/docs/learn.md).
