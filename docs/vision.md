# Where this is going

vegaml is built by [Embedding Space](https://embedding.in), a foundational AI lab. The aim is AI
that knows when it does not know, keeps learning from the people it works for, and makes
decisions that are good for them. The full statement is at
[embedding.in/vision](https://embedding.in/vision); this page is the part that concerns this
repository, with what is shipped and what is still research marked as such.

## What we believe

- **A model must be able to say no.** Today's language models always answer, sure or not. A
  model you can trust tells you when it should not decide (`abstain`) and when the input is
  unlike anything it knows (`unbound`).
- **Learning should never stop.** A model fixed on the day it ships repeats its mistakes. vega
  learns from every corrected answer, from a person or from a larger model.
- **AI has to be aligned with people.** A decision is only good if it is good for the people it
  affects. The models should learn how people actually decide, defer to them when unsure, and
  stay correctable.

## The path

| | Step | Status | Where |
| --- | --- | --- | --- |
| 1 | **laya**, the System 1 decision model: one pass, no text generated | Shipped, September 2026 | [laya](https://github.com/NandhaKishorM/laya) |
| 2 | **vega**, a decision engine built on physics: every answer is a valley, the input is a particle that settles | Shipped, October 2026 | [How it works](how-it-works.md), [vega-U](vega-u.md) |
| 3 | **Continuous learning**, from a person or an LLM teacher | Shipped, November 2026 | [Continuous learning](learn.md) |
| 4 | **Direct weight-space editing**: a decision policy written straight into a model's weights, with no training run | In use | [What we measured](https://embedding.in/blog/a-decision-skill-written-into-the-weights) |
| 5 | **Abstain and unbound for every LLM**, so a chatbot or an agent can stop and hand over instead of guessing | Research | |
| 6 | **Systems that improve themselves**, from their own checked answers, with people setting the goal and keeping the controls | Research, early | |
| 7 | **Abstract reasoning beyond the transformer**: models built on energy and physics, where reasoning means settling into a consistent state | Research | |

On step 6: in our first runs the decision engine got better and the gains did not carry over to
wider tests, so we make no claim yet.

## Why decisions first

A decision can be checked. It was right or it was wrong, and somebody finds out. That makes it
the one place where an AI system can be measured honestly, corrected, and trusted a little more
each time it earns it. Everything in this repository is built so that the numbers can be rerun:
see [Evals](evals.md) and the [benchmarks in the README](https://github.com/NandhaKishorM/vegaml#benchmarks).
