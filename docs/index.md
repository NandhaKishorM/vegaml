# vegaml

Typed decisions from a frozen language model and a small physics engine. You give it a **state**
and a **question whose answer type is fixed in advance**; it returns a value your code can use
directly: a choice, a 0 to 1 score, or the probability that a statement is true. Each carries a
calibrated probability, a conformal answer set and an explicit abstain flag.

```bash
pip install vegaml
```

```python
import vegaml

v = vegaml.load()
out = v.decide("I was billed twice, refund me or I cancel.",
               {"team": {"type": "choice", "instructions": "Which team handles this?",
                         "criteria": {"billing": "payments and refunds", "technical": "bugs"}}})
out["answers"]["team"]["choice"]     # 'billing', with probs, a conformal set and abstain
```

Start with the [quick start](quickstart.md), or read [how it works](how-it-works.md).

## What it does

- **Text, images, audio and video**, with the same typed answers for each.
- **73,728 tokens of context.** Whole contracts, threads and logs in one read.
- **Abstain and unbound.** It says when it should not decide, and when the input is unlike
  anything it knows.
- **Physics, not text generation.** Each candidate answer is a well in an energy landscape; the
  input is a particle that rolls and settles in one. Deterministic, constant cost.
- **Any LLM as the reader.** One 14.3M engine over any frozen reader from 0.8B to 27B.
- **Keeps learning.** With [vegaml-learn](learn.md), it learns from every answer you confirm, or
  from a teacher it asks only when unsure. Free for noncommercial use: see
  [Licensing](licensing.md).

## Measured

| | Result |
| --- | --- |
| vega-U across readers, 50,676 held-out records | 0.8B 76.5%, 4B 79.1%, 27B 81.1%, ECE 0.010 to 0.021 |
| Continuous learning, 14 decision templates | 84.7% on day one, 91.6% after 40 confirmed answers, 96.0% after 120 |
| Learning from a stream, banking77, 4B reader | 87.5% after 4,928 rows, ECE 0.028 |
| With an LLM teacher, 3,000 records | vega alone 35% to 85%; share sent to the teacher 100% to 43% |
| Cost of learning | 300 answers in 34 s on 8 CPU threads, no GPU |

Where it loses: zero-shot on generic public suites a large hosted model leads, and on familiar
labels with enough examples a plain classifier on the same features matches the engine. The
engine's edge is cold start and new options. The full boards, methods and caveats are in the
[README](https://github.com/NandhaKishorM/vegaml#benchmarks) and on the [vega-U](vega-u.md) page.

## All pages

| Page | What it covers |
| --- | --- |
| [Quick start](quickstart.md) | Install, one decision, what comes back, fitting to your labels |
| [How it works](how-it-works.md) | The six steps from prompt to answer, abstain and unbound |
| [vega-U](vega-u.md) | The universal checkpoint: one engine, any reader, and its benchmarks |
| [Questions](questions.md) | The three question types and how to write their criteria |
| [CLI](cli.md) | `vegaml`: decisions from the shell, single or JSONL batches |
| [HTTP API](http-api.md) | `vegaml-serve`: the decide() call over HTTP |
| [MCP](mcp.md) | `vegaml-mcp-server`: decide() as a tool for agents |
| [Structured](structured.md) | A pydantic model as the question sheet and the answer |
| [Guard](guard.md) | A prompt-injection guard as a question sheet, and how to score one |
| [Cascade](router.md) | The 0.8B answers what it can settle, the 4B gets the rest |
| [Continuous learning](learn.md) | `vegaml_learn.Learner` (separate package, free for noncommercial use): answers from day one, learns from every answer you confirm |
| [Policies](policies.md) | Load a trained policy (Studio's signed .zip), verified; every answer routed auto or review |
| [Calibration](calibration.md) | Refit the probabilities on your own labels |
| [Escalation](escalation.md) | The review band, `vegaml-calibrate`, and the drift watch |
| [CPU serving](cpu.md) | Threads, readers and int8 on a box with no GPU |
| [Evals](evals.md) | `vegaml-eval`: measure the model on your labelled decisions |
| [LangChain](langchain.md) | Decisions as chain nodes: route, gate, annotate |
| [ONNX](onnx.md) | Export the engine, run it anywhere onnxruntime does |
| [MLX](mlx.md) | The engine on Apple silicon without torch |
| [Licensing](licensing.md) | Two packages, two licences, and who needs a commercial one |
| [API reference](reference/index.md) | Every argument, generated from the docstrings |
| [Where this is going](vision.md) | The lab's path from laya to vega to what comes next |
| [Docker](docker.md) | The HTTP server as a container |

Every extra is additive and none changes the core: `pip install vegaml` stays enough for the
Python API, and `vegaml[serve]`, `[mcp]`, `[structured]`, `[onnx]`, `[mlx]`, `[audio]` and `[video]`
each add exactly one surface: `[audio]` brings soundfile for `decide_audio` on audio files, and
`[video]` brings PyAV for `decide_video`.

## Links

- Source: [github.com/NandhaKishorM/vegaml](https://github.com/NandhaKishorM/vegaml)
- Hosted vega Proxy: [platform.embedding.in](https://platform.embedding.in)
- The lab: [embedding.in](https://embedding.in) and its [vision](https://embedding.in/vision)
- The paper: [doi:10.20944/preprints202611.0781.v1](https://doi.org/10.20944/preprints202611.0781.v1)
