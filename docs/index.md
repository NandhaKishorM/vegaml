# vegaml documentation

One read of the state, typed answers out, each with a probability you can act on. The
[README](../README.md) covers the Python API and the measured numbers; these pages cover the
surfaces around it.

| Page | What it covers |
| --- | --- |
| [vega-U](vega-u.md) | The universal checkpoint: one engine, any reader, and its benchmarks |
| [Questions](questions.md) | The three question types and how to write their criteria |
| [CLI](cli.md) | `vegaml`: decisions from the shell, single or JSONL batches |
| [HTTP API](http-api.md) | `vegaml-serve`: the decide() call over HTTP |
| [MCP](mcp.md) | `vegaml-mcp-server`: decide() as a tool for agents |
| [Structured](structured.md) | A pydantic model as the question sheet and the answer |
| [Guard](guard.md) | A prompt-injection guard as a question sheet, and how to score one |
| [Cascade](router.md) | The 0.8B answers what it can settle, the 4B gets the rest |
| [Calibration](calibration.md) | Refit the probabilities on your own labels |
| [Escalation](escalation.md) | The review band, `vegaml-calibrate`, and the drift watch |
| [CPU serving](cpu.md) | Threads, readers and int8 on a box with no GPU |
| [Evals](evals.md) | `vegaml-eval`: measure the model on your labelled decisions |
| [LangChain](langchain.md) | Decisions as chain nodes: route, gate, annotate |
| [ONNX](onnx.md) | Export the engine, run it anywhere onnxruntime does |
| [MLX](mlx.md) | The engine on Apple silicon without torch |
| [Docker](docker.md) | The HTTP server as a container |

Every extra is additive and none changes the core: `pip install vegaml` stays enough for the
Python API, and `vegaml[serve]`, `[mcp]`, `[structured]`, `[onnx]` and `[mlx]` each add exactly
one of the surfaces above.
