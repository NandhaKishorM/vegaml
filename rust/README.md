# vegaml (Rust)

This is an inference-only port of VegaML to Rust. You can run a vega-U checkpoint from a native
binary with no Python: the reader runs on llama.cpp from a GGUF file, and the engine is plain
Rust.

- **Engine** (`engine`, `weights`, `prompt`):
  - Mirrors `_common.VegaEngine` behind `_universal.Port`: scalar layer mixes, world latent with
    the chunk memory reader and the foresight step, wells with the score rail, the damped
    integrator with the selective-SSM thermostat, and halting.
  - Includes template-0 prompt building with query-first reading.
  - Includes thermal calibration, the conformal set and the abstain / unbound flags.
  - Loads `vega_config.json`, `engine.safetensors`, `port.safetensors` and `stats.npz` without
    executing anything. Only F32 data is accepted, and every offset is bounds-checked.
  - Needs only `serde_json` and `libm`.
- **Reader** (`reader`, default feature `llama`):
  - Runs any GGUF build of the checkpoint's reader through llama.cpp.
  - Taps `l_out-<i>`, the residual stream after decoder block *i*, through the scheduler eval
    callback. This is what a forward hook on `layers[i]` sees.
  - Pools state, question, final-token, option and chunk features as `Perception.pooled` does.

**Not ported:** training, `fit`/adapters, `views` > 1, the long-record prefix cache (states of
4,096 tokens or more), and vision.

## Usage

```rust
use llama_cpp_2::llama_backend::LlamaBackend;
use vegaml::{Vega, VegaQuestion};

let backend = LlamaBackend::init()?;
let mut vega = Vega::load(&backend, "vega-U".as_ref(), "Qwen3.5-0.8B-Q8_0.gguf".as_ref(), 4096, 8)?;
let q = VegaQuestion::boolean("Is this a bill or invoice that I must pay?", None, None);
let (_, answer) = vega.decide("From: billing@host.test\nSubject: Invoice\n\nTotal due: €89.00", &q)?;
println!("p(yes) = {:.3}, abstain = {}", answer.probs[1], answer.abstain);
```

- **Checkpoint:** the root of `nandakishorm/vega-U` (the Qwen3.5-0.8B reader).
- **Reader GGUF:** for example `unsloth/Qwen3.5-0.8B-GGUF`, file `Qwen3.5-0.8B-Q8_0.gguf`.

## Build and test

You need Rust 1.88 or newer. The `llama` feature also needs a C/C++ toolchain, and bindgen needs
`libclang` (set `LIBCLANG_PATH`).

```bash
cd rust
cargo test --release                        # engine + reader
cargo test --release --no-default-features  # pure-Rust engine only
```

## Golden check against the Python package

```bash
python rust/tests/reference/golden_reference.py . <vega-U dir> rust/tests/reference/cases.json golden.json
cd rust && cargo run --release --example golden -- ../golden.json <vega-U dir> Qwen3.5-0.8B-Q8_0.gguf
```

The reference runs this repository's `vegaml` on CPU in fp32 with the Hugging Face reader. The
Rust side checks three things, in increasing scope:

1. Prompt tokens and spans.
2. The engine on the reference's own pooled features. This is asserted below 1e-4.
3. End to end with the GGUF reader. Drift is reported, not asserted.

Measured on `cases.json` (20 cases: 4 synthetic emails × boolean, score and choice questions,
including the memory-reader path). The setup was vega-U root at `4d20502`, with `vegaml` at
`0eb2c28`, on Windows 11 x64:

| Check | Result |
|---|---|
| Tokenization and spans (llama.cpp vocab against the HF tokenizer) | identical in 20/20 cases |
| Engine on reference features | max relative logit error 2.4e-6, max probability error 3.7e-7 |
| `Qwen3.5-0.8B-Q8_0.gguf` reader end to end | top-1 identical in 20/20; max probability drift 0.017; item-feature cosine ≥ 0.9997 |
