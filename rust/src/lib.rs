//! Rust inference port of [VegaML](https://github.com/NandhaKishorM/vegaml): a frozen causal LM
//! reads the situation once per question, and a small latent-physics engine turns the pooled span
//! features into a calibrated answer distribution with a conformal prediction set.
//!
//! The engine (`engine`, `weights`, `prompt`) is pure Rust with no runtime dependencies beyond
//! `serde_json` and `libm`. The `llama` feature adds `reader`, which runs a GGUF reader through
//! llama.cpp and taps its decoder-layer outputs.

pub mod engine;
pub mod prompt;
#[cfg(feature = "llama")]
pub mod reader;
pub mod weights;

pub use engine::{EngineOutput, Features, VegaAnswer, VegaConfig, VegaEngine};
pub use prompt::{Spans, VegaQuestion, build_prompt};

/// Answer shape of a question (VegaML's `QTYPES`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QType {
    /// One of several labelled options.
    Choice,
    /// An ordinal level; options sit on a rail so neighbours are physical neighbours.
    Score,
    /// Yes / no; option 1 is "yes".
    Boolean,
}

impl std::str::FromStr for QType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "choice" => Ok(Self::Choice),
            "score" => Ok(Self::Score),
            "boolean" => Ok(Self::Boolean),
            other => anyhow::bail!("unknown question type {other:?} (expected choice, score or boolean)"),
        }
    }
}

/// Engine + reader.
#[cfg(feature = "llama")]
pub struct Vega {
    pub engine: VegaEngine,
    pub reader: reader::VegaReader,
}

#[cfg(feature = "llama")]
impl Vega {
    /// Load a checkpoint directory and its reader GGUF. `n_ctx` bounds the prompt length.
    pub fn load(backend: &llama_cpp_2::llama_backend::LlamaBackend, checkpoint: &std::path::Path, reader_gguf: &std::path::Path, n_ctx: usize, threads: i32) -> anyhow::Result<Self> {
        let engine = VegaEngine::load(checkpoint)?;
        let reader = reader::VegaReader::load(backend, reader_gguf, &engine.cfg, n_ctx, threads)?;
        Ok(Self { engine, reader })
    }

    /// Prompt tokens and spans for one question (the context length bounds the state).
    pub fn prompt(&self, state: &str, q: &VegaQuestion) -> (Vec<i32>, Spans) {
        let cfg = &self.engine.cfg;
        let max_len = cfg.max_len.min(self.reader.n_ctx());
        build_prompt(&|s| self.reader.tokenize(s), self.reader.bos(), state, q, max_len, cfg.opt_max_tokens, cfg.query_first_max_state)
    }

    /// One reading of `state` for `q`; returns the raw engine output and the calibrated answer.
    pub fn decide(&mut self, state: &str, q: &VegaQuestion) -> anyhow::Result<(EngineOutput, VegaAnswer)> {
        let (ids, spans) = self.prompt(state, q);
        let f = self.reader.pooled(&ids, &spans, &self.engine.cfg)?;
        let out = self.engine.forward(&f, q.kind);
        let ans = self.engine.calibrate(&out, q.kind);
        Ok((out, ans))
    }
}
