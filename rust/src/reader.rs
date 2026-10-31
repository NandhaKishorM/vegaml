//! llama-cpp-2 does not expose the scheduler's eval callback, so this reader talks to
//! `llama-cpp-sys-2` directly: the callback asks for the `l_out-<i>` tensors (the residual
//! stream after decoder block i, the same thing a PyTorch forward hook on `layers[i]` sees) and
//! copies them out as each micro-batch is computed.

use std::ffi::{CStr, CString, c_void};
use std::path::Path;
use std::ptr;

use anyhow::{Result, anyhow, bail, ensure};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_sys_2 as sys;

use crate::engine::{Features, VegaConfig};
use crate::prompt::Spans;

const UBATCH: usize = 512;

/// Tensors the callback collects: `bufs[i]` grows by `d` floats per token of layer `layers[i]`.
struct Capture {
    names: Vec<CString>,
    bufs: Vec<Vec<f32>>,
    d: usize,
    error: Option<String>,
}

unsafe extern "C" fn eval_cb(t: *mut sys::ggml_tensor, ask: bool, ud: *mut c_void) -> bool {
    // SAFETY: llama.cpp passes back the `Capture` we registered; it outlives the context's decodes.
    let cap = unsafe { &mut *(ud as *mut Capture) };
    let name = unsafe { CStr::from_ptr(sys::ggml_get_name(t)) };
    let Some(slot) = cap.names.iter().position(|n| n.as_c_str() == name) else {
        return !ask; // not ours: do not ask for it, but keep computing if called anyway
    };
    if ask {
        return true;
    }
    let tensor = unsafe { &*t };
    if tensor.type_ != sys::GGML_TYPE_F32 || tensor.ne[0] as usize != cap.d || !unsafe { sys::ggml_is_contiguous(t) } {
        cap.error = Some(format!("{name:?}: unexpected layout (type {}, ne {:?})", tensor.type_, tensor.ne));
        return true;
    }
    let n = (tensor.ne[0] * tensor.ne[1]) as usize;
    let buf = &mut cap.bufs[slot];
    let start = buf.len();
    buf.resize(start + n, 0.0);
    unsafe { sys::ggml_backend_tensor_get(t, buf[start..].as_mut_ptr() as *mut c_void, 0, n * 4) };
    true
}

pub struct VegaReader {
    model: *mut sys::llama_model,
    ctx: *mut sys::llama_context,
    vocab: *const sys::llama_vocab,
    cap: Box<Capture>,
    layers: Vec<usize>,
    n_ctx: usize,
    pub d: usize,
}

// SAFETY: the reader is owned and used by one worker thread at a time; llama.cpp contexts are not shared.
unsafe impl Send for VegaReader {}

impl Drop for VegaReader {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                sys::llama_free(self.ctx);
            }
            if !self.model.is_null() {
                sys::llama_model_free(self.model);
            }
        }
    }
}

impl VegaReader {
    /// Load the GGUF reader for `cfg`'s tapped layers. `n_ctx` bounds the prompt length.
    pub fn load(_backend: &LlamaBackend, path: &Path, cfg: &VegaConfig, n_ctx: usize, threads: i32) -> Result<Self> {
        let mut layers: Vec<usize> = cfg.feature_layers.iter().chain(&cfg.option_layers).copied().collect();
        layers.sort_unstable();
        layers.dedup();
        let cpath = CString::new(path.to_str().ok_or_else(|| anyhow!("model path is not UTF-8"))?)?;
        let mut me = Self { model: ptr::null_mut(), ctx: ptr::null_mut(), vocab: ptr::null(), cap: Box::new(Capture { names: vec![], bufs: vec![], d: 0, error: None }), layers, n_ctx, d: 0 };
        unsafe {
            let mp = sys::llama_model_default_params();
            me.model = sys::llama_model_load_from_file(cpath.as_ptr(), mp);
            if me.model.is_null() {
                bail!("could not load {}", path.display());
            }
            me.d = sys::llama_model_n_embd(me.model) as usize;
            let n_layer = sys::llama_model_n_layer(me.model) as usize;
            ensure!(me.d == cfg.d_model, "reader width {} differs from the checkpoint's {}", me.d, cfg.d_model);
            ensure!(me.layers.iter().all(|&l| l < n_layer), "reader has {n_layer} layers, checkpoint taps {:?}", me.layers);
            me.vocab = sys::llama_model_get_vocab(me.model);
            me.cap.d = me.d;
            me.cap.names = me.layers.iter().map(|l| CString::new(format!("l_out-{l}")).expect("no NUL")).collect();
            me.cap.bufs = vec![Vec::new(); me.layers.len()];
            let mut cp = sys::llama_context_default_params();
            cp.n_ctx = n_ctx as u32;
            cp.n_batch = UBATCH as u32;
            cp.n_ubatch = UBATCH as u32;
            cp.n_seq_max = 1;
            cp.n_threads = threads;
            cp.n_threads_batch = threads;
            cp.no_perf = true;
            cp.cb_eval = Some(eval_cb);
            cp.cb_eval_user_data = &mut *me.cap as *mut Capture as *mut c_void;
            me.ctx = sys::llama_init_from_model(me.model, cp);
            if me.ctx.is_null() {
                bail!("could not create a reader context");
            }
        }
        Ok(me)
    }

    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    /// The reader's BOS token if its tokenizer prepends one (Qwen's does not).
    pub fn bos(&self) -> Option<i32> {
        unsafe {
            let add = sys::llama_vocab_get_add_bos(self.vocab);
            let b = sys::llama_vocab_bos(self.vocab);
            (add && b >= 0).then_some(b)
        }
    }

    /// Plain-text tokenization: no special tokens added or parsed.
    pub fn tokenize(&self, text: &str) -> Vec<i32> {
        let mut out = vec![0i32; text.len() + 8];
        let mut n = unsafe { sys::llama_tokenize(self.vocab, text.as_ptr() as *const _, text.len() as i32, out.as_mut_ptr(), out.len() as i32, false, false) };
        if n < 0 {
            out.resize((-n) as usize, 0);
            n = unsafe { sys::llama_tokenize(self.vocab, text.as_ptr() as *const _, text.len() as i32, out.as_mut_ptr(), out.len() as i32, false, false) };
        }
        out.truncate(n.max(0) as usize);
        out
    }

    /// One forward pass over `ids`; returns the tapped hidden states, `[layer][token * d]`.
    fn run(&mut self, ids: &[i32]) -> Result<Vec<Vec<f32>>> {
        ensure!(!ids.is_empty() && ids.len() <= self.n_ctx, "prompt of {} tokens does not fit {}", ids.len(), self.n_ctx);
        self.cap.bufs.iter_mut().for_each(Vec::clear);
        self.cap.error = None;
        unsafe {
            sys::llama_memory_clear(sys::llama_get_memory(self.ctx), true);
            for (c, chunk) in ids.chunks(UBATCH).enumerate() {
                let mut batch = sys::llama_batch_init(chunk.len() as i32, 0, 1);
                for (i, &t) in chunk.iter().enumerate() {
                    *batch.token.add(i) = t;
                    *batch.pos.add(i) = (c * UBATCH + i) as i32;
                    *batch.n_seq_id.add(i) = 1;
                    *(*batch.seq_id.add(i)) = 0;
                    // One output per micro-batch keeps the graph identical to a normal decode.
                    *batch.logits.add(i) = (i + 1 == chunk.len()) as i8;
                }
                batch.n_tokens = chunk.len() as i32;
                let rc = sys::llama_decode(self.ctx, batch);
                sys::llama_batch_free(batch);
                if rc != 0 {
                    bail!("llama_decode failed ({rc})");
                }
            }
        }
        if let Some(e) = self.cap.error.take() {
            bail!(e);
        }
        let bufs = std::mem::take(&mut self.cap.bufs);
        self.cap.bufs = vec![Vec::new(); self.layers.len()];
        for b in &bufs {
            ensure!(b.len() == ids.len() * self.d, "captured {} values for {} tokens", b.len(), ids.len());
        }
        Ok(bufs)
    }

    /// `Perception.pooled` for one prompt: span means at the feature/option layers, the final token, and state chunks.
    pub fn pooled(&mut self, ids: &[i32], spans: &Spans, cfg: &VegaConfig) -> Result<Features> {
        let h = self.run(ids)?;
        let d = self.d;
        let at = |layer: usize| &h[self.layers.iter().position(|&l| l == layer).expect("tapped layer")];
        let mean = |buf: &[f32], (a, b): (usize, usize)| -> Vec<f32> {
            let mut acc = vec![0f64; d];
            for t in a..b {
                acc.iter_mut().zip(&buf[t * d..(t + 1) * d]).for_each(|(s, x)| *s += *x as f64);
            }
            let n = (b - a).max(1) as f64;
            acc.into_iter().map(|s| (s / n) as f32).collect()
        };
        let last = ids.len() - 1;
        let mut items = Vec::with_capacity(cfg.feature_layers.len() * 3 * d);
        for &l in &cfg.feature_layers {
            let buf = at(l);
            items.extend(mean(buf, spans.state));
            items.extend(mean(buf, spans.question));
            items.extend_from_slice(&buf[last * d..(last + 1) * d]);
        }
        let options = spans.options.iter().map(|&sp| cfg.option_layers.iter().flat_map(|&l| mean(at(l), sp)).collect()).collect();
        let mut chunks = Vec::new();
        let (s0, s1) = spans.state;
        let n = s1 - s0;
        if cfg.memory_chunks > 0 && n >= cfg.memory_min_tokens.max(2) {
            let k = (n / cfg.memory_chunk_tokens.max(1)).min(cfg.memory_chunks).max(2);
            let buf = at(*cfg.feature_layers.last().expect("feature layers"));
            // Token j of the state goes to chunk j * k / n (equal pieces, as torch.arange(n) * k // n).
            for c in 0..k {
                let a = s0 + (c * n).div_ceil(k);
                let b = s0 + ((c + 1) * n).div_ceil(k);
                chunks.push(mean(buf, (a, b)));
            }
        }
        Ok(Features { items, options, chunks })
    }
}
