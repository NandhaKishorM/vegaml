//! Inference-only mirror of `vegaml._common.VegaEngine` behind a vega-U `Port`, evaluated for one
//! question at a time in f32 (as PyTorch does on CPU). Dropout is identity and adapters are off,
//! so the base checkpoint's numbers are reproduced exactly up to summation order.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use crate::QType;
use serde::Deserialize;

use crate::weights::{Tensor, Tensors, read_npz, read_safetensors};

/// VegaML's question-type index (`QTYPES` in the Python package).
pub fn qtype_index(kind: QType) -> usize {
    match kind {
        QType::Choice => 0,
        QType::Score => 1,
        QType::Boolean => 2,
    }
}

/// The type name the prompt template prints (`Measurement (boolean): …`).
pub fn qtype_name(kind: QType) -> &'static str {
    match kind {
        QType::Choice => "choice",
        QType::Score => "score",
        QType::Boolean => "boolean",
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Calibration {
    pub theta: Vec<Vec<f64>>,
    #[serde(default)]
    pub conformal: HashMap<String, HashMap<String, f64>>,
    #[serde(default)]
    pub abstain_threshold: HashMap<String, f64>,
    #[serde(default = "default_risk")]
    pub default_risk: String,
    #[serde(default = "inf")]
    pub unbound_threshold: f64,
}

fn default_risk() -> String {
    "0.1".into()
}

fn inf() -> f64 {
    f64::INFINITY
}

/// The subset of `vega_config.json` inference needs.
#[derive(Debug, Clone, Deserialize)]
pub struct VegaConfig {
    pub latent_dim: usize,
    pub world_dim: usize,
    pub n_steps: usize,
    pub dt_max: f32,
    pub settle_eps: f32,
    #[serde(default = "force_max")]
    pub force_max: f32,
    #[serde(default)]
    pub memory_chunks: usize,
    #[serde(default = "memory_min_tokens")]
    pub memory_min_tokens: usize,
    #[serde(default = "memory_chunk_tokens")]
    pub memory_chunk_tokens: usize,
    #[serde(default)]
    pub query_first_max_state: usize,
    pub opt_max_tokens: usize,
    pub max_len: usize,
    pub backbone_name: String,
    pub d_model: usize,
    pub d_shared: Option<usize>,
    pub port: Option<String>,
    pub raw_stats: Option<String>,
    pub feature_layers: Vec<usize>,
    pub option_layers: Vec<usize>,
    #[serde(default)]
    pub template: usize,
    pub calibration: Option<Calibration>,
}

fn force_max() -> f32 {
    20.0
}
fn memory_min_tokens() -> usize {
    2048
}
fn memory_chunk_tokens() -> usize {
    128
}

/// Pooled perception features for one prompt, as `Perception.pooled` returns them (raw, unstandardized).
#[derive(Debug, Clone, Default)]
pub struct Features {
    /// `[Lf][3][d]` flattened: per feature layer the state-span mean, question-span mean and final token.
    pub items: Vec<f32>,
    /// One `[Lo][d]` vector per option, in prompt order.
    pub options: Vec<Vec<f32>>,
    /// Valid state-chunk means at the deepest feature layer (at most `memory_chunks`), in order.
    pub chunks: Vec<Vec<f32>>,
}

/// Raw engine output for one question.
#[derive(Debug, Clone)]
pub struct EngineOutput {
    /// Negative well energies (`-E_k`), one per option.
    pub logits: Vec<f32>,
    pub kinetic: f32,
    pub nearest: f32,
    pub steps_used: usize,
}

/// A calibrated answer, with VegaML's conformal set and abstain / out-of-distribution flags.
#[derive(Debug, Clone)]
pub struct VegaAnswer {
    pub probs: Vec<f64>,
    pub best: usize,
    pub confidence: f64,
    /// Option indices in the split-conformal prediction set at alpha = 0.1 (always holds `best`).
    pub prediction_set: Vec<usize>,
    pub abstain: bool,
    pub unbound: bool,
    pub settle_steps: usize,
}

// ------------------------------------------------------------------------------------ layers

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    for (x, y) in ca.iter().zip(cb) {
        for ((s, x), y) in acc.iter_mut().zip(x).zip(y) {
            *s += x * y;
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

fn take(t: &mut Tensors, name: &str) -> Result<Tensor> {
    t.remove(name).ok_or_else(|| anyhow!("missing tensor {name}"))
}

struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    n_in: usize,
}

impl Linear {
    fn load(t: &mut Tensors, prefix: &str) -> Result<Self> {
        let w = take(t, &format!("{prefix}.weight"))?;
        let b = take(t, &format!("{prefix}.bias"))?;
        ensure!(w.shape.len() == 2 && w.shape[0] == b.data.len(), "{prefix}: shape {:?}", w.shape);
        Ok(Self { n_in: w.shape[1], w: w.data, b: b.data })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.n_in);
        self.b.iter().enumerate().map(|(o, b)| b + dot(&self.w[o * self.n_in..(o + 1) * self.n_in], x)).collect()
    }
}

struct LayerNorm {
    g: Vec<f32>,
    b: Vec<f32>,
}

impl LayerNorm {
    fn load(t: &mut Tensors, prefix: &str) -> Result<Self> {
        Ok(Self { g: take(t, &format!("{prefix}.weight"))?.data, b: take(t, &format!("{prefix}.bias"))?.data })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + 1e-5).sqrt();
        x.iter().zip(&self.g).zip(&self.b).map(|((v, g), b)| (v - mean) * inv * g + b).collect()
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `F.softplus` with PyTorch's default threshold of 20.
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// `nn.Sequential(LayerNorm, Linear, GELU, Dropout, Linear)`.
struct Mlp {
    ln: LayerNorm,
    l1: Linear,
    l2: Linear,
}

impl Mlp {
    fn load(t: &mut Tensors, prefix: &str) -> Result<Self> {
        Ok(Self {
            ln: LayerNorm::load(t, &format!("{prefix}.net.0"))?,
            l1: Linear::load(t, &format!("{prefix}.net.1"))?,
            l2: Linear::load(t, &format!("{prefix}.net.4"))?,
        })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let h: Vec<f32> = self.l1.forward(&self.ln.forward(x)).into_iter().map(gelu).collect();
        self.l2.forward(&h)
    }
}

fn add(a: &mut [f32], b: &[f32]) {
    a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn softmax_f32(v: &[f32]) -> Vec<f32> {
    let m = v.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = v.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

// ------------------------------------------------------------------------------------ engine

/// Standardized inputs: item slots `[Lf][3][d]`, options `[K][Lo][d]`, state chunks `[C][d]`.
type Standardized = (Vec<Vec<Vec<f32>>>, Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>);

struct Memory {
    q: Mlp,
    kv_ln: LayerNorm,
    kv: Linear,
    pos: Tensor,
    out: Linear,
    heads: usize,
}

/// vega-U per-reader port with the reader's raw extraction statistics.
struct Port {
    mu: Tensor,
    sd: Tensor,
    omu: Vec<f32>,
    osd: Vec<f32>,
    lin: Vec<Linear>,
}

pub struct VegaEngine {
    pub cfg: VegaConfig,
    port: Option<Port>,
    feat_mu: Tensor,
    feat_sd: Tensor,
    opt_mu: Tensor,
    opt_sd: Tensor,
    mix: Tensor,
    opt_mix: Vec<f32>,
    world: Mlp,
    foresight_ln: LayerNorm,
    foresight_inp: Linear,
    foresight_out: Linear,
    probe: Mlp,
    impulse: Mlp,
    qtype_emb: Tensor,
    to_z0: Mlp,
    to_p0: Mlp,
    well: Mlp,
    rail_dir: Mlp,
    log_kappa: f32,
    ssm_in: Linear,
    a_log: Vec<f32>,
    ssm_out: Linear,
    force_scale: f32,
    memory: Option<Memory>,
    /// Feature width the engine itself runs at (`d_shared` behind a port, else `d_model`).
    d: usize,
}

impl VegaEngine {
    /// Load a checkpoint directory (`vega_config.json`, `engine.safetensors`, and for vega-U the port and stats).
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: VegaConfig = serde_json::from_slice(&std::fs::read(dir.join("vega_config.json")).context("read vega_config.json")?)?;
        let mut t = read_safetensors(&dir.join("engine.safetensors"))?;
        let port = match &cfg.port {
            Some(p) => {
                let mut pt = read_safetensors(&dir.join(p))?;
                let mut st = read_npz(&dir.join(cfg.raw_stats.as_deref().unwrap_or("stats.npz")))?;
                let lin = (0..cfg.feature_layers.len()).map(|i| Linear::load(&mut pt, &format!("lin.{i}"))).collect::<Result<Vec<_>>>()?;
                Some(Port {
                    mu: take(&mut st, "mu")?,
                    sd: take(&mut st, "sd")?,
                    omu: take(&mut st, "omu")?.data,
                    osd: take(&mut st, "osd")?.data,
                    lin,
                })
            }
            None => None,
        };
        let memory = if cfg.memory_chunks > 0 {
            Some(Memory {
                q: Mlp::load(&mut t, "memory.q")?,
                kv_ln: LayerNorm::load(&mut t, "memory.kv.0")?,
                kv: Linear::load(&mut t, "memory.kv.1")?,
                pos: take(&mut t, "memory.pos")?,
                out: Linear::load(&mut t, "memory.out")?,
                heads: 4,
            })
        } else {
            None
        };
        let scalar = |t: &mut Tensors, n: &str| -> Result<f32> { Ok(take(t, n)?.data[0]) };
        let e = Self {
            port,
            feat_mu: take(&mut t, "feat_mu")?,
            feat_sd: take(&mut t, "feat_sd")?,
            opt_mu: take(&mut t, "opt_mu")?,
            opt_sd: take(&mut t, "opt_sd")?,
            mix: take(&mut t, "mix")?,
            opt_mix: take(&mut t, "opt_mix")?.data,
            world: Mlp::load(&mut t, "world")?,
            foresight_ln: LayerNorm::load(&mut t, "foresight.norm")?,
            foresight_inp: Linear::load(&mut t, "foresight.inp")?,
            foresight_out: Linear::load(&mut t, "foresight.out")?,
            probe: Mlp::load(&mut t, "probe")?,
            impulse: Mlp::load(&mut t, "impulse")?,
            qtype_emb: take(&mut t, "qtype_emb.weight")?,
            to_z0: Mlp::load(&mut t, "to_z0")?,
            to_p0: Mlp::load(&mut t, "to_p0")?,
            well: Mlp::load(&mut t, "well")?,
            rail_dir: Mlp::load(&mut t, "rail_dir")?,
            log_kappa: scalar(&mut t, "log_kappa")?,
            ssm_in: Linear::load(&mut t, "ssm_in")?,
            a_log: take(&mut t, "A_log")?.data,
            ssm_out: Linear::load(&mut t, "ssm_out")?,
            force_scale: scalar(&mut t, "force_scale")?,
            memory,
            d: cfg.d_shared.filter(|_| cfg.port.is_some()).unwrap_or(cfg.d_model),
            cfg,
        };
        ensure!(e.qtype_emb.shape == [3, e.cfg.latent_dim], "qtype_emb shape {:?}", e.qtype_emb.shape);
        Ok(e)
    }

    fn qemb(&self, qt: usize) -> &[f32] {
        let d = self.cfg.latent_dim;
        &self.qtype_emb.data[qt * d..(qt + 1) * d]
    }

    /// Standardize (and, behind a vega-U port, map into the shared space) raw perception features.
    fn standardize(&self, f: &Features) -> Standardized {
        let (lf, lo, dm) = (self.cfg.feature_layers.len(), self.cfg.option_layers.len(), self.cfg.d_model);
        let z = |x: &[f32], mu: &[f32], sd: &[f32]| -> Vec<f32> { x.iter().zip(mu).zip(sd).map(|((x, m), s)| (x - m) / s).collect() };
        match &self.port {
            Some(p) => {
                let items = (0..lf)
                    .map(|l| {
                        (0..3)
                            .map(|s| {
                                let r = (l * 3 + s) * dm..(l * 3 + s + 1) * dm;
                                p.lin[l].forward(&z(&f.items[r.clone()], &p.mu.data[r.clone()], &p.sd.data[r]))
                            })
                            .collect()
                    })
                    .collect();
                let opts = f
                    .options
                    .iter()
                    .map(|o| (0..lo).map(|l| p.lin[lf - 1].forward(&z(&o[l * dm..(l + 1) * dm], &p.omu[l * dm..(l + 1) * dm], &p.osd[l * dm..(l + 1) * dm]))).collect())
                    .collect();
                let r = (lf - 1) * 3 * dm..((lf - 1) * 3 + 1) * dm;
                let chunks = f.chunks.iter().map(|c| p.lin[lf - 1].forward(&z(c, &p.mu.data[r.clone()], &p.sd.data[r.clone()]))).collect();
                (items, opts, chunks)
            }
            None => {
                let items = (0..lf)
                    .map(|l| {
                        (0..3)
                            .map(|s| {
                                let r = (l * 3 + s) * dm..(l * 3 + s + 1) * dm;
                                z(&f.items[r.clone()], &self.feat_mu.data[r.clone()], &self.feat_sd.data[r])
                            })
                            .collect()
                    })
                    .collect();
                let opts = f
                    .options
                    .iter()
                    .map(|o| (0..lo).map(|l| z(&o[l * dm..(l + 1) * dm], &self.opt_mu.data[l * dm..(l + 1) * dm], &self.opt_sd.data[l * dm..(l + 1) * dm])).collect())
                    .collect();
                let r = (lf - 1) * 3 * dm..((lf - 1) * 3 + 1) * dm;
                let chunks = f.chunks.iter().map(|c| z(c, &self.feat_mu.data[r.clone()], &self.feat_sd.data[r.clone()])).collect();
                (items, opts, chunks)
            }
        }
    }

    fn memory_read(&self, m: &Memory, qx: &[f32], chunks: &[Vec<f32>]) -> Vec<f32> {
        let s = m.out.b.len();
        let (h, dh) = (m.heads, s / m.heads);
        let q = m.q.forward(qx);
        let n = chunks.len() as f32;
        let p_rows = m.pos.shape[0];
        let kvs: Vec<Vec<f32>> = chunks
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let pidx = (((i as f32 + 0.5) / n * p_rows as f32) as usize).min(p_rows - 1);
                let mut kv = m.kv.forward(&m.kv_ln.forward(c));
                add(&mut kv, &m.pos.data[pidx * 2 * s..(pidx + 1) * 2 * s]);
                kv
            })
            .collect();
        let scale = (dh as f32).sqrt();
        let mut r = vec![0f32; s];
        for hd in 0..h {
            let qh = &q[hd * dh..(hd + 1) * dh];
            let att = softmax_f32(&kvs.iter().map(|kv| dot(qh, &kv[hd * dh..(hd + 1) * dh]) / scale).collect::<Vec<_>>());
            for (a, kv) in att.iter().zip(&kvs) {
                for j in 0..dh {
                    r[hd * dh + j] += a * kv[s + hd * dh + j];
                }
            }
        }
        m.out.forward(&r)
    }

    /// Run the physics for one question with `kind` over its pooled features (halting settled particles).
    pub fn forward(&self, f: &Features, kind: QType) -> EngineOutput {
        let qt = qtype_index(kind);
        let (dz, lf) = (self.cfg.latent_dim, self.cfg.feature_layers.len());
        let (items, opts, chunks) = self.standardize(f);
        // ELMo-style scalar mixes over the tapped layers.
        let slots: Vec<Vec<f32>> = (0..3)
            .map(|s| {
                let w = softmax_f32(&self.mix.data[s * lf..(s + 1) * lf]);
                let mut v = vec![0f32; self.d];
                for (l, wl) in w.iter().enumerate() {
                    v.iter_mut().zip(&items[l][s]).for_each(|(a, x)| *a += wl * x);
                }
                v
            })
            .collect();
        let ow = softmax_f32(&self.opt_mix);
        let o: Vec<Vec<f32>> = opts
            .iter()
            .map(|layers| {
                let mut v = vec![0f32; self.d];
                for (l, wl) in ow.iter().enumerate() {
                    v.iter_mut().zip(&layers[l]).for_each(|(a, x)| *a += wl * x);
                }
                v
            })
            .collect();

        let mut s = self.world.forward(&slots[0]);
        if let (Some(m), false) = (&self.memory, chunks.is_empty()) {
            let qx: Vec<f32> = slots[1].iter().chain(&slots[2]).cloned().collect();
            add(&mut s, &self.memory_read(m, &qx, &chunks));
        }
        let s_next = {
            let u = self.foresight_inp.forward(&self.foresight_ln.forward(&s));
            let half = u.len() / 2;
            let g: Vec<f32> = (0..half).map(|i| gelu(u[i]) * sigmoid(u[half + i])).collect();
            let mut n = s.clone();
            add(&mut n, &self.foresight_out.forward(&g));
            n
        };
        let mut probe = self.probe.forward(&slots[1]);
        add(&mut probe, self.qemb(qt));
        let imp = self.impulse.forward(&slots[2]);
        let ctx: Vec<f32> = s.iter().chain(&s_next).chain(&probe).chain(&imp).cloned().collect();
        let mut z = self.to_z0.forward(&ctx);
        let mut p = self.to_p0.forward(&ctx);

        // Landscape: one Gaussian well per option; score levels sit on a rail.
        let k = o.len();
        let mut centres = Vec::with_capacity(k);
        let mut log_a = Vec::with_capacity(k);
        let mut sigma = Vec::with_capacity(k);
        let mut gaps = Vec::with_capacity(k);
        for ok in &o {
            let x: Vec<f32> = ok.iter().chain(&probe).cloned().collect();
            let e = self.well.forward(&x);
            let mut c = e[..dz].to_vec();
            add(&mut c, self.qemb(qt));
            centres.push(c);
            log_a.push(e[dz].clamp(-4.0, 4.0));
            sigma.push(0.3 + softplus(e[dz + 1]));
            gaps.push(softplus(e[dz + 2]));
        }
        if kind == QType::Score {
            let mut r0 = vec![0f32; dz];
            centres.iter().for_each(|c| add(&mut r0, c));
            r0.iter_mut().for_each(|v| *v /= k.max(1) as f32);
            let u = self.rail_dir.forward(&probe);
            let un = norm(&u).max(1e-12);
            let mut t = Vec::with_capacity(k);
            let mut acc = 0f32;
            for g in &gaps {
                acc += g;
                t.push(acc);
            }
            let mean = t.iter().sum::<f32>() / k.max(1) as f32;
            for (c, ti) in centres.iter_mut().zip(&t) {
                for j in 0..dz {
                    c[j] = r0[j] + (ti - mean) * (u[j] / un);
                }
            }
        }

        let kappa = self.log_kappa.exp();
        // U is not needed for the answer; the gradient and the well exponents q are.
        let potential = |z: &[f32]| -> (Vec<f32>, Vec<f32>) {
            let mut grad: Vec<f32> = z.iter().map(|v| kappa * v).collect();
            let mut q = Vec::with_capacity(k);
            for i in 0..k {
                let diff: Vec<f32> = z.iter().zip(&centres[i]).map(|(a, b)| a - b).collect();
                let d2: f32 = diff.iter().map(|x| x * x).sum();
                let s2 = sigma[i] * sigma[i];
                let qi = d2 / (2.0 * s2);
                let g = (log_a[i] - qi).exp();
                grad.iter_mut().zip(&diff).for_each(|(gr, df)| *gr += g * df / s2);
                q.push(qi);
            }
            (grad, q)
        };
        let a: Vec<f32> = self.a_log.iter().map(|x| x.exp()).collect();
        let mut h = vec![0f32; a.len()];
        let (mut grad, mut q) = potential(&z);
        let mut gn = norm(&grad);
        let (mut settled, mut steps_used) = (false, self.cfg.n_steps);
        for t in 0..self.cfg.n_steps {
            let x: Vec<f32> = z.iter().chain(&p).chain(&grad).cloned().chain(std::iter::once(gn)).collect();
            let u = self.ssm_in.forward(&x);
            let nm = u.len() - 2;
            let dt = self.cfg.dt_max * sigmoid(u[nm]);
            let gamma = softplus(u[nm + 1]);
            for i in 0..nm {
                h[i] = (-a[i] * dt).exp() * h[i] + dt * u[i];
            }
            let force: Vec<f32> = self.ssm_out.forward(&h).into_iter().map(|v| self.force_scale * v.tanh()).collect();
            if !settled {
                let damp = (-gamma * dt).exp();
                let cap = 1.0 + gn / self.cfg.force_max;
                for j in 0..dz {
                    p[j] = p[j] * damp - dt * (grad[j] / cap) + dt * force[j];
                    z[j] += dt * p[j];
                }
            }
            (grad, q) = potential(&z);
            gn = norm(&grad);
            let kin = 0.5 * p.iter().map(|v| v * v).sum::<f32>();
            if !settled && kin < self.cfg.settle_eps && gn < 10.0 * self.cfg.settle_eps {
                settled = true;
                steps_used = t + 1;
            }
        }
        let logits = q.iter().zip(&log_a).map(|(qi, la)| -(qi - la)).collect();
        let nearest = q.iter().cloned().fold(f32::INFINITY, f32::min);
        EngineOutput { logits, kinetic: 0.5 * p.iter().map(|v| v * v).sum::<f32>(), nearest, steps_used }
    }

    /// Thermal calibration, conformal set and flags, exactly as `Vega.decide` reads them (alpha 0.1, default risk).
    pub fn calibrate(&self, out: &EngineOutput, kind: QType) -> VegaAnswer {
        let qt = qtype_index(kind);
        let k = out.logits.len();
        let phys = [(out.kinetic as f64).ln_1p(), (out.nearest as f64).ln_1p(), out.steps_used as f64 / self.cfg.n_steps.max(1) as f64, (k.max(2) as f64).ln()];
        let tau = match &self.cfg.calibration {
            Some(c) => {
                let th = &c.theta[qt];
                let lt = th[0] + th[1..].iter().zip(&phys).map(|(w, x)| w * x).sum::<f64>();
                lt.clamp(0.2f64.ln(), 20f64.ln()).exp()
            }
            None => 1.0,
        };
        let zs: Vec<f64> = out.logits.iter().map(|l| *l as f64 / tau).collect();
        let m = zs.iter().cloned().fold(f64::MIN, f64::max);
        let e: Vec<f64> = zs.iter().map(|z| (z - m).exp()).collect();
        let sum: f64 = e.iter().sum();
        let probs: Vec<f64> = e.into_iter().map(|x| x / sum).collect();
        let (best, confidence) = probs.iter().cloned().enumerate().fold((0, f64::MIN), |a, (i, v)| if v > a.1 { (i, v) } else { a });
        let cal = self.cfg.calibration.as_ref();
        let th = cal.and_then(|c| conformal_for(c, kind)).and_then(|t| t.get("0.1").copied());
        let prediction_set = match th {
            Some(th) => (0..k).filter(|&i| probs[i] >= (1.0 - th).max(1e-12) || i == best).collect(),
            None => vec![best],
        };
        let abstain = cal.map(|c| confidence < c.abstain_threshold.get(&c.default_risk).copied().unwrap_or(0.0)).unwrap_or(false);
        let unbound = cal.map(|c| out.nearest as f64 > c.unbound_threshold).unwrap_or(false);
        VegaAnswer { probs, best, confidence, prediction_set, abstain, unbound, settle_steps: out.steps_used }
    }
}

/// `normalise_conformal`: a single unrecognised key stands in for the single missing type name.
fn conformal_for(c: &Calibration, kind: QType) -> Option<&HashMap<String, f64>> {
    let name = qtype_name(kind);
    if let Some(t) = c.conformal.get(name) {
        return Some(t);
    }
    let known = ["choice", "score", "boolean"];
    let unknown: Vec<&String> = c.conformal.keys().filter(|k| !known.contains(&k.as_str())).collect();
    let absent: Vec<&str> = known.iter().copied().filter(|k| !c.conformal.contains_key(*k)).collect();
    (unknown.len() == 1 && absent == [name]).then(|| &c.conformal[unknown[0]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activations_match_torch_reference_values() {
        // torch.nn.functional.gelu / softplus at a few points.
        assert!((gelu(1.0) - 0.841_344_7).abs() < 1e-6);
        assert!((gelu(-2.0) - -0.045_500_26).abs() < 1e-6);
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-7);
        assert_eq!(softplus(25.0), 25.0);
    }

    #[test]
    fn renamed_conformal_key_maps_to_the_missing_type() {
        let c = Calibration {
            theta: vec![vec![0.0; 5]; 3],
            conformal: [("choice", 0.7), ("score", 0.8), ("noul", 0.5)].iter().map(|(k, v)| (k.to_string(), [("0.1".to_string(), *v)].into())).collect(),
            abstain_threshold: HashMap::new(),
            default_risk: "0.1".into(),
            unbound_threshold: 1.0,
        };
        assert_eq!(conformal_for(&c, QType::Boolean).unwrap()["0.1"], 0.5);
        assert_eq!(conformal_for(&c, QType::Choice).unwrap()["0.1"], 0.7);
    }
}
