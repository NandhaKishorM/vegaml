//! Usage: cargo run --release --example golden -- <golden.json> <checkpoint-dir> <reader.gguf>
//!
//! `golden.json` comes from `tests/reference/golden_reference.py`. Three layers are checked:
//! tokenization and spans, the engine on the reference's own features (asserted to 1e-4), and the
//! end-to-end answer with the GGUF reader (drift is reported, not asserted).

use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};
use llama_cpp_2::llama_backend::LlamaBackend;
use serde_json::Value;
use vegaml::{Features, QType, Spans, Vega, VegaQuestion};

fn floats(v: &Value) -> Vec<f32> {
    v.as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN) as f32).collect()).unwrap_or_default()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        ab += (*x as f64) * (*y as f64);
        aa += (*x as f64).powi(2);
        bb += (*y as f64).powi(2);
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

fn question(q: &Value) -> Result<VegaQuestion> {
    Ok(VegaQuestion {
        kind: q["type"].as_str().context("q.type")?.parse()?,
        instructions: q["instructions"].as_str().context("q.instructions")?.to_string(),
        options: q["options"].as_array().context("q.options")?.iter().filter_map(|o| o.as_str().map(String::from)).collect(),
    })
}

fn spans(g: &Value) -> Result<Spans> {
    let pair = |v: &Value| (v[0].as_u64().unwrap_or(0) as usize, v[1].as_u64().unwrap_or(0) as usize);
    let sp = &g["spans"];
    Ok(Spans { state: pair(&sp["state"]), question: pair(&sp["question"]), options: sp["options"].as_array().context("spans.options")?.iter().map(pair).collect() })
}

fn argmax(p: &[f64]) -> usize {
    p.iter().cloned().enumerate().fold((0, f64::MIN), |a, (i, v)| if v > a.1 { (i, v) } else { a }).0
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 4 {
        bail!("usage: golden <golden.json> <checkpoint-dir> <reader.gguf>");
    }
    let golden: Vec<Value> = serde_json::from_slice(&std::fs::read(&a[1]).context("read golden")?)?;
    let mut backend = LlamaBackend::init()?;
    backend.void_logs();
    let threads = std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4).min(8);
    let mut v = Vega::load(&backend, &PathBuf::from(&a[2]), &PathBuf::from(&a[3]), 4096, threads)?;
    let (mut tok_ok, mut agree) = (0usize, 0usize);
    let (mut engine_err, mut prob_err, mut drift_max, mut cos_min) = (0f64, 0f64, 0f64, 1f64);
    for g in &golden {
        let name = g["name"].as_str().unwrap_or("?");
        let q = question(&g["q"])?;
        let state = g["state"].as_str().unwrap_or_default();
        let ref_ids: Vec<i32> = g["ids"].as_array().context("ids")?.iter().map(|x| x.as_i64().unwrap_or(-1) as i32).collect();
        let (ids, _) = v.prompt(state, &q);
        if ids == ref_ids {
            tok_ok += 1;
        } else {
            let first = ids.iter().zip(&ref_ids).position(|(x, y)| x != y).unwrap_or(ids.len().min(ref_ids.len()));
            println!("TOKENS differ for {name}: {} vs {} tokens, first difference at {first}", ids.len(), ref_ids.len());
        }
        let rf = Features {
            items: floats(&g["feats"]),
            options: g["opts"].as_array().context("opts")?.iter().map(floats).collect(),
            chunks: g["chunks"].as_array().map(|c| c.iter().map(floats).collect()).unwrap_or_default(),
        };
        let out = v.engine.forward(&rf, q.kind);
        let ref_logits = floats(&g["logits"]);
        ensure!(ref_logits.len() == out.logits.len(), "{name}: logit count differs");
        let err = out.logits.iter().zip(&ref_logits).map(|(x, y)| ((x - y).abs() / y.abs().max(1.0)) as f64).fold(0f64, f64::max);
        let ref_p: Vec<f64> = g["probs"].as_array().context("probs")?.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let ans = v.engine.calibrate(&out, q.kind);
        let perr = ans.probs.iter().zip(&ref_p).map(|(x, y)| (x - y).abs()).fold(0f64, f64::max);
        engine_err = engine_err.max(err);
        prob_err = prob_err.max(perr);
        let ours = v.reader.pooled(&ref_ids, &spans(g)?, &v.engine.cfg)?;
        cos_min = cos_min.min(cosine(&ours.items, &rf.items));
        let e2e = v.engine.calibrate(&v.engine.forward(&ours, q.kind), q.kind);
        let drift = e2e.probs.iter().zip(&ref_p).map(|(x, y)| (x - y).abs()).fold(0f64, f64::max);
        drift_max = drift_max.max(drift);
        agree += (e2e.best == argmax(&ref_p)) as usize;
        let kind = match q.kind {
            QType::Boolean => "boolean",
            QType::Score => "score",
            QType::Choice => "choice",
        };
        println!("{name:<12} {kind:<7} engine {err:.1e} prob {perr:.1e} | reader best {} ref {} drift {drift:.3}", e2e.best, argmax(&ref_p));
    }
    let n = golden.len();
    println!("\ntokenization identical: {tok_ok}/{n}");
    println!("engine on reference features: max rel logit error {engine_err:.2e}, max prob error {prob_err:.2e}");
    println!("GGUF reader: top-1 agreement {agree}/{n}, max prob drift {drift_max:.3}, min item-feature cosine {cos_min:.5}");
    ensure!(engine_err < 1e-4 && prob_err < 1e-4, "engine port does not match the reference");
    Ok(())
}
