"""Export the decision engine to ONNX and run it anywhere onnxruntime does.

    pip install "vegaml[onnx]"

    from vegaml.onnx_export import export_engine, ONNXEngine
    export_engine("path/to/checkpoint", "engine.onnx")
    eng = ONNXEngine("engine.onnx")
    out = eng(feats, opts, opt_mask, qtype)             # numpy in, numpy out

What ships is the settling computation: pooled perception features in, well energies and the
physics readouts out, with the integrator's steps unrolled into the graph at the checkpoint's
own n_steps. Batch size and option count stay dynamic axes, so one file serves any request
shape. The calibrated probability layer above the engine (thermal temperatures, conformal
sets) is plain numpy already and needs no runtime at all.

Scope: the base engine, halt semantics included. A checkpoint with the long-state memory
reader exports with two extra inputs (chunks, chunk_mask); a record without chunks passes an
all-false mask and the reader contributes exactly zero, matching the torch path.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

_OUTPUTS = ("logits", "zT", "kinetic", "grad_norm", "nearest", "steps_used")


def _load_engine(model_dir: str):
    from safetensors.torch import load_file

    from ._common import VegaEngine
    with open(os.path.join(model_dir, "vega_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    eng = VegaEngine(cfg["d_model"], len(cfg["feature_layers"]), len(cfg["option_layers"]), cfg)
    eng.load_state_dict(load_file(os.path.join(model_dir, "engine.safetensors")))
    return eng.eval(), cfg


def export_engine(model_dir_or_engine, out_path: str, halt: bool = True, opset: int = 18,
                  sample: Optional[Dict] = None) -> str:
    """Write the engine's forward as an ONNX graph. Returns out_path.

    `sample` overrides the example tensors the tracer runs (dict with feats/opts/opt_mask/qtype,
    plus chunks/chunk_mask for a memory checkpoint); the default is a small random batch shaped
    from the engine itself. The example's shapes do not pin the graph: batch and option count
    are exported as dynamic axes.
    """
    import torch
    if isinstance(model_dir_or_engine, str):
        eng, _ = _load_engine(model_dir_or_engine)
    else:
        eng = model_dir_or_engine.eval()
    has_memory = eng.memory is not None

    class _Wrapped(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, feats, opts, opt_mask, qtype, chunks=None, chunk_mask=None):
            out = self.inner(feats, opts, opt_mask, qtype, halt=halt, return_traj=False,
                             chunks=chunks, chunk_mask=chunk_mask)
            return tuple(out[k] for k in _OUTPUTS)

    d = eng.feat_mu.shape[-1]
    Lf, Lo = eng.feat_mu.shape[0], eng.opt_mu.shape[0]
    if sample is None:
        g = torch.Generator().manual_seed(0)
        sample = {"feats": torch.randn(2, Lf, 3, d, generator=g),
                  "opts": torch.randn(2, 3, Lo, d, generator=g),
                  "opt_mask": torch.ones(2, 3, dtype=torch.bool),
                  # One score question in the example, or the tracer drops the rail branch and
                  # every score row the deployed graph ever sees reads unrailed centres.
                  "qtype": torch.tensor([1, 0])}
        if has_memory:
            C = eng.memory_chunks
            sample["chunks"] = torch.randn(2, C, d, generator=g)
            sample["chunk_mask"] = torch.ones(2, C, dtype=torch.bool)
    else:
        sample = {k: torch.as_tensor(v) for k, v in sample.items()}
    names = ["feats", "opts", "opt_mask", "qtype"] + (["chunks", "chunk_mask"] if has_memory else [])
    dyn = {"feats": {0: "batch"}, "opts": {0: "batch", 1: "options"}, "qtype": {0: "batch"},
           "opt_mask": {0: "batch", 1: "options"}}
    if has_memory:
        dyn["chunks"] = {0: "batch"}
        dyn["chunk_mask"] = {0: "batch"}
    dyn.update({k: {0: "batch"} for k in _OUTPUTS if k != "logits"})
    dyn["logits"] = {0: "batch", 1: "options"}
    with torch.no_grad():
        torch.onnx.export(_Wrapped(eng), tuple(sample[n] for n in names), out_path,
                          input_names=names, output_names=list(_OUTPUTS), dynamic_axes=dyn,
                          opset_version=opset, dynamo=False)
    return out_path


class ONNXEngine:
    """Run an exported engine through onnxruntime: numpy in, the torch readout names out."""

    def __init__(self, path: str, providers=None):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=providers or ["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.sess.get_inputs()]

    def __call__(self, feats, opts, opt_mask, qtype, chunks=None, chunk_mask=None) -> Dict:
        feed = {"feats": np.asarray(feats, dtype=np.float32),
                "opts": np.asarray(opts, dtype=np.float32),
                "opt_mask": np.asarray(opt_mask, dtype=bool),
                "qtype": np.asarray(qtype, dtype=np.int64)}
        if "chunks" in self.inputs:
            if chunks is None or chunk_mask is None:
                B = feed["feats"].shape[0]
                d = feed["feats"].shape[-1]
                C = self.sess.get_inputs()[self.inputs.index("chunks")].shape[1]
                chunks = np.zeros((B, C, d), dtype=np.float32)
                chunk_mask = np.zeros((B, C), dtype=bool)
            feed["chunks"] = np.asarray(chunks, dtype=np.float32)
            feed["chunk_mask"] = np.asarray(chunk_mask, dtype=bool)
        return dict(zip(_OUTPUTS, self.sess.run(list(_OUTPUTS), feed), strict=True))
