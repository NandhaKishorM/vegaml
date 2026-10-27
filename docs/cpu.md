# Serving on a CPU box

A bank's on-prem machine often has no GPU, and the published CPU figure for a decision was 703.8
ms against 118.7 ms on Metal. This page is what we found when we went looking for the 6x, and the
short answer is that most of it was configuration rather than arithmetic.

Everything below is one process per configuration on one box: Apple M5, 10 cores, 16 GB, macOS
26.5.2, torch 2.14.1, vegaml 0.13.0, float32 unless the row says otherwise. Twelve short states
(57 characters mean) and one three-option choice question, median of twelve after two warmups.
`bench/cpu_path.py` runs the same sweep on your box, which is the only one whose numbers matter to
you.

## The thread count belongs to the reader

| torch threads | Qwen3.5-0.8B (the default reader) | gemma-3-1b-it |
| --- | --- | --- |
| 1 | **258.2 ms** | 167.6 ms |
| 2 | 763.8 ms | 138.9 ms |
| 4 (torch's default here) | 1268.1 ms | **135.2 ms** |
| 6 | 1852.0 ms | 142.7 ms |
| 10 | 3612.1 ms | 146.3 ms |

The default reader gets 4.9x slower going from one thread to four and 14x slower going to ten. The
cause is not the engine and not the prompt length: Qwen3.5's linear-attention blocks fall back to a
reference PyTorch implementation when `causal_conv1d` and `flash-linear-attention` are missing,
which on CPU they always are, and that fallback does not thread. A plain-transformer reader has no
such fallback and behaves the ordinary way, mildly preferring four.

The thread count changes nothing about the answers. Across the twelve cases, one thread and four
threads agreed on 12 of 12 choices with a confidence difference of 0.0000. It is free.

It holds at length too. At 855 characters: 546.9 ms at one thread, 1639.2 ms at four, 4348.1 ms at
ten. At 3,426 characters: 1018.8 ms at one, 2853.2 ms at four.

```python
v = vegaml.load(device="cpu", threads=1)                       # the default reader
v = vegaml.load("nandakishorm/vega-U", backbone="google/gemma-3-1b-it",
                device="cpu", threads=4)                       # a plain-transformer reader
```

`vegaml --threads 1` and `vegaml-calibrate fit --threads 1` take it too. Left unset, the loader
prints one line saying what torch chose and that it is worth measuring.

## The reader is the other lever, and the engine does not change

The engine is the same 14.3M parameters behind every reader, checksum-identical, so swapping the
reader for a CPU deployment is a configuration change and not a different model. On the six-corpus
run, fitted and calibrated:

| corpus | Qwen3.5-0.8B | gemma-3-1b-it |
| --- | --- | --- |
| tweet-sentiment | 0.660 | **0.667** |
| tweet-sentiment-shifted | **0.642** | 0.600 |
| finance-sentiment | 0.761 | **0.799** |
| moderation | 0.629 | 0.629 |
| boolq | **0.500** | 0.494 |
| counterfactual | **0.883** | 0.845 |
| mean accuracy | **0.679** | 0.672 |
| mean ECE | 0.069 | **0.059** |

Seven tenths of a point of accuracy behind, one point of calibration error ahead, and 135.2 ms
instead of 258.2 ms. Against the published 703.8 ms that is 5.2x, with no quantization and no GPU
in the box.

## int8 is a memory path, not a speed path

`quantize="int8"` puts dynamic int8 over every linear layer in the reader. What it buys is bytes:

| | weights | median latency | same choice | mean confidence shift |
| --- | --- | --- | --- | --- |
| float32, 1 thread | 3.75 GB | 258.2 ms | (the reference) | |
| int8, 1 thread | **0.95 GB** | 1048.8 ms | 10 of 12 | 0.102, max 0.257 |

Four times the weight reduction, four times the latency, and it moves two answers in twelve. The
latency is a backend fact rather than an int8 fact: this box has `qnnpack`, while float32 here
runs through Apple's AMX path at a measured 1,357 GFLOP/s, so there is nothing for int8 to beat. A
Linux x86 box has `fbgemm` instead, where the comparison is different and we have not measured it.

So: reach for int8 when the box cannot hold 3.75 GB of weights, measure it before you reach for it
to go faster, and never ship it without checking the answers it changes. The loader reports which
backend it selected, and raises rather than falling back silently if torch was built without one.

bfloat16 and float16 are not CPU paths at all on this build: both run matmuls at about 1.5
GFLOP/s, a thousandth of float32.

## What is still true

The engine itself is 14.3M parameters and is not where the time goes; one frozen read of the state
is. Several questions about one state cost about one read, so a sheet of ten questions is far
cheaper than ten decisions. A fitted head answers in 13.9 microseconds once the features exist,
which means the read is the entire cost on CPU as well as on GPU.
