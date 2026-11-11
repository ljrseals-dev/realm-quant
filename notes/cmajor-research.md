```markdown
# Cmajor Code-as-Weights — Closed Research Note

**Author:** ljrseals-dev — Lewis James Seals Jr
**System:** UNREAL AI Systems — HarmoniaCodec
**Status:** Closed. Compiler project, not a delivery path for this graph class.
**Date:** 10 November 2026

---

## Verdict

Cmajor's code-as-weights pipeline is architecturally elegant and, for the MIT
descript DAC 24 kHz class, not viable. The failure is operator coverage, not
concept. The shipped `Conv` processor is structurally unsuitable for
convolutional audio codecs, and the encoder carries a node cluster that requires
language features Cmajor does not have. This note records what was measured,
what was read from source, and what would be required to reopen the route.

---

## What was done

- Cloned `cmajor-lang/cmajor`, release 1.0.3209.
- Ran `tools/onnxToCmajor.py` against `onnx-community/dac_24khz-ONNX` decoder
  (`decoder_model.onnx`, 202 MB, opset 14, 52.89 M params) and encoder
  (`encoder_model.onnx`, 87 MB, opset 14, 1,574 nodes).
- Ran the same converter against the FP16 and INT8 siblings.
- Read `tools/onnx/patchTemplate/onnx.cmajor` source for the processor set and
  the `Conv` implementation.
- Did **not** add processors, **not** compile-check any shim, **not** produce
  WASM. `cmaj` does not start in the test environment.

---

## Processor set (release 1.0.3209)

`tools/onnx/patchTemplate/onnx.cmajor` implements exactly sixteen processors:

```

Tanh, Reshape, MatMul, Gemm, Add, Unsqueeze, Squeeze, Conv,
GRU, LSTM, Slice, Concat, Gather, Pad, Transpose, Constant

```

Notably absent: `Mul`, `Sin`, `Pow`, `ConvTranspose`, `ArgMax`, `ReduceL2`,
`ReduceSum`, `Shape`, `Expand`, `ConstantOfShape`. The absence of `Mul` — one of
the most basic operations in any neural network — is the single clearest signal
that the processor set was not designed for this graph class.

`onnxToCmajor.py` does not reject unknown ops. It emits
`onnx::operation::<OpType>` for every node; failure surfaces at Cmajor compile
time when the named processor does not exist. This is a deliberate design
choice (permissive converter, strict compiler), and it means the converter's
success is not evidence of anything.

---

## Decoder operator census

347 compute nodes (plus 247 `Constant`). Name-match against the sixteen-processor
set: **227 / 347**.

| Op | Count | Cmajor processor | Category |
|---|---:|---|---|
| `Constant` | 247 | yes | — |
| `Add` | 72 | yes | — |
| `Gather` | 64 | yes | codebook index |
| `Conv` | 58 | yes | **structurally broken, see below** |
| `Mul` | 58 | **no** | Snake only |
| `Transpose` | 32 | yes | — |
| `Sin` | 29 | **no** | Snake only |
| `Pow` | 29 | **no** | Snake only |
| `ConvTranspose` | 4 | **no** | upsample ladder |
| `Tanh` | 1 | yes | — |

The 120 unmatched nodes collapse to two gaps:

**Snake**, 29 sites, ONNX expression `Mul → Sin → Pow → Mul/Add`. Three missing
pointwise processors. Addable, low risk.

**ConvTranspose**, 4 sites, the only temporal upsamplers:

| Block | Weight | Kernel | Stride | Pads |
|---|---|---|---|---|
| 0 | `[1536, 768, 16]` | 16 | 8 | 4, 4 |
| 1 | `[768, 384, 10]` | 10 | 5 | 3, 3 |
| 2 | `[384, 192, 8]` | 8 | 4 | 2, 2 |
| 3 | `[192, 96, 4]` | 4 | 2 | 1, 1 |

Each rank-3 `ConvTranspose1d`, non-zero pads, stride > 1. No processor exists.
Not a rank shim; a full operator.

The decoder gap is manageable engineering: add four processors, fix one. This
was the "four processors and a shim" framing in early turns. The next section
falsifies the shim.

---

## The `Conv` processor structural failure

Source read of the shipped `Conv` processor establishes three behaviours that
disqualify it for this graph class:

| Behaviour | Consequence |
|---|---|
| Sets `widthPadding = 0`, never reads ONNX `pads` | Every same-padded conv emits wrong-length output |
| Takes dilation as fourth argument | Dilation expressible; pads not, so dilated convs wrong-length |
| **No `stride` argument at all** | Encoder's four strided downsamplers cannot be expressed |

DAC conv census (both sides):

- **Decoder**: 58 convs — 44× 1×1 zero-pad, 6× kernel-7 pad-3, 4× dilation-3, 4× dilation-9.
- **Encoder**: 93 convs — includes 4 strided downsamplers (`[128,64,4] k4 s2 p1`,
  `[256,128,8] k8 s4 p2`, `[512,256,10] k10 s5 p3`, `[1024,512,16] k16 s8 p4`),
  plus dilation-3 and dilation-9 residuals.

Every conv has non-zero padding, non-unit stride, or non-unit dilation. A rank
shim that unsqueezes `H=1` and reshapes `[O,I,K] → [O,I,1,K]` satisfies the
`static_assert` and then produces the wrong tensor for **all 151 convs**.

The honest correction: **the shipped `Conv` processor must be rewritten to read
`pads` and accept `strides`.** That is not a patch; it is a rewrite of the
language's most important operator for a use case it was not designed for.

---

## Encoder operator census and the RVQ problem

1,574 compute nodes. Name-match: **702 / 1,574**.

The 872 unmatched nodes split into two groups:

**Group A — Snake, 29 sites.** Same as decoder. `Mul`, `Sin`, `Pow`. Easy.

**Group B — RVQ nearest-code search, ~750 nodes.**

| Op | Count | Category |
|---|---:|---|
| `Shape` | 157 | Dynamic shape |
| `Expand` | 94 | Dynamic shape |
| `Sub` | 94 | Pointwise |
| `Pow` | 92 | Pointwise |
| `Mul` | 90 | Pointwise |
| `Div` | 63 | Pointwise |
| `ReduceSum` | 63 | Axis reduction |
| `ConstantOfShape` | 62 | Dynamic shape |
| `ReduceL2` | 32 | Axis reduction |
| `Clip` | 32 | Pointwise |
| `Neg` | 32 | Pointwise |
| **`ArgMax`** | **32** | **Reduction-with-index** |
| `Sin` | 29 | Pointwise |

This is not "a few more processors." It is four separate conceptual gaps:

1. **Pointwise** (`Sub`, `Div`, `Clip`, `Neg`, `Mul`, `Pow`, `Sin`) — addable.
2. **Axis reductions returning values** (`ReduceSum`, `ReduceL2`) — addable but
   rank-changing; none of the sixteen existing processors change rank.
3. **Reduction returning an index** (`ArgMax`) — a language-feature addition.
   Cmajor has no processor returning a position rather than a value.
4. **Dynamic shape** (`Shape`, `Expand`, `ConstantOfShape`) — the deepest
   problem. These produce tensors whose shape depends on runtime contents.
   A static compiler cannot lower them without compile-time shape knowledge or
   a runtime shape-tracking mechanism, neither of which Cmajor has. `Shape` at
   157 occurrences is the single most frequent unmatched op in the encoder.

The encoder's gap is a research problem. The decoder's gap is a task. The
difference matters: it means the encoder cannot be compiled while the decoder
can (after the Conv rewrite), and any serious Cmajor effort would have to accept
that split.

---

## Literal expansion and the emission problem

`onnxToCmajor.py` has two initializer modes: inline weights as literal constants,
or emit typed slot references. Inline mode is what makes code-as-weights
attractive; the compiler can specialize around the actual values.

Measured expansion ratio on DAC codebook weights: **5.39×**. A float32 weight
renders as `str(3.14159...)`, a decimal string longer than its binary
representation.

Applied:
- Decoder: 211.6 MB f32 → **~1.14 GB of `.cmajor` text**.
- Encoder: 90.3 MB f32 → **~487 MB of `.cmajor` text**.

The decoder figure exceeds the 3.6 GB RAM of the test environment before the
compiler starts. The structural patch (typed slots) is 162 KB for the decoder
and 492 KB for the encoder — a 7,000× reduction — but it defeats the entire
point: with slot references, weights load at runtime and the compiler cannot
specialize. The generated code becomes a runtime interpreter with Cmajor syntax,
offering no advantage over ONNX Runtime Web.

The escape is quantization before emission: ternary weights `-1`, `0`, `+1`
emit as single-character literals, eliminating the 5.39× expansion entirely.
But the emitter's `getDataType` whitelist is `float32`, `float64`, `int32`,
`int64`, `bool` — no ternary, no int8, no float16. The emitter needs extension
before quantization can be exploited.

This is the second structural problem (after operator coverage) and it is not
smaller than the first.

---

## Quantized sibling results

| File | Raw | Nodes | Match | Emitter |
|---|---:|---:|---:|---|
| decoder f32 | 211.6 MB | 347 | 227 | emits graph; literals not inlined |
| decoder fp16 | 105.8 MB | 348 | 227 (+1 `Cast`) | **aborts on `FLOAT16`** |
| decoder int8 | 120.6 MB | 663 | 285 | aborts on `UINT8` after shape inference |
| encoder f32 | 90.3 MB | 1,574 | 702 | emits graph; weights elided |
| encoder int8 | 24.5 MB | 2,167 | 794 | same `UINT8` wall |

FP16 is the only one that shrinks the payload (211.6 → 105.8 MB decoder) without
changing the operator multiset, and the script cannot name that dtype. INT8 does
not help coverage: decoder `Conv` becomes `ConvInteger` (58), plus
`DynamicQuantizeLinear` (26) and `Cast` (58) — four new missing processors. The
encoder's int8 file is the smallest (24.5 MB) and the worst graph (2,167 nodes,
`ConvInteger` 93, `DynamicQuantizeLinear` 94, `DequantizeLinear` 31,
`MatMulInteger` 1).

No WASM was produced, so there is no measured WASM size delta.

---

## SoulX-Singer — the failure surface, read from source

Read from `Soul-AILab/SoulX-Singer` `main` (`flow_matching.py`, `llama.py`,
`decoder.py`, `vocoder.py`) and the published `config.yaml` (`flow_matching`:
22 layers, hidden 1024, 16 heads, cosine time, `n_steps: 32`, `cfg: 3`). Weights
not downloaded.

The vocoder is **Vocos** (ConvNeXt + ISTFT/IMDCT), not a DAC-style stack.
Against the same sixteen processors, a single forward already needs:

- **Attention** — `Softmax`, `Div`, `Sqrt`, `Mul` (only `MatMul`/`Gemm` exist)
- **LlamaAdaptiveRMSNorm** — `Rsqrt`, `Mul`, `ReduceMean`
- **SiLU / GELU** — `Sigmoid`, `Mul`, `Erf`
- **ConvTranspose1d** in the flow resampler — same hole as DAC, plus rank-3 Conv1d
- **SinusoidalPosEmb** — `Sin`, `Cos`
- **Vocos head** — `STFT` / `DFT` / complex `Exp`; none of those exist
- **The sampler** — Python ODE loop, 32 steps, CFG is a second forward. No
  `Loop` / `Scan` / `If`. Unrolling is 32 copies of a 22-layer 1024-d transformer.

DAC's hole is four processors plus a Conv rewrite. SoulX's hole is norms,
attention, activations, a spectral head, and control flow. Code-as-weights stays
a DAC problem first; SoulX is a separate frontier and likely a separate project.

---

## Environment blocker

`cmaj` 1.0.3209 for Linux x64 does not start:

```

libgtk-3.so.0
libwebkit2gtk-4.0.so.37
libsoup-2.4.so.1
libjavascriptcoregtk-4.0.so.18
libjack.so.0

```

Missing, and packages cannot be installed in this environment. Consequences:

- No shim was compile-checked.
- No processor was added.
- No WASM was produced.
- Statements about what a processor addition *would* achieve are source-level
  readings, not compile-verified results.

A working `cmaj` is a prerequisite for reopening this route.

---

## What would be required to reopen

For the **decoder**:

1. Rewrite `Conv` to read `pads` and accept `strides` — not a patch.
2. Add `ConvTranspose` processor (4 sites, rank-3).
3. Add `Mul`, `Sin`, `Pow` processors (Snake, 29 sites).
4. Extend `getDataType` to include `FLOAT16` and quantized types.
5. Quantize before emission to avoid the 5.39× literal expansion.
6. Get `cmaj` to load in the target environment.
7. Compile-check everything. Nothing above has been compile-checked.

For the **encoder**, additionally:

8. Add `ArgMax`, `ReduceL2`, `ReduceSum` — the first two are language features,
   not processors.
9. Handle `Shape`, `Expand`, `ConstantOfShape` — a research problem requiring
   dynamic-shape support Cmajor does not have.

For **SoulX-Singer**, additionally: attention, RMSNorm, SiLU/GELU, positional
encoding, Vocos STFT/DFT, and the 32-step ODE sampler's control flow — a
compiler project.

Items 1–5 are a substantial engineering task. 6–7 are environmental. 8–9 are
research problems. The SoulX list is a compiler project. **Code-as-weights, for
this graph class, is a compiler project.**

---

## Verdict restated

Cmajor's code-as-weights is the right long-term design: weights as compile-time
constants, LLVM-optimized WASM, potential to specialize around the actual weight
values. It is not a delivery path for a browser demo in the near term. Every
measurement in the accompanying technical report is consistent with that
conclusion.

The recommendation stands: **move the browser path to ONNX Runtime Web**, which
handles every op in the census natively — `Conv` with pads and strides,
`ConvTranspose`, `ArgMax`, `ReduceL2`, `Cast`, FP16, INT8. Retain Cmajor as a
research direction. Do not attempt the processor additions without a working
`cmaj` and a settled quantized emitter.

---

## Artifacts referenced

- `cmajor-lang/cmajor`, release 1.0.3209 — `tools/onnx/onnxToCmajor.py`,
  `tools/onnx/patchTemplate/onnx.cmajor`
- `onnx-community/dac_24khz-ONNX` — decoder/encoder FP32, FP16, INT8
- `Soul-AILab/SoulX-Singer` — source read only, no weights downloaded
- `MEASUREMENTS.md` — the ONNX Runtime Web baseline numbers
- `PAPER.md` — the full technical report

---

*End of note. Closed.*
```
