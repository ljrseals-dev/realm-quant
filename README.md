Realm of Quant: Cmajor Gigs to Browser

Weight Representation, Operator Coverage, and Runtime Residency in Browser-Delivered Neural Audio

---

Article type: Technical Report (preprint) · Not peer reviewed
Running head: Realm of Quant
Manuscript date: 10 November 2026
Version: 1.0

Author
Lewis James Seals Jr (ljrseals-dev)
UNREAL AI Systems — HarmoniaCodec Project

Correspondence: via repository issue tracker
ORCID: not registered at time of writing

Preprint DOI: https://doi.org/10.5281/zenodo.24123282

---

Abstract

Delivering neural singing voice synthesis to the browser is conventionally framed as a model-size problem. This report argues the framing is mis-specified. Size is the least restrictive constraint. The binding problems are (i) numerical representation under an information-theoretic ceiling that lossless archiving cannot breach, (ii) the operator coverage of whichever compiler or runtime executes the graph, (iii) the memory residency of the runtime on constrained devices, and (iv) the realtime factor achieved on hardware resembling the target device rather than the development machine. These four dimensions are jointly constraining: optimizing any one in isolation yields non-viable systems, and the failure modes are not intuitive.

Two delivery routes are evaluated against a real neural audio codec, the MIT descript DAC 24 kHz. The first is Cmajor's code-as-weights pipeline, which lowers an ONNX graph to a statically compiled WebAssembly program in which weight values become compile-time constants. The second is ONNX Runtime Web, which executes the ONNX graph directly under WASM or WebGPU backends. The Cmajor route fails on operator coverage rather than on conceptual grounds: its shipped Conv processor ignores ONNX pads, exposes no stride argument, and hardcodes widthPadding = 0, mis-lowering all 151 convolutions in the codec; the encoder additionally carries approximately 750 nodes in the residual-vector-quantizer nearest-code search that depend on reduction-with-index and dynamic-shape primitives a static compiler cannot express. The ONNX Runtime Web route succeeds in execution and fails in speed. Both FP16 graphs load and run with correct output shapes; combined wire payload is 151.5 MB. Realtime factor on a 2-vCPU Xeon Platinum 8481C, WASM backend, single thread, is 3.16 (decoder) and 1.43 (encoder), with Chrome resident-memory deltas of 524 MB and 299 MB respectively.

We conclude that the viable browser path is ONNX Runtime Web with FP16 weights, further optimized by WebGPU backend selection and multi-threaded WASM, and that code-as-weights constitutes a compiler project rather than a delivery path for this graph class. The architectural transfer of the singing fine-tune remains unverified.

Keywords: neural audio codec · quantization · WebAssembly · ONNX Runtime · Cmajor · browser inference · singing voice synthesis · realtime factor

---

Key results

Result Value
FP16 decoder wire size 105.9 MB
FP16 encoder wire size 45.6 MB
Combined payload 151.5 MB
Decoder output shape [1, 1, 10232] = 320×32 − 8
Encoder output shape [1, 32, 75] = ⌊24000/320⌋
Decoder RTF (WASM, 1 thread, 2-vCPU Xeon) 3.16
Encoder RTF (same) 1.43
Decoder Chrome RSS delta +524 MB
Encoder Chrome RSS delta +299 MB
DAC convolutions mis-lowered by Cmajor Conv 151 / 151
Cmajor name-match, decoder 227 / 347
Cmajor name-match, encoder 702 / 1574

---

1. Introduction

The standard account of browser deployment for neural audio proceeds from model size to compression algorithm and arrives at quantization as an afterthought. This ordering is incorrect. Lossless compression of neural weights is bounded near 34% by state-of-the-art methods and near 36% by generic entropy coding; neither figure bridges the gap between a 286 MB checkpoint and browser-scale payloads. The operative lever is representational — altering what a weight is rather than how it is packed — and the choice of representation is inseparable from the choice of runtime, because an operator present in ONNX may be absent from the backend, rendering a compressed graph unexecutable and therefore valueless.

This report therefore treats browser delivery as a joint optimization over four dimensions: representation, operator coverage, runtime backend, and memory residency. We measure all four on a representative convolutional codec and report the results without interpolation. Where a measurement could not be taken, we say so and identify what would be required to take it.

The scope is deliberately narrow. We do not propose a new architecture, a new quantization method, or a new runtime. We measure two existing routes against an existing artifact and report which survives contact with the constraints.

---

2. Methods

Artifact under test. The MIT descript DAC 24 kHz encoder and decoder, obtained as ONNX exports from onnx-community/dac_24khz-ONNX. A three-stage codec: convolutional encoder, residual vector quantizer with 32 codebooks at a 320-sample hop, convolutional decoder with ConvTranspose1d upsampling and Snake activations.

Representations evaluated. FP32, FP16, INT8 (as exported upstream), against a Cmajor 1.0.3209 processor set of sixteen operators (Tanh, Reshape, MatMul, Gemm, Add, Unsqueeze, Squeeze, Conv, GRU, LSTM, Slice, Concat, Gather, Pad, Transpose, Constant).

Compilation route. tools/onnxToCmajor.py from the Cmajor repository, invoked with shape inference and the ONNX checker enabled, in weight-elided mode (typed slots rather than inlined literals).

Execution route. ONNX Runtime Web 1.22.0, WASM backend, single thread, headless Chromium, on a 2-vCPU Xeon Platinum 8481C. Not a phone. Not WebGPU. Not multi-threaded.

Metrics. Wire size; node census by operator; name-match against the processor set; inference wall time; realtime factor (inference time divided by audio duration); JavaScript heap delta; Chrome resident-set delta; output shape and finiteness.

---

3. Results

3.1 Representation

FP16 halves the payload without altering the operator multiset; the only delta is one Cast per graph. INT8 is counterproductive on the decoder, where the quantized file (120.6 MB) exceeds its FP16 sibling (105.9 MB) because wrapper nodes and per-channel metadata consume more than 8-bit storage saves on a graph of this size. The encoder's INT8 file is smaller (24.5 MB) but the graph degrades from 1,575 to 2,167 nodes with four operators absent from the processor set.

3.2 Operator coverage

The decoder's 120 unmatched compute nodes collapse to two gaps: Snake (29 sites, requiring Mul, Sin, Pow) and ConvTranspose (4 sites). The encoder's 872 unmatched nodes do not mirror these; roughly 750 belong to the RVQ nearest-code search and require ArgMax, ReduceL2, ReduceSum, Shape, Expand, and ConstantOfShape. The first two categories are addition tasks; the third is a language-feature task; the fourth is an impedance mismatch between ONNX's dynamic-shape semantics and static compilation.

3.3 The Conv structural failure

Source inspection of the shipped Conv processor establishes that it sets widthPadding = 0 without reading ONNX pads, accepts dilation as a fourth argument, and exposes no stride argument. Every convolution in the codec carries non-zero padding, non-unit stride, or non-unit dilation. A rank shim satisfying the processor's static_assert therefore passes the type check and emits an incorrect tensor for all 151 convolutions. The prior characterization of this gap as "a rank shim away" is falsified.

3.4 Execution and speed

Both FP16 graphs execute correctly under ONNX Runtime Web. Output shapes confirm that strided convolution, transposed convolution, Snake activation, and the codebook search all ran. Realtime factor is 3.16 for the decoder and 1.43 for the encoder — neither faster than realtime. On mobile hardware, a further three- to five-fold degradation is expected.

3.5 Memory residency

Wire size and resident memory diverge by approximately fivefold. The decoder's 105.9 MB occupies 524 MB of Chrome resident memory; the encoder's 45.6 MB occupies 299 MB. Concurrent loading yields 823 MB, above the threshold at which iOS Safari terminates tabs.

---

4. Conclusions

Payload is solved at FP16. Operator coverage is solved by ONNX Runtime Web and unsolved by Cmajor. Runtime speed is unsolved and is the decisive unmeasured variable. Memory residency, not wire size, is the mobile risk.

The recommended browser path is ONNX Runtime Web with FP16 weights, WebGPU backend, multi-threaded WASM fallback, and QDQ INT8 for CPU-only devices. Code-as-weights is retained as a research direction, not a delivery path. The next decisive measurement is WebGPU: if it yields an order-of-magnitude improvement on the reference hardware, the path becomes realtime-capable on desktop and plausible on mobile; if not, the path is an offline renderer and realtime claims must be withdrawn.

---

Limitations

Scope. The original request specified a 6,000-page treatment. This report is a short-form communication of approximately 50 pages; the longer treatment was not produced and the discrepancy is acknowledged rather than elided. Each section is expandable, and expansion points are named in the full manuscript.

Hardware. All measurements are on a 2-vCPU server CPU. No phone was tested. The reported realtime factors are an upper bound on achievable performance, not a characterization of target devices.

Backend. Only the WASM backend, single-threaded, was measured. WebGPU, multi-threaded WASM, and QDQ INT8 are identified as levers but were not evaluated.

Artifact transfer. The singing fine-tune dac_singing_finetune_24khz.pth and its accompanying README were both unavailable. Every finding is therefore on the MIT descript DAC 24 kHz class. Whether these findings transfer to the singing fine-tune depends on whether that checkpoint altered architecture or only weights — a question the report cannot answer and which constitutes its load-bearing unverified claim.

Compilation. No Cmajor processor was added, and no shim was compiled. The cmaj binary does not start in the test environment owing to missing system libraries. Statements about what a processor addition would or would not achieve are therefore source-level readings, not compile-verified results.

Accuracy. No quality metric was evaluated. The INT8 speed expectation is unverified against perceptual or objective fidelity.

---

Data availability

Model weights are not redistributed. The MIT descript DAC 24 kHz FP16 encoder and decoder are publicly available from onnx-community/dac_24khz-ONNX on Hugging Face. The singing fine-tune dac_singing_finetune_24khz.pth is licensed CC BY-NC 4.0 (research and non-commercial use) and was unavailable at the time of writing. Soul-AILab/SoulX-Singer is Apache-2.0 and was not executed in this study.

All measurement outputs are recorded in MEASUREMENTS.md and in machine-readable form under bench/results/ against the schema in bench/schema.json.

---

Code availability

The benchmark harness is released under the MIT License and reproduces every measured figure reported here. It comprises a Playwright driver, an ONNX Runtime Web execution page, a COOP/COEP-enabled static server, a model fetch script, a QDQ INT8 quantization script, and a chunked-decoding module implementing exact-boundary reconstruction across the 8-sample transposed-convolution seam. Tests cover shape arithmetic and chunk arithmetic.

The Cmajor research note documenting the closed compilation route is included at notes/cmajor-research.md.

---

Author contributions

L.J.S. conceived the framework, designed the measurement protocol, executed the Cmajor conversion and ONNX Runtime Web benchmarks, and wrote the manuscript.

---

Competing interests

The author declares no competing financial interests. The author is the originator of the HarmoniaCodec framework and the ljrseals-dev repository namespace. The NumPy repository published under that namespace (ljrseals-dev/HarmoniaCodec) is a distinct artifact from the HarmoniaCodec cascade described in this work and was not used in any measurement. No commercial relationship exists with Cmajor, Microsoft, Hugging Face, Descript, or Soul-AILab.

---

Funding

No external funding was received. The work was conducted independently.

---

Acknowledgements

The measurements depend on openly released artifacts: the MIT descript DAC 24 kHz ONNX exports, the Cmajor toolchain, and ONNX Runtime Web. The browser-audio projects surveyed in the accompanying research digest — CHIMERA, FrankenTTS, Phonon-WASM, Kokoro, RVC-Web-Runtime, and the Magenta RealTime-2 ONNX exports — establish the feasibility envelope within which this work is situated.

---

References

1. Kumar, R., et al. Descript Audio Codec: high-fidelity neural audio compression. Descript Research (2023). Artifact: descript-audio-codec.
2. Hugging Face. onnx-community/dac_24khz-ONNX — ONNX exports of DAC 24 kHz, including FP32, FP16, and INT8 variants. Model repository.
3. Cmajor. cmajor-lang/cmajor, release 1.0.3209; tools/onnx/onnxToCmajor.py; tools/onnx/patchTemplate/onnx.cmajor.
4. Microsoft. ONNX Runtime Web 1.22.0. WASM and WebGPU execution providers.
5. Soul-AILab. SoulX-Singer — zero-shot singing voice synthesis and conversion. Apache-2.0. GitHub.
6. Seals, L.J. HarmoniaCodec: a SoulX-Singer × singing-DAC cascade. UNREAL AI Systems technical design series (2026).
7. OpenBonsai. Ternary post-training quantization pipeline: Hadamard rotation, sequential GPTQ, group scales. Reference implementation.
8. Brevis. Lossless tensor compression via synthesized reversible DSL programs. Reported ~34% reduction, 6.61 GB/s decompression, bit-exact.
9. An, J., et al. HiFTNet: a fast high-quality neural vocoder with harmonic-plus-noise source filter. arXiv (2023).
10. Lookahead Reasoning: step-level speculative decoding for accelerated generation. arXiv (2025).
11. Constrained decoding with speculative lookaheads. arXiv (2024).
12. CSSinger: chunkwise streaming singing voice synthesis. arXiv (2024).
13. jamesondh. CHIMERA Vocal Synth — physical-modeling browser voice, Rust→WASM. MIT.
14. FrankenTTS — pure-Rust browser voice cloning. Reference implementation.
15. fusor-rs. Phonon-WASM — 31M-parameter TTS in-browser on CPU. Reference implementation.
16. Hexgrad. Kokoro-82M — sub-100 MB quantized ONNX TTS. Model repository.
17. RVC-Web-Runtime — singing voice conversion under ONNX Runtime Web with ContentVec and RMVPE. Reference implementation.
18. Magenta. RealTime-2 ONNX exports — SpectroStream codec and Depthformer. Reference implementation.

---

How to cite

Seals, Jr., L. J. (2026). Realm of Quant: Cmajor Gigs to Browser — Weight Representation, Operator Coverage, and Runtime Residency in Browser-Delivered Neural Audio. Technical Report, UNREAL AI Systems. 10 November 2026.

---

Editorial note

This document is a technical report posted without peer review. It contains measured results on publicly available artifacts and source-level readings of publicly available code. It does not contain a trained model, a perceptual evaluation, or a mobile measurement. Readers should treat the reported realtime factors as hardware-specific and the reported operator coverage as artifact-specific. The load-bearing unverified claim — architectural transfer to the singing fine-tune — is stated as such in Limitations and in the full manuscript.
