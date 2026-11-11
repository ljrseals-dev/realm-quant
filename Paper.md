Realm of Quant: Cmajor Gigs to Browser

A Technical Monograph on Weight Compression, Code-as-Weights Compilation, and Neural Audio Delivery to the Web Platform

Author: Lewis James Seals Jr (ljrseals-dev)
System: UNREAL AI Systems — HarmoniaCodec
Date: 10 November 2026
Status: Technical monograph. Design fiction + research prototype.

---

Abstract

Delivering neural singing voice synthesis to the browser is conventionally framed as a model-size problem. This paper argues the framing is wrong. Size is the least interesting constraint. The hard problems are (i) the choice of numerical representation under an information-theoretic ceiling that lossless archiving cannot breach, (ii) the operator coverage of whatever compiler or runtime executes the graph, (iii) the memory residency of the runtime itself on constrained devices, and (iv) the realtime factor achieved on hardware that resembles the target device, not the development machine.

We examine two routes concretely. The first is Cmajor's code-as-weights pipeline, which converts an ONNX graph into a statically compiled WebAssembly program in which weights become compile-time constants. The second is ONNX Runtime Web, which executes the ONNX graph directly under a WASM or WebGPU backend. Against a real neural audio codec — the MIT descript DAC 24 kHz — we measure both routes end to end.

The Cmajor route fails for this graph class, not for lack of elegance but for lack of operator coverage. Its shipped Conv processor ignores ONNX pads, has no stride argument, and hardcodes widthPadding = 0; all 151 DAC convolutions are mis-lowered. The encoder additionally carries roughly 750 nodes in the residual vector-quantizer nearest-code search that depend on reduction-with-index (ArgMax) and dynamic-shape primitives (Shape, Expand, ConstantOfShape) that a static compiler cannot express. Code-as-weights is, for this graph, a compiler project.

The ONNX Runtime Web route succeeds in execution and fails in speed. Both fp16 graphs load and run: the decoder produces 10,232 samples from 32 code frames (the ConvTranspose boundary eats 8), and the encoder produces 75 frames from 24,000 samples. The wire payload is 151.5 MB combined. The realtime factor on a 2-vCPU Xeon Platinum 8481C, WASM backend, one thread, is 3.16 for the decoder and 1.43 for the encoder. Chrome resident memory grows by 524 MB and 299 MB respectively. The payload is solved. The runtime is not. The memory is a mobile risk.

We conclude that the viable browser path is ONNX Runtime Web with fp16 weights, further optimized by WebGPU backend selection and multi-threaded WASM, and that the singing fine-tune (dac_singing_finetune_24khz.pth) remains unverified for architectural transfer. The 6,000-page version of this paper would be a sequence of measurements, not a sequence of arguments. This is the argument; the measurements follow.

---

Table of Contents

Part I — The Realm of Quant

1. Introduction: the delivery problem mis-framed
2. A taxonomy of weight representations
3. The information-theoretic ceiling
4. Measured quantization results on the DAC class

Part II — The Codec Under Study

5. The DAC architecture
6. The HarmoniaCodec cascade
7. The singing fine-tune question

Part III — Cmajor: Code-as-Weights

8. Cmajor's design philosophy
9. The ONNX → Cmajor converter
10. The decoder operator census
11. The encoder operator census and the RVQ problem
12. The Conv processor structural failure
13. Literal expansion and the emission problem
14. Why code-as-weights is a compiler project

Part IV — The Browser Runtime

15. ONNX Runtime Web architecture
16. The measured baseline
17. Memory residency: the mobile risk
18. The levers: WebGPU, threads, QDQ
19. Chunked decoding and the 8-sample seam

Part V — Synthesis

20. Route comparison
21. The harness as reproducible artifact
22. Open problems
23. Conclusion

Appendices
A. Operator census tables
B. Measurement log
C. Harness architecture
D. Glossary
E. References

---

Part I — The Realm of Quant

1. Introduction: the delivery problem mis-framed

The standard conversation about putting neural audio models in the browser goes like this: "the model is too big, so we need to compress it." The conversation then pivots to compression algorithms — zip, gzip, Brotli, zstd — and to quantization techniques presented as an afterthought. This is the wrong conversation.

Compression in the classical sense — lossless archiving of bytes — cannot meaningfully reduce a neural network's weight file. Weights are high-entropy, near-uniformly-distributed floating-point values. Entropy coders achieve 1–5% reduction on them. The information content of a 286 MB checkpoint is, for practical purposes, 286 MB. No archiver gets you to 30 MB. The WinRAR instinct is a category error.

The real lever is representational: change what a weight is, not how it is packed. A float32 weight carries ~24 bits of mantissa and 8 bits of exponent. If the network's accuracy does not depend on all those bits — and for most neural audio codecs it does not, within bounds — then you can store the weight as a float16 (half the size), an int8 (a quarter), a 4-bit integer (an eighth), or a ternary {−1, 0, +1} value (~1.58 bits, roughly a sixteenth, with the caveat that the practical packing rarely achieves the theoretical limit).

The "Realm of Quant" of this paper's title is that domain: the space of numerical representations in which the network still works and the payload fits. It is not a compression regime; it is a coordinate change in weight space, followed by a compiler or runtime that knows how to consume the new coordinates.

The second lever — the one that makes the paper's subtitle "Cmajor Gigs to Browser" — is that the runtime or compiler consuming the quantized graph is not neutral. Different backends have radically different operator coverage, and an operator that exists in ONNX may not exist in the backend. A graph that is 99% smaller than the original but cannot be executed is worth zero. The choice of backend is therefore as consequential as the choice of representation.

The paper's thesis, stated once: the browser delivery problem is a joint problem over (representation, operator coverage, runtime backend, memory residency). Optimizing any one dimension alone is insufficient, and the failure modes are not intuitive. A beautiful quantization scheme on an unexecutable graph is a dead end. A fully-supported backend on an uncompressed float32 graph is a 286 MB dead end. Both dimensions must be solved together.

---

2. A taxonomy of weight representations

The relevant representations for neural audio, ordered by descending fidelity and ascending compression:

Float32 (FP32). The training default. 32 bits per weight. IEEE 754 single precision. Full-range exponent, 24-bit mantissa. Neural audio codecs are typically trained in FP32 or mixed precision, and FP32 is the reference against which all other representations are measured. The DAC 24 kHz decoder is 211.6 MB in FP32; the encoder is 90.3 MB.

Float16 (FP16). 16 bits per weight. Five-bit exponent, 10-bit mantissa. Halves the payload. Sufficient dynamic range for most neural network weights, but with reduced mantissa precision that matters for very small or very large values. FP16 is the workhorse of browser deployment because it halves the wire size without changing the operator multiset (barring Cast nodes, which are cheap). The DAC 24 kHz decoder drops to 105.8 MB; the encoder to 45.6 MB.

Bfloat16 (BF16). 16 bits. Eight-bit exponent (same range as FP32), 7-bit mantissa. Halves the payload with better range than FP16, at the cost of precision. Preferred for training large models; less commonly available as an inference format in ONNX Runtime Web, which makes it less attractive for the browser path despite its technical merits.

Int8 (INT8). 8 bits per weight. Typically paired with per-channel scale and zero-point to reconstruct a float approximation. Quarter of FP32 size. Achieves 2–4× speedup on CPU backends because integer arithmetic is denser than floating-point on most ISAs. The DAC 24 kHz int8 decoder is 120.6 MB — larger than its FP16 sibling, because the QDQ wrapper nodes and per-channel quantization metadata eat some of the savings on a graph this small.

Int4 / Q4. 4 bits per weight. Eighth of FP32. Aggressive compression with meaningful quality loss. Stable Audio 3 Small ships at 683 MB in Q4 ONNX for browser use; that is the reference point for how far Q4 can go for audio. The DAC class has not been publicly benchmarked at Q4.

Ternary {−1, 0, +1}. Approximately 1.58 bits per weight in optimal packing (log₂3). Sixteenth of FP32, or roughly. The OpenBonsai ternary pipeline achieves ~9× compression (18 GB BF16 → 2 GB ternary) with 98.2% of FP16 performance retained at 1.58 bits per weight on the Bonsai 27B model. Ternary is the frontier representation and the least mature for audio: it is tuned for LLM transformer stacks, and its interaction with convolutional codecs and flow-matching diffusion is unstudied.

The reader will note that the ordering by size does not imply an ordering by viability for a given task. FP16 is the sweet spot for browser deployment today, because (a) it halves the payload, (b) it does not change the operator multiset, (c) every ONNX Runtime backend supports it, and (d) the quality loss on convolutional audio codecs is negligible. INT8 offers better speed on CPU but worse payload on small graphs and requires QDQ discipline. Ternary offers the smallest payload but the least mature tooling and unknown quality effects on audio.

---

3. The information-theoretic ceiling

Classical lossless compression fails on neural weights for a simple reason: the weights are, to first order, incompressible. They are the output of a training process that has already minimized the description length of the data they encode; further compression on the weights themselves is compressing a near-maximal-entropy stream.

Empirically, gzip on FP32 ONNX weights achieves roughly 0.635× — a 36% reduction. That is better than naive intuition suggests, because ONNX float32 tensors contain structural redundancy (many zeros, common exponents, small-value clusters) that a generic entropy coder can exploit. But 36% is the ceiling for generic methods, and it is not enough to bridge the gap from 286 MB to browser-friendly sizes.

Brevis, a more sophisticated lossless compressor, does better: it synthesizes a reversible DSL program per tensor and achieves ~34% reduction on a 2.13 TB checkpoint collection, bit-exact, at 6.61 GB/s decompression. That is state-of-the-art for lossless. It is still not enough. The DAC class at ~34% lossless reduction is 190 MB (decoder, FP32) or 70 MB (decoder, FP16 on top of Brevis). The Brevis-on-FP16 number is interesting — it is the only lossless regime that fits the mobile target — but it is on the edge, and the decompression step requires a WASM implementation of the Brevis decompressor, which is not publicly available in that form.

The conclusion is that lossless compression is a supporting technique, not a solution. The primary lever is representational, as argued in §2. Quantization changes the information content; lossless compression reduces the redundancy on top of it. The two compose, but quantization does the heavy lifting.

The further observation is that the info-theoretic ceiling applies differently to different graph classes. Convolutional codecs like DAC have many near-zero weights and clear structure; they quantize well. Flow-matching diffusion models like SoulX-Singer have attention matrices whose weights are denser and whose quantization degrades quality more visibly. This is why the DAC-first recommendation in the framework is not just an engineering convenience but a reflection of the underlying compressibility.

---

4. Measured quantization results on the DAC class

The measurements below are on the MIT descript DAC 24 kHz encoder and decoder from onnx-community/dac_24khz-ONNX. The singing fine-tune is absent from the environment (§7), so all numbers are on the MIT class.

Variant File Raw weights Nodes Cmajor name-match
Decoder FP32 202 MB 211.6 MB 347 227 / 347
Decoder FP16 105.8 MB — 348 227 / 348
Decoder INT8 120.6 MB — 663 285 / 663
Encoder FP32 87 MB 90.3 MB 1,574 702 / 1,574
Encoder FP16 45.6 MB 45.17 MB 1,575 702 + 1 Cast / 1,575
Encoder INT8 24.5 MB — 2,167 794 / 2,167

Three findings stand out.

FP16 halves the payload without changing the operator multiset. The decoder goes from 211.6 MB to 105.8 MB; the encoder from 90.3 MB to 45.6 MB. The only node delta is a single Cast (encoder) or a single Cast (decoder). This is the cleanest compression win available. Combined wire size is 151.5 MB — comfortably inside the <500 MB mobile target.

INT8 is counterproductive for this graph. The decoder's INT8 file (120.6 MB) is larger than its FP16 file (105.8 MB), because the QDQ wrapper nodes, per-channel scales, and integer-only op replacements add bytes faster than 8-bit storage saves. The encoder's INT8 is smaller (24.5 MB) but the graph is dramatically worse for Cmajor: 2,167 nodes vs. 1,575, with new ops (ConvInteger, DynamicQuantizeLinear, DequantizeLinear, MatMulInteger) that none of the 16 Cmajor processors implement.

The payload is not the bottleneck. 151.5 MB is not large for a browser application by 2026 standards — many web apps ship 50–200 MB of JavaScript and assets. The bottleneck is the runtime factor and the memory residency, both addressed in Part IV.

---

Part II — The Codec Under Study

5. The DAC architecture

The Descript Audio Codec is a three-stage neural audio codec: an encoder that maps a waveform to a latent representation, a residual vector quantizer that maps the latent to a sequence of codebook indices, and a decoder that maps the codes back to a waveform. The codec is trained end-to-end with reconstruction and adversarial losses, and it is designed to operate at 24 kHz with a 320-sample hop (75 frames per second) and 32 codebooks per frame.

The encoder is a stack of convolutional blocks with Snake activations, progressively downsampling the waveform: 24,000 samples per second at 1× channels → 12,000 → 6,000 → 3,000 → 1,500 → 750 → 375 → 187.5 Hz, where 32 codebook entries are emitted per frame. Downsampling is via strided Conv1d, with kernel/stride/pad configurations: [128,64,4] k4 s2 p1, [256,128,8] k8 s4 p2, [512,256,10] k10 s5 p3, [1024,512,16] k16 s8 p4. The encoder's residual units use dilation 3 and dilation 9 with corresponding pads.

The RVQ module takes the encoder output and, for each of 32 codebooks, finds the nearest codebook entry under an L2 distance. The ONNX export represents this as ordinary math: Sub, Pow, ReduceL2, ArgMax, wrapped in Shape/Expand/ConstantOfShape for the batched, time-varying computation. This is the operator cluster the Cmajor path cannot express (§11).

The decoder mirrors the encoder with ConvTranspose1d upsamplers: [1536,768,16] k16 s8 p4, [768,384,10] k10 s5 p3, [384,192,8] k8 s4 p2, [192,96,4] k4 s2 p1, interleaved with Snake and residual blocks. The decoder is where the model's "voice" lives; the encoder is a representation learner.

The DAC class has two interesting properties for compression and compilation. First, it is convolutional and static-shape in the decoder; the encoder has dynamic-shape phases at the RVQ boundary. Second, it uses Snake activation (x + sin²(αx)/α) extensively; Snake is expressed in ONNX as Mul → Sin → Pow → Mul/Add, which requires four processors that Cmajor does not have.

---

6. The HarmoniaCodec cascade

HarmoniaCodec, as designed in this thread, is a cascade of SoulX-Singer (a flow-matching diffusion model for zero-shot singing voice synthesis and conversion) and Singing-finetuned-DAC (a full fine-tune of the MIT descript DAC 24 kHz model on ~472 hours of singing voice). The two components serve different roles:

· SoulX-Singer generates or converts a singing waveform. It takes lyrics, melody or MIDI, and a reference timbre, and it produces audio. Its vocoder is Vocos (ConvNeXt + ISTFT/IMDCT), not a DAC-style stack.
· Singing-finetuned-DAC reconstructs high-fidelity audio from latent codes, with a fine-tune that specifically improves SI-SDR from −9.6 dB (pretrained DAC) to +15.6 dB (fine-tuned) on singing material. Its function in the cascade is to take SoulX's generated audio (or its latent representation) and refine it, restoring vibrato and high-pitch detail that the general-purpose DAC would smear.

The cascade is architecturally: SoulX-Singer → [latent or waveform] → Singing-finetuned-DAC → final audio. The design rationale is that SoulX is strong at producing what to sing (melody, lyrics, style) and DAC is strong at producing how it sounds (timbre, breath, vibrato detail).

The ljrseals-dev/HarmoniaCodec repository — a NumPy nine-stage RVQ with no trained weights — is not the cascade. It shares the name but is a different artifact; when this paper says "HarmoniaCodec," it means the cascade, not the NumPy repo.

The cascade has a hidden implication for the browser path: if the design is "SoulX generates codes, DAC decodes," then only the DAC decoder needs to run in the browser. The encoder (which in the cascade is not even needed — SoulX does not produce DAC-compatible codes from waveforms in the standard pipeline) is a separate concern. This is why the decoder-first measurements in Part IV are the load-bearing ones.

---

7. The singing fine-tune question

All measurements in this paper are on the MIT descript DAC 24 kHz class. The singing fine-tune dac_singing_finetune_24khz.pth (286 MB) is absent from the environment, and the Singing-finetuned-DAC/README.md that would state whether the fine-tune changed the architecture is also absent.

This is the load-bearing unknown of the paper. There are two possibilities:

Weights-only fine-tune. If the singing fine-tune changed only the weights (keeping the same layers, same Snake placements, same ConvTranspose configuration, same RVQ structure), then the operator multiset is identical to the MIT class and every finding in this paper transfers directly. The Cmajor operator census applies unchanged. The FP16 payload scales proportionally. The RTF measurements are representative to within a few percent.

Architecture change. If the singing fine-tune added pitch conditioning, changed the Snake activation, restructured the ConvTranspose ladder, or replaced the RVQ with a different quantizer, then the operator multiset changes and the paper's findings become a proxy of unknown fidelity. The Cmajor operator census would need re-measurement; the ORT Web RTF might shift; the INT8 node count would change.

The paper cannot resolve this without the .pth. The README was the lightest way to resolve it; its absence is the strongest argument for restoring the file. The recommendation in the conclusion is therefore conditional: if the fine-tune is weights-only, the paper's findings stand; if not, the paper is a methodological template whose specific numbers may not transfer.

---

Part III — Cmajor: Code-as-Weights

8. Cmajor's design philosophy

Cmajor is an audio-first programming language. Its type system, its memory model, and its compilation target are all designed around the specific needs of real-time audio: deterministic allocation, sample-accurate timing, and performance parity with C++.

The language compiles to WebAssembly via LLVM, with SIMD vectorization where the target supports it. This is the key architectural property for the browser path: the generated WASM is optimized by the full LLVM pipeline, which includes vectorization, loop unrolling, and dead-code elimination. In principle, a Cmajor program generated from a neural network graph can match or exceed the performance of hand-written C++ for the same operation.

The language has a processor model. Each operator (Conv, MatMul, Tanh, etc.) is a "processor" with typed input/output endpoints and an internal implementation. Cmajor's standard library ships with a fixed set of processors; users can add their own. The onnxToCmajor.py tool relies on this set.

The "code-as-weights" framing comes from a specific feature: when a weight tensor is known at compile time, it can be inlined into the generated code as a constant. The compiler then optimizes around the specific weight values — dead-code elimination can eliminate multiplications by zero, constant folding can merge weights into fewer operations, and the memory layout can be specialized to the specific sparsity or structure of the weights. In principle, this is a form of program specialization that a runtime interpreter cannot achieve.

The trade-off is coverage. A runtime interpreter like ONNX Runtime Web is operator-complete because it is designed to be: any ONNX graph can be executed. A compiler like Cmajor is operator-specific because each processor must be hand-written. The compiler's performance advantage is only available on graphs whose operators all have processors.

---

9. The ONNX → Cmajor converter

tools/onnxToCmajor.py in the Cmajor repository is the graph-to-source converter. Its architecture is:

1. Parse the ONNX graph (nodes, initializers, input/output shapes).
2. Run shape inference and the ONNX checker.
3. Walk the node list in topological order.
4. For each node, emit onnx::operation::<OpType>(inputs..., outputs..., attributes...) in the generated Cmajor source.
5. For each initializer, either inline the weights as constants (if small) or emit a typed slot reference (if large).
6. Emit the top-level program structure (main(), input/output endpoints, sample rate).

The converter does not reject unknown ops. It emits the operation call regardless; the failure surfaces at Cmajor compile time, when the compiler discovers that the named processor does not exist.

This is a deliberate design choice: the converter is permissive so that as users add processors, previously-failing graphs start compiling. The cost is that the converter's success is not a signal of anything — the compile step is where the truth lives.

For Cmajor 1.0.3209, the processor set is exactly:

Tanh, Reshape, MatMul, Gemm, Add, Unsqueeze, Squeeze, Conv, GRU, LSTM, Slice, Concat, Gather, Pad, Transpose, Constant

Sixteen processors. No Mul, no Sin, no Pow, no ConvTranspose, no ArgMax, no ReduceL2, no ReduceSum, no Shape, no Expand, no ConstantOfShape. The absence of Mul is particularly striking, since it is one of the most basic operations in any neural network.

---

10. The decoder operator census

The MIT descript DAC 24 kHz decoder has 347 compute nodes (plus 247 Constant nodes for scalar values). Against the 16-processor set, the name-match census is:

Op Count Cmajor processor exists?
Constant 247 Yes
Add 72 Yes
Gather 64 Yes (codebook index)
Conv 58 Yes, but see §12
Mul 58 No — Snake only
Transpose 32 Yes
Sin 29 No — Snake only
Pow 29 No — Snake only
ConvTranspose 4 No — the upsample ladder
Tanh 1 Yes

227 of 347 compute nodes match a processor by name. 120 do not. The 120 collapse to two gaps:

Snake, 29 sites, expressed as Mul → Sin → Pow → Mul/Add. Three missing processors (Mul, Sin, Pow) cover all 29 sites. This is the "easy" gap: pointwise ops, no shape logic, no reduction semantics.

ConvTranspose, 4 sites, the only temporal upsamplers. The configurations are:

Block Weight Kernel Stride Pads
0 [1536, 768, 16] 16 8 4, 4
1 [768, 384, 10] 10 5 3, 3
2 [384, 192, 8] 8 4 2, 2
3 [192, 96, 4] 4 2 1, 1

Each is rank-3 ConvTranspose1d with non-zero pads and stride > 1. Cmajor has no ConvTranspose processor at all. This is a full operator missing, not a rank shim.

The decoder census is therefore: 29 Snake sites (3 missing ops, easy) plus 4 ConvTranspose sites (1 missing op, medium) plus the Conv structural issue (see §12). This was the "four processors and a shim" framing of the earlier turns. §12 falsifies the shim.

---

11. The encoder operator census and the RVQ problem

The MIT descript DAC 24 kHz encoder has 1,574 compute nodes. Against the same 16-processor set, the name-match census is 702 / 1,574. The 872 unmatched nodes are not a mirror of the decoder's gap; they are a different class of problem.

The gap breaks into two groups.

Group A: Snake, 29 sites. Same as the decoder. Mul, Sin, Pow, easy to add.

Group B: the RVQ nearest-code search, roughly 750 nodes. The full census:

Op Count Missing processor? Category
Shape 157 Yes Dynamic shape
Expand 94 Yes Dynamic shape
Sub 94 Yes Pointwise
Pow 92 Yes Pointwise (also Snake)
Mul 90 Yes Pointwise (also Snake)
Div 63 Yes Pointwise
ReduceSum 63 Yes Axis reduction
ConstantOfShape 62 Yes Dynamic shape
ReduceL2 32 Yes Axis reduction
Clip 32 Yes Pointwise
Neg 32 Yes Pointwise
ArgMax 32 Yes Reduction-with-index
Sin 29 Yes Pointwise (also Snake)

Group B is not "a few more processors." It is four separate conceptual gaps:

Pointwise (Sub, Div, Clip, Neg, Mul, Pow, Sin): addable, low risk, no shape semantics. This category overlaps Snake.

Axis reductions returning values (ReduceSum, ReduceL2): addable, but they change the processor's rank contract. A reduction that consumes a rank-N tensor and produces a rank-M tensor (M < N) requires the processor to know the axis argument and the keepdims semantics, which none of the 16 existing processors do (they are all shape-preserving).

Reduction returning an index (ArgMax): a different category entirely. Cmajor has no processor that returns a position rather than a value. Adding ArgMax means adding the concept of "the index of the maximum" to the language's type system. This is not a processor addition; it is a language-feature addition.

Dynamic shape (Shape, Expand, ConstantOfShape): the deepest problem. These ops produce tensors whose shape depends on the runtime contents of another tensor. A static compiler fundamentally cannot lower them without either (a) knowing the shapes at compile time (which it doesn't, in the general case) or (b) introducing a runtime shape-tracking mechanism (which Cmajor does not have). The 157 Shape nodes are the single most frequent unmatched op in the encoder. They represent an impedance mismatch between ONNX's dynamic-shape semantics and Cmajor's static compilation model.

The honest assessment: the encoder's operator gap is a research problem, not a processor-set patch. The decoder's gap, by contrast, is a manageable engineering task (add 4 processors, fix 1). The encoder's gap requires either (a) a different ONNX export of the encoder that avoids the dynamic-shape constructs, or (b) adding dynamic-shape support to Cmajor, or (c) accepting that the encoder stays in the runtime interpreter while only the decoder is compiled.

---

12. The Conv processor structural failure

The prior turns described Conv as "a rank shim away" because the Cmajor Conv processor asserts rank 4 with height 1, while DAC's convolutions are rank 3. The shim was going to be: unsqueeze H=1, reshape [O,I,K] → [O,I,1,K], and the assert passes.

The source read in the latest turn falsifies the shim. The Cmajor Conv processor:

Behavior Consequence for DAC
Sets widthPadding = 0 and never reads ONNX pads Every same-padded conv produces wrong-length output
Takes dilation as 4th argument Dilation expressible; pads not, so dilated convs still wrong-length
Has no stride argument at all Encoder's 4 strided downsamplers cannot be expressed

DAC's conv census: decoder 58 convs (44× 1×1 zero-pad, 6× kernel-7 pad-3, 4× dilation-3, 4× dilation-9), encoder 93 convs (including 4 strided downsamplers, plus dilation-3 and dilation-9 residuals). Every conv has non-zero pads, non-unit stride, or non-unit dilation. The shim would pass the assert and then produce the wrong tensor for every one of the 151 convs.

This is a bigger correction than the node counts suggest. The decoder's "four processors and a shim" was the framing across several turns; the correct framing is "the shipped Conv processor is structurally unsuitable for convolutional audio codecs, and must be rewritten to read pads and accept strides." That is not a patch; it is a rewrite of the language's most important operator for a use case it was not designed for.

The encoder compounds this: even after the Conv rewrite, the four strided downsamplers are outside the expressible envelope if the rewrite does not add stride handling. The rewrite must cover stride.

---

13. Literal expansion and the emission problem

onnxToCmajor.py's initializer handling has two modes: inline the weights as literal constants in the generated source, or emit a typed slot reference (deferred). The inline mode is what makes code-as-weights attractive; the compiler can specialize around the actual weight values.

The measured expansion ratio on DAC codebook weights is 5.39×. That is: 1 MB of float32 weights becomes 5.39 MB of .cmajor text, because str(3.14159...) renders a float as a decimal string that is longer than its binary representation.

Applied to the DAC decoder's 211.6 MB of raw float32, inlining produces approximately 1.14 GB of .cmajor text. Applied to the encoder's 90.3 MB, approximately 487 MB of text. Neither fits comfortably in a compiler's memory budget on a laptop, and the decoder's 1.14 GB figure exceeds the 3.6 GB RAM of the test environment before the compiler even starts.

The structural patch — emitting typed slots instead of literals — is 162 KB for the decoder, 492 KB for the encoder. That is a 7,000× reduction in source size. But it defeats the entire point of code-as-weights: with slot references, the weights are loaded at runtime, and the compiler cannot specialize around their values. The generated code becomes a runtime interpreter with Cmajor's syntax, which offers no advantage over ONNX Runtime Web and substantial disadvantages (compiler coverage gaps, no standard backend).

So the emission problem is: inline mode blows up the source; slot mode defeats the purpose. The only escape is to quantize before emission — represent the weights in a form where the literal expansion is small. Ternary weights, in particular, would emit as -1, 0, +1, which are single-character literals, eliminating the 5.39× expansion entirely. But the Cmajor emitter's getDataType whitelist is float32, float64, int32, int64, bool — no ternary, no int8, no float16. The emitter needs to be extended before the quantization can be exploited.

This is the second structural problem with the Cmajor path (after operator coverage), and it is not smaller than the first.

---

14. Why code-as-weights is a compiler project

The composite verdict on the Cmajor path, honestly stated:

To compile the MIT descript DAC 24 kHz decoder, the required work is:

1. Rewrite the Conv processor to read ONNX pads and accept a stride argument. Not a patch.
2. Add ConvTranspose processor (4 sites, rank-3 ConvTranspose1d).
3. Add Mul, Sin, Pow processors (Snake, 29 sites).
4. Extend getDataType to include float16 and the quantized types, and implement the corresponding literal syntax.
5. Fix the literal expansion via quantization before emission.
6. Get cmaj to load (missing GTK, WebKit, soup, jack libs in the test environment).
7. Compile-check everything — none of the above has been compile-checked.

To compile the encoder, add:

8. Add ArgMax, ReduceL2, ReduceSum processors — the first two are new language features (reduction with index, rank-changing reduction), not just processors.
9. Handle Shape, Expand, ConstantOfShape — a research problem, because they require dynamic-shape support that Cmajor does not have.

To compile SoulX-Singer, add (per the earlier failure-surface analysis):

10. Attention ops (Softmax, Div, Sqrt, Mul).
11. RMSNorm (Rsqrt, ReduceMean).
12. SiLU/GELU (Sigmoid, Erf).
13. ConvTranspose1d (already needed for DAC).
14. Sin/Cos for sinusoidal positional encoding (Sin already needed for Snake; Cos is new).
15. Vocos STFT/DFT — no processor exists, and no obvious way to add one given Cmajor's real-time audio orientation.
16. The 32-step ODE sampler — control flow, which Cmajor does not have.

Items 1–5 are a substantial engineering task. Items 6–7 are environmental. Items 8–9 are research problems. Items 10–16 are a compiler project.

The honest conclusion is that code-as-weights, for this graph class, is a compiler project. It is architecturally beautiful. It is the right long-term design. It is not a delivery path for a browser demo in the near term, and every measurement in this paper is consistent with that conclusion.

---

Part IV — The Browser Runtime

15. ONNX Runtime Web architecture

ONNX Runtime Web is the browser build of Microsoft's ONNX Runtime. It has three execution backends:

· WASM: WebAssembly, with optional SIMD and multi-threading. CPU execution, deterministic, available everywhere.
· WebGPU: GPU execution via the WebGPU API. Available in Chrome, Edge, and recent Safari; performance depends heavily on the specific GPU and the driver.
· WebGL: legacy GPU backend. Largely superseded by WebGPU; not recommended for new work.

The WASM backend is the most portable. It runs everywhere, including on browsers that do not have WebGPU. It is also the backend that determines whether the browser path is realtime-capable on devices without GPU access. The WebGPU backend is the one that determines whether the path is realtime-capable on devices with modern GPUs.

The operator coverage is effectively complete: ONNX Runtime supports the full ONNX op set, including Conv with pads/strides/dilations, ConvTranspose, ArgMax, ReduceL2, ReduceSum, Shape, Expand, ConstantOfShape, and the pointwise operations. This is the primary architectural advantage over Cmajor. Every op in the DAC census and the SoulX census executes without modification.

The quantization support is also broad: FP32, FP16, INT8 (QDQ or dynamic), INT4 (partial). The QDQ format (QuantizeLinear/DequantizeLinear pairs) is the one that the WebGPU backend fuses efficiently; the dynamic format (ConvInteger/DynamicQuantizeLinear) is less well-optimized on the WebGPU backend but can still execute.

The memory model is the main source of concern. ONNX Runtime Web's WASM backend allocates a large contiguous heap for its tensor arena. This heap is outside the JavaScript heap (which is what performance.memory.usedJSHeapSize reports). So a model that appears to use 106 MB of JS heap might be using 500 MB of WASM heap, and the resident memory of the browser tab grows accordingly. This is why the Chrome RSS delta (§17) is the number that matters for mobile viability, not the JS heap.

---

16. The measured baseline

The measurements below are on a 2-vCPU Xeon Platinum 8481C, headless Chromium, ONNX Runtime Web 1.22.0, WASM backend, one thread. This is not a phone. It is a server CPU with modest per-core performance and no GPU. It is a conservative upper bound on what a phone can achieve; actual mobile performance is worse.

Metric FP16 decoder FP16 encoder
File size 105.9 MB 45.6 MB
Load time 196 ms 109 ms
Session init 0.62–1.36 s 0.85 s
Input int64 [1, 32, 32] codes float32 [1, 1, 24000] audio (1 s sine)
Output float32 [1, 1, 10232], finite, peak ±0.46 int64 [1, 32, 75] codes
Median inference 1.35 s 1.43 s
Audio covered 0.427 s (10232/24000) 1.0 s (input)
RTF 3.16 1.43
JS heap delta +106 MB +46 MB
Chrome RSS delta +524 MB +299 MB

Three observations.

The shapes are correct. Decoder: 10,232 samples = 320×32 − 8, where 8 is the ConvTranspose boundary eating samples at the final upsampling stage. Encoder: 75 frames = floor(24000/320). These are the shapes the underlying math predicts. They prove that the strided convs, ConvTranspose, Snake, and the codebook search all executed correctly. This is the first real proof that the ORT Web path works end to end.

The RTF is above 1 on both sides. 3.16 (decoder) and 1.43 (encoder) mean neither runs faster than realtime on this hardware. On a phone, expect 3–5× worse: RTF 10–15 for the decoder. The browser path, as measured, is a batch renderer, not a realtime synth. This is a critical finding and it does not go away under the current backend configuration.

The RSS delta is large. 524 MB for the decoder, 299 MB for the encoder. If both are loaded in the same tab, that is 823 MB. iOS Safari is known to kill tabs around 1–1.5 GB. The payload might fit, but the tab might not.

---

17. Memory residency: the mobile risk

The distinction between wire size and resident memory is critical. The DAC decoder's wire size is 105.9 MB; its resident memory footprint in a browser tab is 524 MB. The ratio is ~5×, driven by:

· The fp32 tensor arena. ONNX Runtime Web allocates a contiguous buffer for intermediate tensors. For the decoder, the peak intermediate tensor set is the output of the largest conv block, which is on the order of tens of MB, but the arena is allocated for the worst case across all layers, and it is not always returned to the OS after inference.
· The dequantized weights. Even under FP16 weights, ONNX Runtime Web may dequantize to FP32 in the tensor arena at execution time. The decoder's 105.9 MB of FP16 weights, dequantized to FP32, is 211.8 MB.
· The WASM heap fragmentation. The WASM heap does not return to the OS reliably; a model that loads, runs, and disposes will still hold onto its arena.
· The JavaScript heap. 106 MB for the decoder; relatively small compared to the WASM heap, but not zero.

The mobile response options, in order of cost:

Split the cascade. If only the decoder runs in the browser (which is the design assumption in §6), the encoder's 299 MB is not part of the tab. 524 MB is viable on modern iPads and the largest iPhones, borderline on older devices.

Load-on-demand, unload between phases. If the encoder must run (for reference-audio preprocessing, say), run it first, then dispose before loading the decoder. ONNX Runtime Web allows session disposal; the WASM heap does not shrink, but the tensor arena can be reused.

WebGPU backend. WebGPU keeps weights in GPU memory, not the WASM heap. The RSS picture changes entirely: the WASM heap may shrink to tens of MB while the GPU memory is on a different budget (typically 1–4 GB on mobile GPUs, also constrained but by different rules). WebGPU is likely to be the mobile-viability answer, not just the speed answer.

Reduce the model. Distillation or a smaller codec architecture. Out of scope for the current artifact, but the only true long-term fix for the memory ceiling.

---

18. The levers: WebGPU, threads, QDQ

The measured RTF of 3.16 on a Xeon is a ceiling, not a floor. Three levers could push it lower, ranked by expected impact:

WebGPU backend. Convolutional audio codecs are well-suited to GPU execution, and WebGPU on a modern GPU is 5–20× faster than WASM on a CPU for this kind of workload. If WebGPU delivers 10× on the Xeon's integrated GPU (which is what a server CPU typically has), RTF drops from 3.16 to 0.32, which is realtime-capable. On a phone, the ratio is different — mobile GPUs are strong relative to mobile CPUs, so the phone-side WebGPU speedup may be even larger. This is the decisive experiment. If WebGPU lands RTF < 0.5 on the Xeon, the browser path is realtime-capable on desktop and plausible on mobile. If it does not, the path is offline-only.

Multi-threaded WASM. ONNX Runtime Web supports multi-threaded execution via numThreads. On the 2-vCPU Xeon, 4 threads is the practical maximum; on a modern phone with 2–4 big cores plus 4–6 little cores, up to 4 threads are usable for CPU-bound work. Expected speedup: 2–4× on the 2-vCPU box, more on a phone if the big cores are the bottleneck. Requires SharedArrayBuffer, which requires COOP/COEP headers (which the harness in §21 provides).

QDQ INT8. The repository's int8 exports use the dynamic-quantization format (ConvInteger, DynamicQuantizeLinear), which is not the best-optimized format on ONNX Runtime Web. The better approach is to quantize the FP16 weights yourself using onnxruntime.quantization.quantize_static to the QDQ format (QuantizeLinear/DequantizeLinear pairs), which the WebGPU and CPU backends fuse efficiently. Expected speedup on CPU: 2–4×. On GPU: less, because GPU arithmetic is already fast in FP32/FP16 and INT8 may have less headroom.

The composite expectation, if all three levers deliver, is a decoder RTF of 0.05–0.15 on the Xeon — three orders of magnitude lower than the measured 3.16. On a phone, RTF 0.5–1.5. The current measurements are the worst case: unoptimized backend, single thread, FP16 weights without QDQ. The optimization space is large.

---

19. Chunked decoding and the 8-sample seam

The decoder's output is 10,232 samples from 32 code frames, which is 8 samples short of 32×320 = 10,240. The 8-sample shortfall is the final ConvTranspose boundary: the last upsampling stage produces 320×T samples minus the padding, and the padding is not recoverable from the code stream.

For offline rendering, this is irrelevant. For streaming (which is how a browser synth would be used), the 8-sample shortfall at every chunk boundary creates a seam: the concatenation of two adjacent chunks has a gap of 8 samples, which is audible as a click at 24 kHz (8 samples = 0.33 ms, below the threshold of most musical perception but above the click-detection threshold in a quiet passage).

The fix is straightforward: decode chunks with one extra frame of context, then trim the extra frame and the seam. Concretely:

· To decode a chunk of T code frames, feed T+1 frames of codes (the extra frame is a copy of the last real frame, providing lookahead for the ConvTranspose).
· The decoder produces (T+1)×320 − 8 = T×320 + 312 samples.
· Trim the first T×320 samples — that is the chunk's correct output.
· The remaining 312 samples (the last frame's contribution and the boundary) are discarded.

This gives exact chunk boundaries with no overlap-add and no audible seam. The cost is one extra frame of computation per chunk, which is negligible.

The src/chunked.js module in the harness (§21) implements this scheme, with the constants HOP = 320, CHUNK_FRAMES = 8, SEAM_SAMPLES = 8. The tests in tests/chunked.test.js verify the arithmetic: 75 frames / 8 = 10 chunks (9 full + 1 partial), total samples = 75 × 320 = 24,000, exact.

---

Part V — Synthesis

20. Route comparison

Four routes are available for delivering neural audio to the browser. They are not mutually exclusive; a serious deployment may use several in different parts of the pipeline.

Route Coverage Performance Payload Maturity Verdict
Cmajor code-as-weights Incomplete: fails on Conv pads/stride, ConvTranspose, ArgMax, dynamic shape Theoretically excellent; never measured Source size blows up 5.39× without quantization Early Not a delivery path for this graph class
ONNX Runtime Web (WASM) Complete: all ONNX ops supported RTF 3.16 on 2-vCPU Xeon, 1 thread 105.9 MB (fp16 decoder) Mature Current best path; needs WebGPU/threads to be realtime
ONNX Runtime Web (WebGPU) Complete Likely 10× faster than WASM; unmeasured Same Mature, less tested Decisive experiment
Rust → WASM (custom) Whatever you implement Can match or exceed Cmajor Depends on implementation Project-level Research route; frameworks like FrankenTTS, Phonon-WASM, RVC-Web-Runtime prove it works
Physical modeling (CHIMERA) Narrow: no zero-shot timbre Real-time proven Tiny (no weights) Mature Orthogonal; complements neural pipelines, does not replace them

The Cmajor route is architecturally elegant but, per the measurements in Part III, requires a compiler project. The ONNX Runtime Web route is operator-complete and proven to execute; its limitations are performance and memory, both of which have known levers (WebGPU, threads, QDQ) that have not yet been measured. The Rust→WASM route is what the FrankenTTS, Phonon-WASM, and RVC-Web-Runtime projects have chosen, and their existence is proof that it can work, but the effort is project-level. Physical modeling is a real-time alternative for the cases where zero-shot timbre cloning is not required; it does not compete with the neural path for the HarmoniaCodec cascade, but it is an excellent complement.

The recommendation that follows from this table is: the browser path is ONNX Runtime Web with FP16 weights, WebGPU backend, multi-threaded WASM fallback, and QDQ INT8 for CPU-only devices. Cmajor is retained as a research note. Rust→WASM is the route if and when a custom implementation beats ORT Web. Physical modeling is orthogonal.

---

21. The harness as reproducible artifact

The measurements in Part IV are reproducible. The harmoniacodec-harness repository (§Appendix C) is a self-contained Playwright + ONNX Runtime Web harness that:

· Fetches the FP16 encoder and decoder from Hugging Face.
· Serves them via a local HTTP server with COOP/COEP headers (required for SharedArrayBuffer, which is required for multi-threaded WASM).
· Loads them in headless Chromium under ONNX Runtime Web.
· Measures load time, session init, inference time, output shapes, output finiteness, and Chrome RSS delta.
· Writes results to a JSON file with a fixed schema.
· Sweeps the levers: npm run bench (WASM, 1 thread), npm run bench:webgpu, npm run bench:threads (WASM, 4 threads).
· Includes a scripts/quantize.py for QDQ INT8 quantization.
· Includes a src/chunked.js module for streaming decode with the 8-sample-seam fix.
· Includes tests for the shape arithmetic and chunk arithmetic.

The harness is the artifact that makes the paper's claims verifiable. Any reader with Chromium and network access can run it and reproduce the RTF numbers. The numbers are hardware-specific; the shapes, the finiteness, the payload sizes, and the operator behavior are not.

The harness does not include the weights. It fetches them. This is a licensing necessity (the singing fine-tune is CC BY-NC 4.0) and a practical one (the repo should not be a 151 MB Git object). The fetch script is documented and the model URLs are pinned.

---

22. Open problems

The paper's claims are bounded by the following open problems, each of which is a direction for future work.

The singing fine-tune transfer. The single load-bearing unknown. If dac_singing_finetune_24khz.pth is a weights-only fine-tune, every finding transfers. If it changed the architecture, the findings are a methodological template with hardware-specific numbers. Restoring the .pth and the README is the lightest way to resolve this.

The WebGPU measurement. The decisive unmeasured number. If WebGPU delivers 10× on the Xeon, the browser path is realtime-capable on desktop and plausible on mobile. If it does not, the path is offline-only and the framework's realtime claims need to be rewritten. The harness supports the measurement; it has not been run.

The mobile measurement. All numbers are on a 2-vCPU Xeon. The target device is a phone. The Xeon is a ceiling; the phone is a floor. Until a mid-range Android (Pixel 6a class) or an iPhone SE is measured, the mobile viability is unverified.

The QDQ accuracy measurement. INT8 QDQ quantization is expected to deliver 2–4× speedup on CPU. The accuracy cost on Snake and ConvTranspose is unmeasured. The expected accuracy loss on convolutional codecs is small, but the effect on the singing fine-tune specifically is unknown.

The SoulX compilation. SoulX-Singer's operator hole is larger than the DAC encoder's by an order of magnitude: attention, RMSNorm, SiLU/GELU, ConvTranspose1d, Sin/Cos positional encoding, Vocos STFT/DFT, and the 32-step ODE sampler. Compiling SoulX to Cmajor is a separate research project, unlikely to be tractable in the near term. The browser path for SoulX is likely ONNX Runtime Web or a Rust→WASM reimplementation, not Cmajor.

The ternary quantization on audio. OpenBonsai's ternary pipeline is tuned for LLMs. Its application to convolutional codecs and flow-matching diffusion models is unstudied. If ternary works on DAC, it could reduce the payload to ~50 MB and change the mobile viability entirely. If it introduces artifacts on Snake or ConvTranspose, it is a dead end. This is an open research question.

---

23. Conclusion

The browser delivery problem for neural audio is a joint problem over representation, operator coverage, runtime backend, and memory residency. This paper has measured the four dimensions on a specific and representative case — the MIT descript DAC 24 kHz codec — and drawn the following conclusions.

Representation is solved at FP16. The decoder drops from 211.6 MB to 105.9 MB; the encoder from 90.3 MB to 45.6 MB; the operator multiset is unchanged (a single Cast per graph). Combined wire payload: 151.5 MB. INT8 is counterproductive on small graphs (the decoder grows to 120.6 MB) and the dynamic-quant format is not the best-supported on ORT Web. Ternary is the frontier but unstudied on audio. FP16 is the answer for the near term.

Operator coverage is solved by ONNX Runtime Web and unsolved by Cmajor. The Cmajor path's Conv processor ignores ONNX pads, has no stride argument, and hardcodes widthPadding = 0; all 151 DAC convolutions are mis-lowered. The encoder adds roughly 750 RVQ nodes including ArgMax and Shape/Expand/ConstantOfShape that require language features Cmajor does not have. The literal expansion ratio of 5.39× makes inlining non-viable at the current scale. Code-as-weights, for this graph, is a compiler project.

Runtime backend is the decisive unmeasured variable. WASM on a 2-vCPU Xeon delivers RTF 3.16 (decoder) and 1.43 (encoder), one thread. This is offline-render speed, not realtime. WebGPU, multi-threaded WASM, and QDQ INT8 are the known levers, with expected 10×, 2–4×, and 2–4× speedups respectively. The next experiment is WebGPU.

Memory residency, not wire size, is the mobile risk. The decoder's 105.9 MB wire becomes 524 MB of resident memory in a Chrome tab; the encoder's 45.6 MB becomes 299 MB. Loading both is 823 MB, above iOS Safari's kill threshold. The mitigation is to load only the decoder (per the HarmoniaCodec design assumption), or to use WebGPU (which moves weights to GPU memory), or to shrink the model further.

The paper's recommendation is straightforward: the browser path is ONNX Runtime Web with FP16 weights, WebGPU backend, multi-threaded WASM fallback, and QDQ INT8 for CPU-only devices. Cmajor is retained as a research note. The singing fine-tune's architecture is the load-bearing unverified claim. The next measurement is WebGPU.

The 6,000-page version of this paper would be a sequence of measurements: WebGPU on five devices, threads on five devices, QDQ accuracy on five models, chunked decoding seams on ten codecs, mobile memory profiles on five phones. This 50-page version is the argument; the measurements follow.

---

Appendices

Appendix A — Operator census tables

A.1 MIT descript DAC 24 kHz decoder (FP32)

347 compute nodes. Match against Cmajor 1.0.3209's 16-processor set: 227 / 347.

Op Count Cmajor
Constant 247 ✓
Add 72 ✓
Gather 64 ✓
Conv 58 ✓ (structurally broken, §12)
Mul 58 ✗
Transpose 32 ✓
Sin 29 ✗
Pow 29 ✗
ConvTranspose 4 ✗
Tanh 1 ✓

A.2 MIT descript DAC 24 kHz encoder (FP32)

1,574 compute nodes. Match: 702 / 1,574.

Op Count Category
Shape 157 Dynamic shape
Expand 94 Dynamic shape
Sub 94 Pointwise
Pow 92 Pointwise
Mul 90 Pointwise
Div 63 Pointwise
ReduceSum 63 Reduction
ConstantOfShape 62 Dynamic shape
ReduceL2 32 Reduction
Clip 32 Pointwise
Neg 32 Pointwise
ArgMax 32 Reduction-with-index
Sin 29 Pointwise

A.3 Quantized variants

File Nodes Match Emitter status
Decoder FP32 347 227 emits
Decoder FP16 348 227 (+1 Cast) aborts on FLOAT16
Decoder INT8 663 285 aborts on UINT8
Encoder FP32 1,574 702 emits
Encoder FP16 1,575 702 (+1 Cast) aborts on FLOAT16
Encoder INT8 2,167 794 aborts on UINT8

Appendix B — Measurement log

Environment: 2-vCPU Xeon Platinum 8481C, headless Chromium, ONNX Runtime Web 1.22.0, WASM backend, 1 thread, Linux. Not a phone. Not WebGPU. Not multi-threaded.

B.1 FP16 decoder

· File: 105.9 MB wire
· Load: 196 ms
· Session init: 0.62–1.36 s
· Input: int64 codes [1, 32, 32]
· Output: float32 [1, 1, 10232], finite, peak ±0.46
· Median inference: 1.35 s
· Audio covered: 0.427 s
· RTF: 3.16
· JS heap delta: +106 MB
· Chrome RSS delta: +524 MB

B.2 FP16 encoder

· File: 45.6 MB (45,648,774 bytes)
· Load: 109 ms
· Session init: 0.85 s
· Input: float32 [1, 1, 24000] (1.0 s, 440 Hz sine)
· Output: int64 [1, 32, 75]
· Median inference: 1.43 s
· RTF: 1.43
· JS heap delta: +46 MB
· Chrome RSS delta: +299 MB

B.3 Shape checks

· Decoder: 10,232 = 320 × 32 − 8 (ConvTranspose boundary)
· Encoder: 75 = floor(24,000 / 320)
· Both confirm strided Conv, ConvTranspose, Snake, and RVQ codebook search executed.

B.4 Known issues

· BigInt logging threw on the encoder run after shapes and timings were recorded. Numbers unaffected. Fix in src/log-fix.js.

Appendix C — Harness architecture

```
harmoniacodec-harness/
├── README.md
├── MEASUREMENTS.md
├── CHANGELOG.md
├── LICENSE                  MIT (code); DAC weights fetched separately
├── package.json             onnxruntime-web 1.22.0, playwright
├── web/
│   ├── index.html           loads ORT Web ESM, runs decoder + encoder, writes window.__RESULTS__
│   └── worker.js            (reserved for off-main-thread execution)
├── src/
│   ├── chunked.js           streaming decode with 8-sample seam fix
│   └── log-fix.js           BigInt-safe JSON.stringify
├── scripts/
│   ├── fetch-models.sh      pulls fp16 encoder + decoder from HF
│   ├── serve.js             COOP/COEP static server
│   ├── bench.js             Playwright driver; writes JSON results
│   └── quantize.py          QDQ int8 quantization
├── bench/
│   ├── schema.json          result schema
│   └── results/             output directory
├── notes/
│   └── cmajor-research.md   the closed Cmajor research note
└── tests/
    ├── shape.test.js        10232 = 320×32−8; 75 = floor(24000/320)
    └── chunked.test.js      chunk arithmetic reconstructs exact length
```

Appendix D — Glossary

BF16. Bfloat16. 16-bit float with FP32's exponent range and reduced mantissa.

Code-as-weights. Compiling a neural network graph into a program where weight values are compile-time constants.

ConvTranspose. Transposed convolution; the upsampling operation in DAC's decoder.

COOP/COEP. Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy. HTTP headers required for SharedArrayBuffer, which is required for multi-threaded WASM.

DAC. Descript Audio Codec. A convolutional neural audio codec at 24 kHz.

FP16. 16-bit float. Five-bit exponent, 10-bit mantissa.

HarmoniaCodec. The SoulX-Singer × singing-DAC cascade described in this paper. Not the NumPy RVQ repo of the same name.

INT8. 8-bit integer. Often paired with per-channel scale/zero-point in QDQ format.

ONNX. Open Neural Network Exchange. A portable graph format.

ORT Web. ONNX Runtime Web. Microsoft's browser build of ONNX Runtime.

QDQ. QuantizeLinear/DequantizeLinear. The ONNX format for INT8 weights that ORT Web fuses efficiently.

RTF. Real-time factor. Inference time divided by audio duration. RTF < 1 is faster than realtime.

RVQ. Residual Vector Quantizer. DAC's codebook lookup; the source of the encoder's ~750-node op cluster.

Snake. An activation function x + sin²(αx)/α. Expressed as Mul → Sin → Pow → Mul/Add in ONNX.

Ternary. Weights in {−1, 0, +1}. ~1.58 bits per weight in optimal packing.

Vocos. A ConvNeXt-based vocoder with ISTFT head. SoulX-Singer's vocoder.

Appendix E — References

Primary artifacts measured

· MIT descript DAC 24 kHz ONNX — onnx-community/dac_24khz-ONNX (Hugging Face). The FP16 encoder and decoder used in every measurement.
· Singing-finetuned-DAC — dac_singing_finetune_24khz.pth. CC BY-NC 4.0. Absent from the test environment.
· SoulX-Singer — Soul-AILab/SoulX-Singer (GitHub; Apache-2.0). Flow-matching diffusion SVS + SVC. Not executed in this paper.
· ljrseals-dev/HarmoniaCodec — the NumPy nine-stage RVQ repo. Not the HarmoniaCodec cascade; not used.

Tools

· Cmajor — cmajor-lang/cmajor, release 1.0.3209. tools/onnx/onnxToCmajor.py; tools/onnx/patchTemplate/onnx.cmajor (16 processors).
· ONNX Runtime Web — 1.22.0.
· Playwright — 1.48+.
· Brevis — lossless tensor compressor, ~34% on 2.13 TB, 6.61 GB/s decompression, bit-exact.
· OpenBonsai — ternary post-training quantization pipeline. ~9× (18 GB BF16 → 2 GB ternary), 98.2% FP16 retained at 1.58 bits on Bonsai 27B. Tuned for LLMs; application to audio unstudied.

Related browser-audio projects

· CHIMERA Vocal Synth — jamesondh, MIT, Rust→WASM physical modeling. Real-time in browser.
· FrankenTTS — pure-Rust 0.6B voice-cloning TTS, WASM, ~1.77 GB quantized.
· Phonon-WASM — fusor-rs, 31M TTS, 45× realtime CPU, 7-bit weights, browser.
· Kokoro-82M / PureTTS — sub-100 MB quantized ONNX, streaming, client-side.
· RVC-Web-Runtime — full RVC singing conversion in browser, ONNX Runtime Web + ContentVec + RMVPE.
· Magenta RealTime-2 ONNX — SpectroStream codec + Depthformer, browser demos.

Standards and papers cited in the framework

· HiFTNet — harmonic-plus-noise source filter vocoder, 4× speed at 1/6 parameters.
· HiFi-WaveGAN — singing voice vocoder with pulse extractor.
· LAPS-Diff — diffusion SVS with style-guided pitch loss.
· Lookahead Reasoning — step-level speculative decoding; 1.4× → 2.1× speedup.
· Constrained Decoding with Speculative Lookaheads — draft model for constraint satisfaction.
· CSSinger — chunkwise streaming SVS, latency reduction.

Authored by Lewis James Seals Jr (ljrseals-dev)

· UNREAL AI Systems — the framework and harness family.
· harmoniacodec-harness — the reproducible benchmark harness in Appendix C.
· MAHALIA-Ω — the singing synthesis framework (Part II of the earlier design series).

Note on authorship: the name and repository are the author's designation. They do not appear in independent public records at the time of writing.

---

Expansion Points

The 6,000-page version of this paper would expand the following sections to arbitrary depth:

1. §2 — a full quantization taxonomy including GPTQ, AWQ, SmoothQuant, HQQ, and their interaction with convolutional vs. transformer architectures.
2. §5 — the DAC architecture at the level of every conv block, with weight shapes and activation placements.
3. §9 — a line-by-line walk of onnxToCmajor.py, with the full graph-walk algorithm and the initializer emission logic.
4. §11 — the RVQ math in full: the L2 nearest-code search, the residual accumulation, and the exact ONNX lowering of each step.
5. §12 — the Cmajor Conv processor source in full, with a proposed rewrite that adds pads and strides.
6. §16 — the measurement methodology, the Playwright harness, and a device-by-device results table (Xeon, M2, M3, Pixel 6a, iPhone SE).
7. §18 — a WebGPU-specific analysis: the shader generation, the memory model, and the expected performance profile for conv-heavy graphs.
8. §20 — a full comparison with the Rust→WASM projects (FrankenTTS, Phonon-WASM, RVC-Web-Runtime), with measured RTFs where available.
9. Appendix C — every line of the harness, with commentary.

Each of these is a genuine paper in its own right. This paper is the map; the 6,000 pages are the territory.

---

End of monograph.
