# Introduction to LLM Engineering — Book Skeleton (v2)
### Inference, serving and systems for large language models, for FAANG-level interviews

> **Reader.** PhD in mathematics, ~5 years of data science, little software or systems engineering.
> **Promise.** Derive the cost of serving any LLM on any hardware; explain every major inference technique of 2017–2026 and why it appeared when it did; design a production serving system under latency and cost targets; pass ML-systems and inference-engineer interviews.
> **How it is written.** See `CLAUDE.md` (style), `NOTATION.md` (symbols) and `REVIEW.md` (editorial log).

---

## The story the book tells

The book follows one causal chain. Each part answers the question the previous part raised.

1. **Part I — What are we running, and why is it expensive?** Language modelling became a Transformer problem. A Transformer keeps a growing memory of everything it has read (the KV cache), and producing text means running the whole model once per token. *Open question: what machine are we running this on, and what limits it?*
2. **Part II — The machine.** GPUs can do arithmetic far faster than they can fetch data. The roofline model turns that into a quantitative tool, and kernels are how we exploit it. *Open question: the KV cache dominates memory traffic. How do we shrink it and read it efficiently?*
3. **Part III — Attention and memory.** Shrink the cache (MQA → GQA → MLA, quantisation, eviction), read it with IO-aware kernels (FlashAttention), and manage it like virtual memory (PagedAttention, prefix caching). *Open question: many users share one GPU. How do we schedule them?*
4. **Part IV — Serving systems.** Continuous batching, chunked prefill, inference engines, and prefill/decode disaggregation. *Open question: can we make the model itself cheaper to run?*
5. **Part V — Smaller models.** Quantisation, sparsity and distillation. *Open question: the model still does not fit on, or is not fast enough on, one GPU.*
6. **Part VI — Distributed inference.** Collectives, then tensor, pipeline, context and expert parallelism, then MoE. *Open question: decoding is still one token at a time, and contexts keep growing.*
7. **Part VII — Faster and longer.** Speculative decoding, long-context methods, and attention alternatives. *Open question: how do we run all of this in production, for reasoning models, at scale, and talk about it in an interview?*
8. **Part VIII — Production.** Reasoning and test-time compute, fleet-level serving, benchmarking, edge inference, and system-design interviews.

**Running examples, used throughout.** Llama 3 8B (dense, GQA), Llama 3 70B (large dense), DeepSeek-V3 (671B MoE, 37B active, MLA), served on an H100 SXM, with A100, H200 and B200 for comparison.

**Per-chapter components.** Motivating opening → numbered sections (motivation → idea → math → example → consequence → bridge) → Looking ahead → summary → 30+ MCQs → 15+ interview questions (Warm-up / Core / Deep-dive / Design) → references. Target 9–14k words (18–28 pages). Total ≥ 26 × 18 ≈ 470 pages at completion; ≥ 200 pages is reached by Chapter 11.

---

## Part I — Foundations: What We Run and Why It Is Expensive

### Chapter 1 — From Shannon to Serving ✅ *(v2)*
**Motivating question.** Why did *running* language models, rather than training them, become one of the hardest and most expensive problems in computing?
**Arc.** Language as probability → counting (n-grams, sparsity) → learning representations (NPLM, word2vec) → remembering (RNN, LSTM, attention) → the Transformer's bargain (parallel training in exchange for a growing state) → scale and scaling laws → inference-aware scaling → ChatGPT and inference economics → hardware: compute outruns memory → first estimate: why writing is slow → the systems response 2019–2026 as a causal chain → reasoning models → how we measure (TTFT, TPOT, goodput, Little's law, cost) → map of the book.
**Diagrams/charts.** RNN vs Transformer state; parameters over time; tokens per parameter; training vs serving compute; memory wall; request timeline; book map.
**Key references.** Shannon 1948/51; Kneser & Ney 1995; Bengio 2003; Mikolov 2013; Hochreiter & Schmidhuber 1997; Sutskever 2014; Bahdanau 2014; Vaswani 2017; Brown 2020; Kaplan 2020; Hoffmann 2022; Sardana 2023; Gholami 2024; Snell 2024; DeepSeek-AI 2025.

### Chapter 2 — The Transformer, Dissected ✅ *(v2)*
**Motivating question.** To reason about cost we need an exact inventory: which tensors exist, what shapes they have, and how many parameters and FLOPs they contribute.
**Arc.** The shape of the computation → tokenisation (why sub-words; BPE worked by hand; vocabulary trade-offs) → embedding and unembedding → attention as a differentiable dictionary lookup (√d_h derivation; causal mask; heads; MQA/GQA as a preview) → positions (equivariance proof; sinusoidal → learned → ALiBi → RoPE with proof) → residual stream and normalisation (Post-LN → Pre-LN; RMSNorm) → FFN (key–value memories; ReLU → SwiGLU) → architecture evolution 2017–2025 → counting parameters → counting FLOPs (2N, attention term, 6N) → counting bytes.
**Hands off to Chapter 3:** "We know what is computed; what does it cost to *generate*?"
**Key references.** Vaswani 2017; Sennrich 2016; Kudo 2018; Su 2021; Press 2021; Ba 2016; Zhang & Sennrich 2019; Xiong 2020; Shazeer 2019/2020; Geva 2021; Ainslie 2023; Llama 3 2024.

### Chapter 3 — Running the Model: Prefill, Decode and the KV Cache ✅ *(v2, new)*
**Motivating question.** Generating text calls the model once per token. Why is that slow, what must we remember between calls, and what limits the speed?
**Arc.** Generation as a loop and its naive \(NT^2\) cost → the KV cache (proof of exactness) → prefill vs decode → how big the cache is → where time goes: compute, memory, overhead → arithmetic intensity and the roofline → linear layers vs decode attention → a latency model for decode → a latency model for prefill (TTFT) → the life of one request, fully costed → code: a KV-cached decoder → what production engines change.
**Hands off to Chapter 4:** "The model gives logits. How do we turn them into text?"
**Key references.** Pope 2022; Williams 2009; Kaplan 2020; He 2022 (*Go Brrrr*); kipply 2022; Austin 2025; Kwon 2023.

### Chapter 4 — Decoding: From Logits to Text ✅ *(v2)*
**Motivating question.** The model outputs a distribution. Which token should we emit, and how do we do that fast, controllably and reproducibly?
**Arc.** Search vs sampling (greedy, beam, why beam search fails in open-ended generation) → temperature as a Gibbs distribution → truncation (top-k, top-p, min-p, typical) → penalties and logit processors → structured outputs (grammars → automata; XGrammar; jump-forward) → GPU sampling kernels → determinism and batch invariance → stopping, streaming and incremental detokenisation.
**Refs.** Holtzman 2019; Meister 2023; Nguyen 2024 (min-p); Willard & Louf 2023; Dong 2024 (XGrammar); He & Thinking Machines 2025 (nondeterminism).

---

## Part II — The Machine: Hardware for Mathematicians

### Chapter 5 — GPU Architecture from First Principles ✅ *(v2)*
**Motivating question.** Chapter 3 showed that decode is limited by memory bandwidth. What is inside a GPU that makes arithmetic cheap and data movement expensive?
**Arc.** Latency vs throughput machines → SIMT (grids, blocks, warps; divergence) → the streaming multiprocessor → the memory hierarchy and its bandwidth/latency pyramid → tensor cores, Volta → Rubin (FP16 → FP8 → FP4) → HBM → NVLink/NVSwitch and rack-scale systems → TPUs, MI300X/MI355X, Trainium, Groq, Cerebras → reading a spec sheet.
**Refs.** NVIDIA whitepapers; Jouppi 2017/2023; Kirk & Hwu (PMPP); Luo 2024 (dissecting Hopper).

### Chapter 6 — The Roofline Model and Performance Analysis ✅ *(v2)*
**Motivating question.** How do we predict the speed of any kernel or model, and how do we know how far we are from the limit?
**Arc.** Roofline as \(\min(\pi, I\beta)\) → hierarchical rooflines → operators of an LLM on the roofline → critical batch size, including KV traffic → MFU and MBU → Little's law and the latency–throughput frontier → profiling (Nsight, PyTorch profiler).
**Refs.** Williams 2009; Pope 2022; Austin 2025; Yuan 2024.

### Chapter 7 — GPU Programming for Mathematicians
**Motivating question.** Why does fusing operations or changing a loop order make code 10× faster, and how is a fast kernel written?
**Arc.** Kernels as maps over index spaces → coalescing, shared memory, occupancy → tiled GEMM → reductions and online softmax → fusion → Triton → CUDA Graphs and launch overhead → torch.compile → CuTe-DSL, ThunderKittens, TileLang.
**Refs.** CUDA Guide; Tillet 2019; Spector 2024; Ansel 2024; Milakov 2018.

---

## Part III — Attention and Memory

### Chapter 8 — The KV Cache: Shrinking It
**Motivating question.** The cache limits batch size and dominates decode traffic. How small can we make it without hurting quality?
**Arc.** MQA → GQA → MLA (derivation and weight absorption) → sliding-window and local/global hybrids → cross-layer sharing → KV quantisation → eviction (attention sinks, H2O, SnapKV) and why reasoning models break it.
**Refs.** Shazeer 2019; Ainslie 2023; DeepSeek-V2 2024; Xiao 2023; Zhang 2023; Liu 2024 (KIVI); Brandon 2024.

### Chapter 9 — IO-Aware Attention: FlashAttention 1 → 4
**Motivating question.** Attention with a \(T\times T\) matrix moves \(\Theta(T^2)\) bytes. Can we compute exactly the same function while moving far less?
**Arc.** IO complexity → online softmax derivation → FA1 tiling → FA2 → FA3 (Hopper asynchrony, FP8) → FA4 (Blackwell) → decode kernels (FlashDecoding, FlashInfer) → MLA and sparse kernels.
**Refs.** Dao 2022/2023; Shah 2024; FA4 2026; Rabe & Staats 2021; Ye 2025.

### Chapter 10 — Memory Management: PagedAttention and Prefix Caching
**Motivating question.** Sequences grow unpredictably. How do we allocate KV memory without wasting most of it, and reuse work across requests?
**Arc.** Fragmentation → paging and block tables → copy-on-write → prefix caching (hash chains vs radix trees) → eviction policy → hierarchical KV storage (CPU, SSD, remote: LMCache, Mooncake Store) → vAttention.
**Refs.** Kwon 2023; Zheng 2024; Prabhu 2024; Qin 2025.

---

## Part IV — Serving Systems

### Chapter 11 — Batching and Scheduling
**Motivating question.** Requests arrive at random and finish at different times. How do we keep the GPU busy without making anyone wait too long?
**Arc.** Static and dynamic batching → continuous batching (Orca) → chunked prefill (Sarathi) → preemption → fairness, priorities and SLO-aware scheduling → length prediction.
**Refs.** Yu 2022; Agrawal 2024; Sheng 2024 (VTC); Fu 2024.

### Chapter 12 — Inside an Inference Engine
**Motivating question.** How do all these ideas fit together into software such as vLLM or SGLang?
**Arc.** Anatomy of an engine → vLLM V0 → V1 → SGLang → TensorRT-LLM, TGI, llama.cpp → APIs and streaming.
**Refs.** Gordić 2025 (*Inside vLLM*); vLLM and SGLang docs; Park 2025 (engine survey).

### Chapter 13 — Disaggregated Serving
**Motivating question.** Prefill and decode want different hardware and interfere with each other. Should they run on different machines?
**Arc.** Interference and goodput → Splitwise, DistServe → Mooncake → KV transfer (RDMA, NIXL) → Dynamo, llm-d → attention/FFN disaggregation.
**Refs.** Patel 2024; Zhong 2024; Qin 2025.

---

## Part V — Making Models Smaller

### Chapter 14 — Quantisation
**Motivating question.** Decode time is proportional to bytes read. How few bits per weight and per cache entry can we use?
**Arc.** Number formats (FP8, INT4, MXFP4, NVFP4) → quantisation as lattice rounding, error analysis, outliers → LLM.int8, SmoothQuant → GPTQ, AWQ → rotations (QuaRot, SpinQuant) → FP8/FP4 in practice → what actually speeds up.
**Refs.** Dettmers 2022; Xiao 2022; Frantar 2022; Lin 2023; Ashkboos 2024; Kurtic 2024; OCP MX 2023.

### Chapter 15 — Pruning, Sparsity and Distillation
**Refs.** Frantar & Alistarh 2023; Sun 2023; Muralidharan 2024; Hinton 2015; Agarwal 2024; Liu 2023.

---

## Part VI — Distributed Inference

### Chapter 16 — Distributed Computing Fundamentals
**Motivating question.** The model or its cache does not fit on one GPU. What does it cost to move data between GPUs?
**Arc.** Collectives as linear-algebra operations → the α–β cost model → ring and tree all-reduce (proof of bandwidth optimality) → NVLink, InfiniBand, RoCE, topologies → NCCL, NVSHMEM, overlap.
**Refs.** Thakur 2005; Patarasuk & Yuan 2009; NCCL docs.

### Chapter 17 — Parallelism Strategies for Inference
**Arc.** Data parallelism → tensor parallelism (Megatron) → pipeline parallelism → sequence and context parallelism (Ring, Ulysses) → choosing a configuration.
**Refs.** Shoeybi 2019; Narayanan 2021; Pope 2022; Liu 2023; Jacobs 2023.

### Chapter 18 — Mixture of Experts at Inference
**Arc.** From Shazeer 2017 to DeepSeek-V3 and Kimi K2 → routing and balancing → expert parallelism (all-to-all, DeepEP, EPLB) → MoE arithmetic intensity → the DeepSeek-V3 deployment case study.
**Refs.** Shazeer 2017; Fedus 2021; Jiang 2024; DeepSeek-AI 2024/2025.

---

## Part VII — Faster Decoding and Longer Contexts

### Chapter 19 — Speculative Decoding
**Arc.** Exactness via rejection sampling (proof) → expected speed-up → drafts (small models, n-grams, Medusa, EAGLE 1–3, MTP) → tree verification → speculation under batching.
**Refs.** Leviathan 2023; Chen 2023; Cai 2024; Li 2024/2025.

### Chapter 20 — Long-Context Inference
**Arc.** Position extrapolation (PI, NTK, YaRN) → sparse attention (NSA, DSA) → context parallelism and offload → retrieval vs long context.
**Refs.** Chen 2023; Peng 2023; Yuan 2025.

### Chapter 21 — Beyond Attention: SSMs, Linear Attention and Hybrids
**Arc.** Linear attention as an RNN → S4 → Mamba → Mamba-2 → RWKV, Gated DeltaNet → hybrids and their serving implications.
**Refs.** Katharopoulos 2020; Gu & Dao 2023; Dao & Gu 2024; Lieber 2024.

---

## Part VIII — Production

### Chapter 22 — Reasoning Models and Test-Time Compute
### Chapter 23 — Serving at Scale: Routing, Autoscaling, Multi-LoRA and Cost
### Chapter 24 — Benchmarking, Observability and Evaluation
### Chapter 25 — On-Device and Edge Inference
### Chapter 26 — The ML-Systems Design Interview: Worked Case Studies
(These chapters keep the content listed in v1 of this skeleton; their detailed arcs will be written when Part VI is complete.)

## Appendices
A. Numerical precision for mathematicians · B. Back-of-the-envelope cheat-sheet · C. PyTorch you need · D. Linux, containers, Kubernetes · E. Glossary · F. Notation (site page `notation.html`) · G. Bibliography

---

## Writing status
| Ch | Title | Status |
|---|---|---|
| 1 | From Shannon to Serving | ✅ v2 (rewritten after Review 1) |
| 2 | The Transformer, Dissected | ✅ v2 (rewritten, split) |
| 3 | Running the Model: Prefill, Decode and the KV Cache | ✅ v2 (new) |
| 4 | Decoding: From Logits to Text | ✅ v2 |
| 5 | GPU Architecture from First Principles | ✅ v2 |
| 6 | The Roofline Model and Performance Analysis | ✅ v2 |
| 7–26 | — | ⏳ planned |
