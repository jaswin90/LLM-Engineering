# Introduction to LLM Engineering
### Cracking FAANG-Level Interviews in LLM Inference, Serving and Systems

> **Audience.** A reader with a PhD in mathematics and ~5 years of data-science experience, comfortable with
> linear algebra, probability, optimisation and Python/NumPy, but with little formal software- or
> systems-engineering background (no CUDA, no networking, no OS internals assumed).
>
> **Promise.** By the end of the book the reader can (1) derive, from first principles, the cost of running any
> modern LLM on any accelerator; (2) explain every major inference optimisation of 2017–2026 and *why it was
> invented when it was*; (3) whiteboard a production serving system under latency/cost SLOs; and (4) answer the
> conceptual, numerical and design questions asked in ML-systems / inference-engineer loops at FAANG and AI labs.
>
> **Organising principle: evolution.** Every chapter is told as a sequence of *problems → ideas → new
> bottlenecks*. We never present a technique without the bottleneck that motivated it and the bottleneck it
> created in turn. A recurring "timeline" box anchors each idea in time.

---

## 0. How the book is built

| Element | Description |
|---|---|
| **Theory** | Rigorous but conversational. Definitions, derivations, small proofs where they add insight (e.g. why speculative decoding is *exactly* distribution-preserving). |
| **Equations** | Rendered with KaTeX. Every cost formula is boxed and later re-used in back-of-the-envelope exercises. |
| **Diagrams** | Hand-built inline SVG (theme-aware, no images). |
| **Charts** | Pure HTML/SVG charts rendered from data in the page (no image files). |
| **"Software-engineering bridge" boxes** | Short asides that translate a systems concept into mathematical language (e.g. a cache as a memoised function, a GPU warp as SIMD over a 32-element vector). |
| **Timeline boxes** | Year-stamped milestones; cumulatively they form the history of the field. |
| **Worked examples** | Numerical, with real model configs (Llama, Mixtral, DeepSeek, Qwen) and real hardware (A100, H100, H200, B200, MI300X, TPU). |
| **MCQs** | 25–40 per chapter, interactive, with explanations; progress saved (localStorage, and Google sign-in + Firestore sync). |
| **Interview Q&A** | 15–25 per chapter, graded *Warm-up / Core / Deep-dive / Design*; model answers hidden behind a reveal; self-rating saved. |
| **Back-of-the-envelope drills** | Short numeric problems with hidden solutions. |
| **References** | Primary sources (papers, whitepapers, docs) at the end of every chapter + a consolidated bibliography. |

**Length target.** ≥ 200 printed pages (≈ 100–120k words, 25 chapters + appendices). Chapters 1–2 ≈ 35–40 pages.

---

## Part I — Foundations: How We Got Here

### Chapter 1 — From Shannon to Serving: A History of Language Modelling and the Birth of Inference Engineering ✅ *(written)*
**Learning objectives.** Frame language modelling as conditional probability estimation; trace the n-gram → neural → recurrent → attention → Transformer → scaling → reasoning arc; understand why the *cost centre* of AI shifted from training to inference; learn the vocabulary (TTFT, TPOT, goodput, cost/M tokens) and the first bandwidth-bound back-of-the-envelope.

1.1 Why an inference book, why now · how to use this book
1.2 Language modelling as probability: chain rule, cross-entropy, perplexity, Shannon's entropy of English
1.3 The count-based era (1948–2003): Markov assumption, smoothing, Kneser–Ney, the curse of $|V|^n$
1.4 The neural turn (2003–2013): Bengio's NPLM, distributed representations, word2vec
1.5 The recurrent era (2010–2017): RNNs, vanishing gradients, LSTM, seq2seq, Bahdanau attention — *constant-size state*
1.6 The Transformer (2017): parallel training, quadratic attention, *growing state* (the KV cache) — the trade-off that defines this book
1.7 Pre-training and scale (2018–2020): GPT-1/2/3, BERT, T5, in-context learning
1.8 Scaling laws: Kaplan, Chinchilla, and *inference-aware* scaling (why Llama-3-8B saw 15T tokens)
1.9 The ChatGPT moment and inference economics: lifetime compute $6ND_{\text{train}} + 2N D_{\text{infer}}$, price collapse
1.10 The systems explosion (2022–2026): FasterTransformer, Orca, FlashAttention, vLLM, TGI, TensorRT-LLM, SGLang, speculative decoding, MoE, disaggregation
1.11 The reasoning era (2024–): o1, DeepSeek-R1, test-time compute; long outputs make decode dominant
1.12 Hardware co-evolution and the *memory wall* (P100 → V100 → A100 → H100 → B200 → Rubin)
1.13 The metrics vocabulary: TTFT, TPOT/ITL, E2E latency, throughput, goodput, percentiles, cost per million tokens
1.14 First back-of-the-envelope: why a 70B model at batch 1 is capped at a few tens of tokens/s on any single GPU
1.15 A map of the book
**Diagrams.** Evolution timeline; RNN state vs Transformer KV-cache state; serving request lifecycle with TTFT/TPOT annotations; book map.
**Charts.** Model parameters over time (log); tokens-per-parameter over time; FLOPs vs bandwidth growth (memory wall).
**MCQ themes.** Perplexity ↔ cross-entropy; n-gram sparsity; vanishing gradients; RNN vs Transformer inference state; Chinchilla ratio; inference-aware scaling; metric definitions; bandwidth-bound estimates.
**Interview questions.** "Why are LLMs memory-bandwidth bound at decode?" · "Define TTFT and TPOT; which does a chatbot / batch summariser care about?" · "Why train a small model far past Chinchilla-optimal?" · "Estimate max tokens/s for model X on GPU Y."
**Key references.** Shannon 1948/1951; Kneser & Ney 1995; Bengio et al. 2003; Mikolov et al. 2013; Hochreiter & Schmidhuber 1997; Sutskever et al. 2014; Bahdanau et al. 2014; Vaswani et al. 2017; Radford et al. 2018/2019; Brown et al. 2020; Kaplan et al. 2020; Hoffmann et al. 2022; Sardana & Frankle 2023; Pope et al. 2022; Gholami et al. 2024; Snell et al. 2024; DeepSeek-AI 2025.

### Chapter 2 — The Transformer, Dissected for Inference ✅ *(written)*
**Learning objectives.** Know every tensor in a decoder-only LLM, its shape, its parameter count and its FLOPs; derive the $2N$ FLOPs/token rule; understand the architectural evolution 2017→2025 (Post-LN→Pre-LN, LayerNorm→RMSNorm, ReLU→SwiGLU, sinusoidal→RoPE, MHA→MQA→GQA→MLA); distinguish prefill from decode; compute KV-cache size and arithmetic intensity; build a 60-line KV-cached decoder.

2.1 Notation and the decoder-only block diagram
2.2 Tokenisation: BPE, byte-level BPE, Unigram/SentencePiece; vocabulary-size trade-offs; incremental detokenisation
2.3 Embeddings, the LM head, weight tying
2.4 Scaled dot-product attention: derivation, the $\sqrt{d_k}$ variance argument, causal masking, multi-head as block-diagonal structure
2.5 Positional information: sinusoidal, learned, ALiBi, RoPE (with proof of the relative-position property)
2.6 Normalisation: LayerNorm vs RMSNorm; Post-LN vs Pre-LN and training stability
2.7 Feed-forward networks: ReLU → GELU → SwiGLU; the $\tfrac{8}{3}d$ width rule
2.8 Parameter counting (worked: Llama-2-7B, Llama-3-8B, Llama-2-70B)
2.9 FLOP counting: the $2mkn$ rule, $2N$ per token, the attention term, $6N$ for training
2.10 Autoregressive generation: prefill vs decode; naive recomputation vs caching; KV-cache size formula
2.11 Arithmetic intensity and the first roofline: why decode is memory-bound and prefill is compute-bound
2.12 A latency model for decode; memory budget and maximum concurrency
2.13 From logits to tokens (preview of Ch. 3)
2.14 Code: a minimal KV-cached decoder in PyTorch, annotated for mathematicians
**Diagrams.** Decoder-only stack; attention dataflow with shapes; multi-head split; RoPE rotation in the complex plane; Pre-LN vs Post-LN; SwiGLU; prefill vs decode timeline; KV cache growth.
**Charts.** Roofline for H100 with prefill/decode points; KV cache vs context length (MHA vs GQA); decode step time vs batch size.
**MCQs.** Shapes; parameter counts; FLOPs; √d scaling; RoPE properties; RMSNorm; SwiGLU width; KV size; arithmetic intensity; ridge points.
**Interview questions.** "Count the parameters of Llama-3-8B." · "Why divide by √d_k?" · "Derive the KV cache size." · "Why is decode memory-bound?" · "What does GQA buy you?" · "Implement attention with a KV cache."
**Key references.** Vaswani 2017; Sennrich 2016; Kudo & Richardson 2018; Su et al. 2021 (RoPE); Press et al. 2021 (ALiBi); Ba 2016; Zhang & Sennrich 2019; Xiong et al. 2020; Shazeer 2019 (MQA), 2020 (GLU); Ainslie et al. 2023 (GQA); Touvron et al. 2023; Kaplan 2020; Pope 2022; Williams et al. 2009 (roofline).

### Chapter 3 — Decoding: From Logits to Text
3.1 Search vs sampling: greedy, beam search (and why beam died for chat), the "degeneration" problem
3.2 Temperature as inverse-temperature of a Gibbs distribution; entropy vs temperature
3.3 Truncation samplers: top-k, nucleus/top-p, min-p, typical sampling, η-sampling
3.4 Repetition/frequency/presence penalties; logit bias; logprobs
3.5 Structured & constrained decoding: grammars, regex/JSON-schema via finite automata (Outlines, XGrammar), jump-forward decoding
3.6 Batch sampling on GPU: fused top-k/top-p kernels, sorting-free sampling (FlashInfer)
3.7 Determinism and reproducibility: why batch-invariance breaks "temperature 0"
3.8 Stopping criteria, streaming, incremental detokenisation, tool-call parsing
**Interview Qs.** Implement top-p; why can temperature 0 be non-deterministic?; cost of constrained decoding; beam search vs sampling.
**Refs.** Holtzman et al. 2019; Meister et al. 2023; Nguyen et al. 2024 (min-p); Willard & Louf 2023; Dong et al. 2024 (XGrammar); Thinking Machines "Defeating nondeterminism" 2025.

---

## Part II — The Machine: Hardware for Mathematicians

### Chapter 4 — GPU Architecture from First Principles
4.1 Why GPUs: latency-oriented CPUs vs throughput-oriented GPUs; Amdahl and Little's law
4.2 The SIMT model: grids, blocks, warps, threads; divergence
4.3 The streaming multiprocessor: CUDA cores, tensor cores, SFUs, register file, warp schedulers
4.4 The memory hierarchy: registers → shared memory/L1 → L2 → HBM → host DRAM → NVMe; bandwidth/latency pyramid
4.5 Tensor cores and their evolution: Volta (FP16) → Ampere (TF32, BF16, 2:4 sparsity) → Hopper (FP8, TMA, wgmma, thread-block clusters) → Blackwell (FP4/FP6, TMEM, 2-CTA MMA, dual-die) → Rubin (HBM4)
4.6 HBM: stacks, channels, why bandwidth is expensive
4.7 Interconnect inside the box: PCIe, NVLink, NVSwitch, NVL72 racks
4.8 Other accelerators: TPU (systolic arrays, ICI torus), AMD MI300X/MI355X (chiplets, Infinity Fabric), AWS Trainium/Inferentia, Groq LPU, Cerebras WSE, Apple silicon
4.9 Reading a spec sheet critically: dense vs sparse FLOPs, boost clocks, TDP
**Diagrams.** SM block diagram; memory pyramid; tensor-core MMA tile; NVLink topology; systolic array.
**Interview Qs.** What is a warp; why does shared memory matter; compare H100 vs MI300X for a 70B model; why does Blackwell need FP4?
**Refs.** NVIDIA Volta/Ampere/Hopper/Blackwell whitepapers; Jouppi et al. 2017, 2023 (TPU v4); Kirk & Hwu *PMPP*; Jia et al. 2018 (dissecting Volta); Luo et al. 2024 (dissecting Hopper).

### Chapter 5 — The Roofline Model and Performance Analysis
5.1 Work, traffic, arithmetic intensity; the roofline as a min of two linear functions
5.2 Ridge points for A100/H100/B200/MI300X/TPU; hierarchical rooflines (L2, SMEM)
5.3 Rooflines of LLM operators: GEMM, GEMV, attention prefill/decode, norms, elementwise
5.4 Batch size as the dial: the critical batch size $B^*\approx$ FLOPs/BW; with KV traffic
5.5 Latency vs throughput Pareto frontier; Little's law for serving
5.6 Profiling in practice: Nsight Systems/Compute, PyTorch profiler, MFU and MBU
**Interview Qs.** Compute MBU from a benchmark; at what batch does Llama-70B become compute-bound on H100?
**Refs.** Williams, Waterman & Patterson 2009; Pope et al. 2022; Austin et al. 2025 (*How to Scale Your Model*); Yuan et al. 2024 (LLM Inference Unveiled).

### Chapter 6 — GPU Programming for Mathematicians: CUDA, Triton and Kernels
6.1 A kernel as a map over an index space; host/device; asynchronous streams
6.2 Coalescing, bank conflicts, occupancy, latency hiding
6.3 Tiled GEMM from scratch; register blocking; CUTLASS/CuTe abstractions
6.4 Reductions and the online softmax (Milakov & Gimelshein)
6.5 Kernel fusion; why elementwise ops are "free" only when fused
6.6 Triton: block-level programming; writing fused RMSNorm and softmax
6.7 CUDA Graphs, launch overhead and torch.compile
6.8 Newer DSLs: CuTe-DSL, ThunderKittens, TileLang
**Interview Qs.** Write a softmax kernel; why fuse; what does CUDA Graph capture solve for decode?
**Refs.** NVIDIA CUDA C++ Guide; Tillet et al. 2019 (Triton); Spector et al. 2024 (ThunderKittens); Ansel et al. 2024 (PyTorch 2).

---

## Part III — Attention and Memory

### Chapter 7 — The KV Cache: Theory, Size and Shrinking It
7.1 Caching as memoisation of $K=XW_K$, $V=XW_V$; correctness proof under causal masking
7.2 Size formula $2\,L\,n_{kv}\,d_h\,b\,T\,B$; KV vs weights cross-over point
7.3 Multi-Query (2019) → Grouped-Query (2023) → Multi-head Latent Attention (2024): derivations and trade-offs; MLA weight absorption
7.4 Sliding-window, local/global interleaving (Mistral, Gemma 2/3), cross-layer KV sharing (CLA, YOCO)
7.5 KV-cache quantisation (FP8, INT4, KIVI) and its error model
7.6 KV eviction & compression: StreamingLLM attention sinks, H2O, SnapKV, PyramidKV; reasoning-model caveats
**Interview Qs.** Size the KV cache for DeepSeek-V3 vs Llama-3-70B at 128k; derive MLA's absorbed form.
**Refs.** Shazeer 2019; Ainslie 2023; DeepSeek-AI 2024 (V2); Xiao 2023; Zhang 2023 (H2O); Liu 2024 (KIVI); Brandon 2024 (CLA).

### Chapter 8 — IO-Aware Attention: FlashAttention 1 → 4 and FlashDecoding
8.1 Standard attention's HBM traffic $\Theta(T^2)$; the IO-complexity lower bound
8.2 Online softmax and tiling: FlashAttention (2022) with full derivation
8.3 FA2 (2023): parallelism over sequence, fewer non-matmul FLOPs
8.4 FA3 (2024): Hopper asynchrony, warp specialisation, ping-pong, FP8
8.5 FA4 (2026): Blackwell, exp emulation, TMEM, 2-CTA
8.6 Decode attention: FlashDecoding / split-K, FlashInfer, ragged/paged layouts
8.7 Attention variants: MLA kernels (FlashMLA), sparse attention kernels
**Interview Qs.** Derive the online-softmax recurrence; why does FA not reduce FLOPs yet run faster?
**Refs.** Dao et al. 2022, Dao 2023, Shah et al. 2024, Zadouri/Dao et al. 2026; Rabe & Staats 2021; Ye et al. 2025 (FlashInfer).

### Chapter 9 — Memory Management: PagedAttention, Prefix Caching and KV Hierarchies
9.1 Fragmentation (internal/external/reservation) in early servers
9.2 PagedAttention: block tables, virtual memory analogy, copy-on-write for beam/parallel sampling
9.3 Prefix caching: hash-chained blocks (vLLM) vs radix trees (SGLang RadixAttention); eviction policy
9.4 Hierarchical KV storage: GPU ↔ CPU ↔ SSD ↔ remote (LMCache, Mooncake Store, CachedAttention)
9.5 vAttention and other alternatives
**Interview Qs.** Design a prefix cache for a multi-tenant chatbot; why block size 16?
**Refs.** Kwon et al. 2023; Zheng et al. 2024 (SGLang); Prabhu et al. 2024 (vAttention); Qin et al. 2025 (Mooncake); Gao et al. 2024.

---

## Part IV — Serving Systems

### Chapter 10 — Batching and Scheduling
10.1 Static and dynamic batching (Triton Inference Server era)
10.2 Iteration-level / continuous batching (Orca 2022) with a queueing-theory view
10.3 Chunked prefill and stall-free scheduling (Sarathi, Sarathi-Serve)
10.4 Preemption: swap vs recompute; fairness (VTC), priority, SLO-aware scheduling
10.5 Output-length prediction and shortest-job-first approximations
**Interview Qs.** Why does continuous batching raise throughput by an order of magnitude? Prefill-decode interference.
**Refs.** Yu et al. 2022 (Orca); Agrawal et al. 2023/2024; Sheng et al. 2024 (VTC); Fu et al. 2024.

### Chapter 11 — Inside an Inference Engine: vLLM, SGLang, TensorRT-LLM and friends
11.1 Anatomy: API server, tokenizer, scheduler, KV manager, model runner, sampler, detokenizer
11.2 vLLM V0 → V1 (2025): engine-core process, unified token-budget scheduler, zero-overhead prefix caching
11.3 SGLang: frontend language, RadixAttention, overlap scheduler, zero-overhead batch scheduling
11.4 TensorRT-LLM, TGI, LMDeploy, llama.cpp, MLC-LLM: design choices compared
11.5 OpenAI-compatible APIs, streaming (SSE), tool calls
**Refs.** vLLM & SGLang docs/blogs; NVIDIA TensorRT-LLM docs; Zhou et al. 2024 (survey); Park et al. 2025 (inference-engine survey).

### Chapter 12 — Disaggregated Serving
12.1 Prefill/decode interference and SLOs (TTFT vs TPOT); goodput
12.2 Splitwise, DistServe (2024): placement optimisation
12.3 Mooncake (2024–25): KV-centric architecture; Kimi deployment
12.4 KV transfer: RDMA, GPUDirect, NIXL; transfer-time model
12.5 NVIDIA Dynamo, llm-d; attention–FFN disaggregation (MegaScale-Infer, Step-3)
**Interview Qs.** When does disaggregation *not* pay off? Size the KV transfer bandwidth.
**Refs.** Patel 2024; Zhong 2024; Qin 2025; Zhu 2025.

---

## Part V — Making Models Smaller

### Chapter 13 — Quantisation
13.1 Number formats: FP32/TF32/BF16/FP16/FP8 (E4M3/E5M2)/INT8/INT4/FP4; microscaling (MX), NVFP4
13.2 Quantisation as rounding to a lattice; uniform vs non-uniform; error analysis; outlier features
13.3 Weight-only PTQ: GPTQ (OBS heritage), AWQ, HQQ; activations: LLM.int8(), SmoothQuant
13.4 Rotations: QuaRot, SpinQuant (Hadamard incoherence processing; QuIP#)
13.5 FP8/FP4 inference in practice; KV-cache quantisation; QAT
13.6 What speeds up and why (memory-bound vs compute-bound)
**Interview Qs.** Why does W4A16 help decode but not prefill? Why are outliers a problem?
**Refs.** Dettmers 2022; Xiao 2022; Frantar 2022; Lin 2023; Ashkboos 2024; Liu 2024 (SpinQuant); OCP MX spec 2023; Kurtic et al. 2024.

### Chapter 14 — Pruning, Sparsity and Distillation
14.1 Unstructured, semi-structured (2:4), structured pruning; SparseGPT, Wanda
14.2 Width/depth pruning + distillation (Minitron)
14.3 Knowledge distillation: logit, sequence-level, on-policy
14.4 Activation sparsity (Deja Vu, PowerInfer)
**Refs.** Frantar & Alistarh 2023; Sun et al. 2023; Muralidharan 2024; Hinton 2015; Agarwal 2024; Liu 2023 (Deja Vu).

---

## Part VI — Distributed Inference

### Chapter 15 — Distributed Computing Fundamentals
15.1 Why distribute: memory capacity, bandwidth aggregation, latency
15.2 Collectives: broadcast, reduce, all-reduce, all-gather, reduce-scatter, all-to-all; algebraic view
15.3 The α–β cost model; ring and tree algorithms; bandwidth-optimality proof for ring all-reduce
15.4 Fabrics: NVLink/NVSwitch, InfiniBand, RoCE, topology (fat-tree, rail-optimised, torus)
15.5 NCCL, NVSHMEM, GPU-initiated communication; compute–communication overlap
**Interview Qs.** Cost of all-reduce of X bytes over 8 GPUs; ring vs tree.
**Refs.** Thakur et al. 2005; Patarasuk & Yuan 2009; NCCL docs; Hoefler et al.

### Chapter 16 — Parallelism Strategies for Inference
16.1 Data parallelism (replicas) and DP-attention
16.2 Tensor parallelism (Megatron): column/row splits, two all-reduces per layer; latency model
16.3 Pipeline parallelism: bubbles, micro-batching for inference
16.4 Sequence & context parallelism; Ring Attention; Ulysses
16.5 Choosing a configuration: a small optimisation problem
**Refs.** Shoeybi 2019; Narayanan 2021; Pope 2022; Liu 2023 (Ring Attention); Jacobs 2023 (Ulysses).

### Chapter 17 — Mixture of Experts at Inference
17.1 From Shazeer 2017 to Switch, Mixtral, DeepSeek-MoE, Qwen3-MoE, Kimi K2
17.2 Routing, capacity, load balancing, auxiliary-loss-free balancing
17.3 Expert parallelism: all-to-all dispatch/combine; DeepEP; wide-EP; redundant experts (EPLB)
17.4 MoE arithmetic intensity: why MoE needs big batches; total vs active params
17.5 Case study: DeepSeek-V3/R1 production deployment (PD disaggregation + EP)
**Refs.** Shazeer 2017; Fedus 2021; Jiang 2024; DeepSeek-AI 2024; DeepSeek open-infra 2025.

---

## Part VII — Faster Decoding and Longer Contexts

### Chapter 18 — Speculative Decoding
18.1 The idea (Leviathan; Chen 2022–23) and the rejection-sampling proof of exactness
18.2 Expected speed-up as a function of acceptance rate α and draft length γ
18.3 Draft strategies: small models, n-gram/prompt lookup, Medusa heads, EAGLE 1/2/3, MTP (DeepSeek-V3), lookahead decoding
18.4 Tree verification and tree attention masks
18.5 Speculation under batching: when it hurts throughput
**Refs.** Leviathan 2023; Chen 2023; Cai 2024; Li 2024/2025; Fu 2024; Gloeckle 2024.

### Chapter 19 — Long-Context Inference
19.1 Position extrapolation: PI, NTK-aware, YaRN, LongRoPE
19.2 Sparse & efficient attention: sliding windows, block-sparse, NSA, DSA (DeepSeek Sparse Attention)
19.3 Context parallelism at inference; KV offloading
19.4 Retrieval vs long context; needle-in-a-haystack and its limits
**Refs.** Chen 2023 (PI); Peng 2023 (YaRN); Yuan 2025 (NSA); Liu 2023.

### Chapter 20 — Beyond Attention: State-Space Models, Linear Attention and Hybrids
20.1 Linear attention as an RNN; kernel feature maps
20.2 S4 → Mamba → Mamba-2 (selective SSMs; state-space duality)
20.3 RWKV, RetNet, Gated DeltaNet
20.4 Hybrids (Jamba, Samba, Qwen3-Next, Nemotron-H) and their serving implications (constant state, prefix caching difficulties)
**Refs.** Katharopoulos 2020; Gu 2021; Gu & Dao 2023; Dao & Gu 2024; Lieber 2024; Yang 2024.

---

## Part VIII — Production

### Chapter 21 — Reasoning Models and Test-Time Compute
21.1 Chain-of-thought, self-consistency, best-of-N, tree search
21.2 o1/R1-style RL-trained reasoning; inference-time scaling laws
21.3 Serving long outputs: decode dominance, KV growth, early exit/budget forcing
21.4 RL rollouts as an inference workload
**Refs.** Wei 2022; Wang 2022; Snell 2024; DeepSeek-AI 2025; Muennighoff 2025 (s1).

### Chapter 22 — Serving at Scale: Routing, Autoscaling, Multi-LoRA and Cost
22.1 Load balancing: KV/prefix-aware routing, power-of-two choices
22.2 Autoscaling and cold starts; model loading (safetensors, streaming weights)
22.3 Multi-tenant LoRA: S-LoRA, Punica
22.4 Cost modelling: $/M tokens from first principles; spot vs reserved; GPU selection
22.5 Reliability: timeouts, retries, rate limiting, graceful degradation
**Refs.** Sheng 2023 (S-LoRA); Chen 2023 (Punica); Mitzenmacher 2001; Fu 2024 (ServerlessLLM).

### Chapter 23 — Benchmarking, Observability and Evaluation
23.1 Benchmark methodology: open vs closed loop, Poisson arrivals, warm-up
23.2 MLPerf Inference; vendor-benchmark pitfalls
23.3 Observability: per-request tracing, KV utilisation, queue depth
23.4 Quality regressions from optimisations (quantisation, speculative settings)
**Refs.** Reddi et al. 2020 (MLPerf); vLLM benchmarks; SemiAnalysis InferenceMAX.

### Chapter 24 — On-Device and Edge Inference
24.1 Constraints: memory, power, thermals; NPUs
24.2 llama.cpp/GGUF quant schemes, Apple MLX, ExecuTorch
24.3 Small language models and distillation for edge
**Refs.** Gerganov (llama.cpp); Alizadeh 2023 (LLM in a flash); Apple MLX docs.

### Chapter 25 — The ML-Systems Design Interview: Worked Case Studies
25.1 A framework: requirements → workload model → back-of-envelope → architecture → bottlenecks → trade-offs
25.2 Design a ChatGPT-scale chat service
25.3 Design low-latency code completion
25.4 Design a batch summarisation pipeline (throughput-optimal)
25.5 Design serving for a 1T-parameter MoE
25.6 Design an RL rollout fleet
25.7 Rubrics: what interviewers listen for at L5/L6/L7

---

## Appendices
- **A. Numerical precision for mathematicians** — IEEE-754, rounding, Kahan summation, why BF16 exists.
- **B. Back-of-the-envelope cheat-sheet** — hardware table, model table, all boxed formulas.
- **C. A tour of the PyTorch you need** — tensors, dtypes, devices, autograd off, `torch.compile`.
- **D. Linux, Docker and Kubernetes survival guide** for the inference engineer.
- **E. Glossary.**
- **F. Consolidated bibliography.**

---

## Site / infrastructure plan
- Static site in `docs/` (GitHub Pages, deployed by `.github/workflows/pages.yml`).
- `docs/assets/js/book.js` — TOC, theme, MCQ and Q&A engines, progress store (localStorage).
- `docs/assets/js/auth.js` — Firebase Auth (Google) + Firestore sync of progress; inert until `firebase-config.js` is filled.
- `docs/assets/js/charts.js` — tiny SVG chart renderer (data lives in the page).
- KaTeX via CDN for equations; all diagrams are inline SVG.
- `docs/progress.html` — personal dashboard (scores per chapter, questions to review, export/reset).

## Writing status
| Ch | Title | Status |
|---|---|---|
| 1 | From Shannon to Serving | ✅ Draft 1 |
| 2 | The Transformer, Dissected for Inference | ✅ Draft 1 |
| 3–25 | — | ⏳ Planned |
