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

### Chapter 7 — GPU Programming for Mathematicians ✅ *(v2)*
**Motivating question.** Why does fusing operations or changing a loop order make code 10× faster, and how is a fast kernel written?
**Arc.** Kernels as maps over index spaces → coalescing, shared memory, occupancy → tiled GEMM → reductions and online softmax → fusion → Triton → CUDA Graphs and launch overhead → torch.compile → CuTe-DSL, ThunderKittens, TileLang.
**Refs.** CUDA Guide; Tillet 2019; Spector 2024; Ansel 2024; Milakov 2018.

---

## Part III — Attention and Memory

### Chapter 8 — The KV Cache: Shrinking It ✅ *(v2)*
**Motivating question.** The cache limits batch size and dominates decode traffic. How small can we make it without hurting quality?
**Arc.** MQA → GQA → MLA (derivation and weight absorption) → sliding-window and local/global hybrids → cross-layer sharing → KV quantisation → eviction (attention sinks, H2O, SnapKV) and why reasoning models break it.
**Refs.** Shazeer 2019; Ainslie 2023; DeepSeek-V2 2024; Xiao 2023; Zhang 2023; Liu 2024 (KIVI); Brandon 2024.

### Chapter 9 — IO-Aware Attention: FlashAttention 1 → 4 ✅ *(v2)*
**Motivating question.** Attention with a \(T\times T\) matrix moves \(\Theta(T^2)\) bytes. Can we compute exactly the same function while moving far less?
**Arc.** IO complexity → online softmax derivation → FA1 tiling → FA2 → FA3 (Hopper asynchrony, FP8) → FA4 (Blackwell) → decode kernels (FlashDecoding, FlashInfer) → MLA and sparse kernels.
**Refs.** Dao 2022/2023; Shah 2024; FA4 2026; Rabe & Staats 2021; Ye 2025.

### Chapter 10 — Memory Management: PagedAttention and Prefix Caching ✅ *(v2)*
**Motivating question.** Sequences grow unpredictably. How do we allocate KV memory without wasting most of it, and reuse work across requests?
**Arc.** Fragmentation → paging and block tables → copy-on-write → prefix caching (hash chains vs radix trees) → eviction policy → hierarchical KV storage (CPU, SSD, remote: LMCache, Mooncake Store) → vAttention.
**Refs.** Kwon 2023; Zheng 2024; Prabhu 2024; Qin 2025.

---

## Part IV — Serving Systems

### Chapter 11 — Batching and Scheduling ✅ *(v2)*
**Motivating question.** Requests arrive at random and finish at different times. How do we keep the GPU busy without making anyone wait too long?
**Arc.** Static and dynamic batching → continuous batching (Orca) → chunked prefill (Sarathi) → preemption → fairness, priorities and SLO-aware scheduling → length prediction.
**Refs.** Yu 2022; Agrawal 2024; Sheng 2024 (VTC); Fu 2024.

### Chapter 12 — Inside an Inference Engine ✅ *(v2)*
**Motivating question.** How do all these ideas fit together into software such as vLLM or SGLang?
**Arc.** Anatomy of an engine → vLLM V0 → V1 → SGLang → TensorRT-LLM, TGI, llama.cpp → APIs and streaming.
**Refs.** Gordić 2025 (*Inside vLLM*); vLLM and SGLang docs; Park 2025 (engine survey).

### Chapter 13 — Disaggregated Serving ✅ *(v2)*
**Motivating question.** Prefill and decode want different hardware and interfere with each other. Should they run on different machines?
**Arc.** Interference and goodput → Splitwise, DistServe → Mooncake → KV transfer (RDMA, NIXL) → Dynamo, llm-d → attention/FFN disaggregation.
**Refs.** Patel 2024; Zhong 2024; Qin 2025.

---

## Part V — Making Models Smaller and Adapting Them

### Chapter 14 — Quantisation ✅ *(v2)*
**Motivating question.** Decode time is proportional to bytes read. How few bits per weight and per cache entry can we use?
**Arc.** Number formats (FP8, INT4, MXFP4, NVFP4) → quantisation as lattice rounding, error analysis, outliers → LLM.int8, SmoothQuant → GPTQ, AWQ → rotations (QuaRot, SpinQuant) → FP8/FP4 in practice → what actually speeds up.
**Refs.** Dettmers 2022; Xiao 2022; Frantar 2022; Lin 2023; Ashkboos 2024; Kurtic 2024; OCP MX 2023.

### Chapter 15 — Pruning, Sparsity and Distillation ✅ *(v2)*
**Refs.** Frantar & Alistarh 2023; Sun 2023; Muralidharan 2024; Hinton 2015; Agarwal 2024; Liu 2023.

---

### Chapter 16 — Fine-Tuning and Parameter-Efficient Adaptation ✅ *(v3, new)*
- Why adapt; the memory bill of full fine-tuning (weights, gradients, Adam, activations); the low-rank hypothesis and intrinsic dimension; LoRA derivation, initialisation, \(\alpha/r\) scaling, merging; which matrices and which rank; QLoRA (NF4, double quantisation, paged optimisers); DoRA, rsLoRA, LoRA+; adapters, prefix and prompt tuning, IA³; choosing a method; bridge to multi-LoRA serving (Ch 27).

## Part VI — Distributed Inference

### Chapter 17 — Distributed Computing Fundamentals ✅ *(v2)*
**Motivating question.** The model or its cache does not fit on one GPU. What does it cost to move data between GPUs?
**Arc.** Collectives as linear-algebra operations → the α–β cost model → ring and tree all-reduce (proof of bandwidth optimality) → NVLink, InfiniBand, RoCE, topologies → NCCL, NVSHMEM, overlap.
**Refs.** Thakur 2005; Patarasuk & Yuan 2009; NCCL docs.

### Chapter 18 — Parallelism Strategies for Inference ✅ *(v2)*
**Arc.** Data parallelism → tensor parallelism (Megatron) → pipeline parallelism → sequence and context parallelism (Ring, Ulysses) → choosing a configuration.
**Refs.** Shoeybi 2019; Narayanan 2021; Pope 2022; Liu 2023; Jacobs 2023.

### Chapter 19 — Mixture of Experts at Inference ✅ *(v2)*
**Arc.** From Shazeer 2017 to DeepSeek-V3 and Kimi K2 → routing and balancing → expert parallelism (all-to-all, DeepEP, EPLB) → MoE arithmetic intensity → the DeepSeek-V3 deployment case study.
**Refs.** Shazeer 2017; Fedus 2021; Jiang 2024; DeepSeek-AI 2024/2025.

---

## Part VII — Faster Decoding and Longer Contexts

### Chapter 20 — Speculative Decoding ✅ *(v2)*
**Arc.** Exactness via rejection sampling (proof) → expected speed-up → drafts (small models, n-grams, Medusa, EAGLE 1–3, MTP) → tree verification → speculation under batching.
**Refs.** Leviathan 2023; Chen 2023; Cai 2024; Li 2024/2025.

### Chapter 21 — Long-Context Inference ✅ *(v2)*
**Arc.** Position extrapolation (PI, NTK, YaRN) → sparse attention (NSA, DSA) → context parallelism and offload → retrieval vs long context.
**Refs.** Chen 2023; Peng 2023; Yuan 2025.

### Chapter 22 — Beyond Attention: SSMs, Linear Attention and Hybrids ✅ *(v2)*
**Arc.** Linear attention as an RNN → S4 → Mamba → Mamba-2 → RWKV, Gated DeltaNet → hybrids and their serving implications.
**Refs.** Katharopoulos 2020; Gu & Dao 2023; Dao & Gu 2024; Lieber 2024.

---

## Part VIII — Post-Training: Reinforcement Learning, Alignment and Reasoning

### Chapter 23 — Reinforcement Learning for Language Models ✅ *(v3, new)*
- Generation as a token-level MDP; returns and the sequence-level reward; the policy-gradient theorem and REINFORCE; variance, baselines and advantages; value functions and GAE; the KL-regularised objective and its closed-form optimum; trust regions to PPO's clipped surrogate; the four models of PPO for LLMs and what they cost.

### Chapter 24 — Learning from Human Preferences ✅ *(v3, new)*
- SFT → reward model → RL (InstructGPT, 2022); Bradley–Terry reward models; reward hacking and over-optimisation; DPO from the closed-form optimum; IPO, KTO, ORPO, SimPO; online versus offline preference learning; Constitutional AI and RLAIF.

### Chapter 25 — Reinforcement Learning with Verifiable Rewards ✅ *(v3, new)*
- Verifiable rewards (math, code, tests); GRPO derived; RLOO and REINFORCE++; Dr. GRPO and DAPO fixes; outcome versus process rewards; entropy collapse and exploration; rollout systems: synchronous vs asynchronous, weight sync, off-policy correction; bridge to reasoning models (Ch 26).


### Chapter 26 — Reasoning Models and Test-Time Compute ✅ *(v2)*

## Part IX — Production

### Chapter 27 — Serving at Scale: Routing, Autoscaling, Multi-LoRA and Cost ✅ *(v2)*
### Chapter 28 — Benchmarking, Observability and Evaluation ✅ *(v2)*
### Chapter 29 — On-Device and Edge Inference ✅ *(v2)*
### Chapter 30 — The ML-Systems Design Interview: Worked Case Studies ✅ *(v2)*
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
| 7 | GPU Programming for Mathematicians | ✅ v2 |
| 8 | The KV Cache: Shrinking It | ✅ v2 |
| 9 | IO-Aware Attention: FlashAttention 1 → 4 | ✅ v2 |
| 10 | Memory Management: PagedAttention and Prefix Caching | ✅ v2 |
| 11 | Batching and Scheduling | ✅ v2 |
| 12 | Inside an Inference Engine | ✅ v2 |
| 13 | Disaggregated Serving | ✅ v2 |
| 14 | Quantisation | ✅ v2 |
| 15 | Pruning, Sparsity and Distillation | ✅ v2 |
| 16 | Fine-Tuning and Parameter-Efficient Adaptation | ✅ v3 (new) |
| 17 | Distributed Computing Fundamentals | ✅ v2 |
| 18 | Parallelism Strategies for Inference | ✅ v2 |
| 19 | Mixture of Experts at Inference | ✅ v2 |
| 20 | Speculative Decoding | ✅ v2 |
| 21 | Long-Context Inference | ✅ v2 |
| 22 | Beyond Attention: SSMs and Hybrids | ✅ v2 |
| 23 | Reinforcement Learning for Language Models | ✅ v3 (new) |
| 24 | Learning from Human Preferences | ✅ v3 (new) |
| 25 | Reinforcement Learning with Verifiable Rewards | ✅ v3 (new) |
| 26 | Reasoning Models and Test-Time Compute | ✅ v2 |
| 27 | Serving at Scale | ✅ v2 |
| 28 | Benchmarking, Observability and Evaluation | ✅ v2 |
| 29 | On-Device and Edge Inference | ✅ v2 |
| 30 | The ML-Systems Design Interview | ✅ v2 |

---

# Volume II — Cloud AI Infrastructure: Serving Large Models at Datacenter Scale

> **Same reader, same rules.** Everything in `CLAUDE.md` applies (prose first, the motivation → idea → mathematics → worked example → consequences → bridge arc, 30+ MCQs, 15+ interview questions, ≥ 4 SVG diagrams and 2 charts, British spelling). Files live in `docs/cloud/ciNN.html`, use `<body data-book="cloud" data-chapter="ciNN" data-root="../">`, and are registered in `CLOUD_CHAPTERS` in `docs/assets/js/book.js`. Question ids use the prefix `ciNN-` so saved progress never collides with Volume I. Cite Volume I as "Vol. I §11.8".
> **Why a second volume.** Volume I ends at the boundary of the fleet (Chapter 27 of Vol. I). Interviews for infrastructure, ML-platform and inference roles probe everything beyond that boundary — power, instances, networks, storage, Kubernetes, capacity, money, global routing, reliability, security — and several workload families Volume I left open.

## Gaps in Volume I that Volume II closes
| Gap in Volume I | Where Volume II covers it |
|---|---|
| Physical datacenter: power, cooling, rack density, energy per token | Ch 2 |
| Cloud instance types, GPU sharing (MIG, MPS, time-slicing), non-NVIDIA cloud accelerators as products | Ch 3 |
| Datacenter fabric design (front-end vs back-end, fat trees, rail-optimised, congestion control) — Vol. I Ch 17 covers collectives only | Ch 4 |
| Weight storage, file formats, image pulls and the anatomy of a cold start | Ch 5 |
| Containers, Kubernetes, device plugins, DRA; cluster scheduling | Ch 6–7 |
| Serving platforms and managed endpoints (KServe, Ray Serve, Triton, Dynamo, llm-d, Bedrock/Vertex/SageMaker, serverless GPUs) | Ch 8 |
| Procurement (reserved / on-demand / spot), TCO, pricing tokens, FinOps | Ch 9–10 |
| Streaming protocols, gateways, distributed rate limiting | Ch 11 |
| Multi-region serving, data residency, disaster recovery | Ch 12 |
| SLOs and error budgets, GPU failure modes, MTBF at scale | Ch 13 |
| Security: threat models, confidential computing, prompt-cache side channels, weight theft, model supply chain | Ch 14 |
| **Multimodal serving** (vision encoders, image tokens, audio, real-time speech) | Ch 15 |
| **Embeddings, vector search and RAG infrastructure** | Ch 16 |
| **Agents and compound AI** (tool calls, MCP, sandboxes, session state) | Ch 17 |
| **Guardrails and moderation infrastructure** | Ch 18 |
| **Batch / offline inference and RL rollouts as a cloud workload** | Ch 19 |
| Infrastructure system-design interview | Ch 20 |

## Part I — The Physical Cloud
### Chapter 1 — Why Serving Became a Cloud Problem
**Motivating question.** One H100 can serve Llama 3 8B. Why does serving a popular model take a building, a power contract and a global network?
**Arc.** From one request to a planet of requests → the stack (silicon, server, rack, pod, hall, campus, region) → three scarce resources: power, accelerators, people → who builds what: hyperscalers, neoclouds, model labs, enterprises → the running service ("Atlas") and its demand model → orders of magnitude: tokens/day → GPUs → megawatts → dollars → the control plane / data plane split → map of the volume.
### Chapter 2 — The AI Datacenter: Power, Cooling and Space
**Arc.** Watts per GPU and per server → rack density from 10 kW to 120 kW+ → PUE and where the overhead goes → heat removal: air, rear-door, direct-to-chip liquid, immersion → power delivery and redundancy (N, N+1, 2N) → energy per token derived → the grid, interconnection queues and siting → stranded power and oversubscription → water.
### Chapter 3 — Accelerators as Cloud Products
**Arc.** What an "instance" is (VM, bare metal, virtualisation) → instance families on AWS, GCP, Azure, OCI and neoclouds → GPU sharing: time-slicing, MPS, MIG, vGPU, with the mathematics of interference → TPUs, Trainium/Inferentia, MI300X/MI355X as products → price–performance per token as the selection criterion → a hardware-selection procedure.
### Chapter 4 — Datacenter Networks for AI
**Arc.** Two networks (front-end and back-end) → Clos and fat-tree topology mathematics (radix, tiers, bisection, oversubscription) → rail-optimised designs → InfiniBand vs RoCE vs cloud fabrics (EFA/SRD) → congestion, ECMP collisions and load balancing → scale-up domains (NVL72) vs scale-out → what inference needs from the network (KV transfer, EP all-to-all, weight loading).
### Chapter 5 — Storage, Weights and Cold Starts
**Arc.** Anatomy of a cold start → object storage throughput model → file formats (pickle → safetensors → GGUF) → parallel range reads, streaming into GPU memory, GPUDirect Storage → caching tiers and peer-to-peer distribution → container images and lazy pulling → snapshotting engine state → cold-start budget for 8B, 70B, 671B.

## Part II — Orchestration
### Chapter 6 — Containers and Kubernetes for Accelerators
**Arc.** Processes, isolation and the container → the Kubernetes model (desired state, controllers, reconciliation) → pods, deployments, services → exposing GPUs: device plugins → DRA → GPU operator, drivers, DCGM → multi-node serving: LeaderWorkerSet → health, probes and lifecycles for LLM servers.
### Chapter 7 — Scheduling a GPU Cluster
**Arc.** Scheduling as constrained bin packing → fragmentation and its measurement → gang scheduling → topology-aware placement → quotas, fair share and borrowing (Kueue, KAI, Volcano, Slurm) → preemption and priorities → inference and training sharing one cluster.
### Chapter 8 — Serving Platforms and Control Planes
**Arc.** What a platform adds to an engine → Triton, KServe, Ray Serve, BentoML → LLM-native control planes: Dynamo, llm-d, AIBrix, the Gateway API inference extension → managed endpoints (SageMaker, Vertex, Azure, Bedrock) and token APIs → serverless GPUs and scale to zero → model lifecycle: registry, rollout, shadowing.

## Part III — Elasticity and Economics
### Chapter 9 — Capacity, Procurement and Autoscaling
**Arc.** Capacity as the binding constraint → purchase options (reserved, committed, on-demand, capacity blocks, spot) → the capacity mix as a newsvendor / linear programme → forecasting demand → spot and preemption mathematics → autoscaling signals and control → regional capacity and fallbacks.
### Chapter 10 — The Economics of Inference
**Arc.** TCO from first principles (capex, power, space, network, people) → cost per GPU-hour → cost per million tokens across hardware and models → pricing tokens (input vs output, cached input, batch, priority tiers) → build vs buy vs API → FinOps: attribution, chargeback, utilisation metrics that lie.

## Part IV — The Global Request Path
### Chapter 11 — The Front Door: Gateways, Streaming and Rate Limits
**Arc.** A request's journey (DNS, TLS, LB, gateway) → HTTP/1.1, HTTP/2, SSE, WebSockets, gRPC for token streaming → long-lived connections and load balancers → distributed rate limiting (token bucket, sliding window, GCRA) → admission control and load shedding → API design for LLMs.
### Chapter 12 — Multi-Region Serving and Disaster Recovery
**Arc.** Why multiple regions → global load balancing (DNS, anycast, GSLB) → latency vs capacity routing as an optimisation → data residency and sovereignty → state that must follow users (prefix caches, sessions, files) → failover and DR (RTO, RPO) → capacity across regions.
### Chapter 13 — Reliability Engineering for AI Services
**Arc.** SLIs, SLOs, error budgets → availability mathematics (series, parallel, k-of-n) → GPU failure modes (Xid, ECC, NVLink, thermal) and MTBF at scale → health checking and quarantine → safe change: canaries, progressive rollout, model-version rollbacks → overload and graceful degradation → incident response and postmortems.
### Chapter 14 — Security, Isolation and Multi-Tenancy
**Arc.** Threat models for model APIs → isolation layers (process, container, VM, MIG, confidential computing) → side channels in shared serving (prompt-cache timing, KV sharing) → protecting weights (SL1–SL5) → model supply chain (pickle, signing, provenance) → data protection, retention and logging → prompt injection as an infrastructure problem.

## Part V — Beyond Chat: Workloads Volume I Left Open
### Chapter 15 — Serving Multimodal Models
**Arc.** How images become tokens (ViT, patching, dynamic resolution) → the cost of the encoder and of image tokens → encoder caching and EPD disaggregation → audio: ASR, TTS and real-time speech-to-speech → video and long multimodal contexts → image generation (diffusion) serving in brief.
### Chapter 16 — Embeddings, Retrieval and Vector Search at Scale
**Arc.** Why retrieval → embedding models and their serving profile → exact search cost → ANN: IVF, PQ (mathematics of quantisation error), HNSW, DiskANN → recall–latency–memory trade-offs → sharding and replication of indexes → rerankers → RAG latency and cost budgets.
### Chapter 17 — Agents and Compound AI Systems
**Arc.** From one call to a program of calls → tool calling and MCP → sandboxes for code execution (containers, gVisor, Firecracker) → session state and KV retention across tool calls → program-aware scheduling (Parrot, Autellix) → cost and latency of agent loops → reliability of multi-step workflows.
### Chapter 18 — Guardrails and Moderation Infrastructure
**Arc.** Why guardrails are infrastructure → classifier families (Llama Guard, ShieldGemma, constitutional classifiers) → placement: before, during (streaming), after → latency and cost budgets of guards → precision, recall and base rates → audit, appeals and policy as configuration.
### Chapter 19 — Batch, Offline and Rollout Inference
**Arc.** Throughput-optimal serving → batch APIs and their pricing → pipelines on spot capacity with checkpointed progress → synthetic-data generation and evaluation sweeps → RL rollouts as an inference workload → scheduling batch alongside online.

## Part VI — The Interview
### Chapter 20 — The Cloud Infrastructure Design Interview
**Arc.** What infrastructure interviews assess → estimation toolkit (power, GPUs, network, storage, dollars) → case studies: a global LLM API; a serverless GPU platform; a GPU cluster scheduler; a model-rollout system; an embedding and retrieval service; an agent sandbox fleet → rubrics and failure modes.

## Volume II writing status
| Ch | Title | Status |
|---|---|---|
| 1 | Why Serving Became a Cloud Problem | ✅ v1 |
| 2 | The AI Datacenter: Power, Cooling and Space | ✅ v1 |
| 3 | Accelerators as Cloud Products | ✅ v1 |
| 4 | Datacenter Networks for AI | ✅ v1 |
| 5 | Storage, Weights and Cold Starts | ✅ v1 |
| 6 | Containers and Kubernetes for Accelerators | ✅ v1 |
| 7 | Scheduling a GPU Cluster | ✅ v1 |
| 8 | Serving Platforms and Control Planes | ✅ v1 |
| 9 | Capacity, Procurement and Autoscaling | ✅ v1 |
| 10 | The Economics of Inference | ✅ v1 |
| 11 | The Front Door: Gateways, Streaming and Rate Limits | ✅ v1 |
| 12 | Multi-Region Serving and Disaster Recovery | ⏳ |
| 13 | Reliability Engineering for AI Services | ⏳ |
| 14 | Security, Isolation and Multi-Tenancy | ⏳ |
| 15 | Serving Multimodal Models | ⏳ |
| 16 | Embeddings, Retrieval and Vector Search at Scale | ⏳ |
| 17 | Agents and Compound AI Systems | ⏳ |
| 18 | Guardrails and Moderation Infrastructure | ⏳ |
| 19 | Batch, Offline and Rollout Inference | ⏳ |
| 20 | The Cloud Infrastructure Design Interview | ⏳ |
