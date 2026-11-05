# Notation used throughout *Introduction to LLM Engineering*

Every chapter must use these symbols. If a new symbol is needed, add it here first and check it does not collide with an existing one. The same table is rendered on the site at `docs/notation.html`.

## Model architecture
| Symbol | Meaning | Llama 3 8B | Llama 3 70B |
|---|---|---|---|
| \(L\) | number of layers (Transformer blocks) | 32 | 80 |
| \(\ell\) | layer index, \(\ell = 1,\dots,L\) | | |
| \(d\) | model width (residual-stream dimension, `hidden_size`) | 4096 | 8192 |
| \(n_h\) | number of query heads | 32 | 64 |
| \(n_{kv}\) | number of key/value heads | 8 | 8 |
| \(g\) | GQA group size \(n_h/n_{kv}\) | 4 | 8 |
| \(d_h\) | head dimension | 128 | 128 |
| \(d_{ff}\) | FFN hidden width (`intermediate_size`) | 14336 | 28672 |
| \(\mathcal V\), \(\lvert\mathcal V\rvert\) | vocabulary and its size | 128,256 | 128,256 |
| \(N\) | number of parameters (\(N_{\text{act}}\): active per token, for MoE) | 8.03B | 70.6B |
| \(\Theta\) | RoPE base (`rope_theta`); frequencies \(\theta_r=\Theta^{-2r/d_h}\), \(r=0,\dots,d_h/2-1\) | 500,000 | 500,000 |

## Sequences, batches, tensors
| Symbol | Meaning |
|---|---|
| \(x_t\) | token at position \(t\); \(x_{<t}=(x_1,\dots,x_{t-1})\) |
| \(T\) | length of a sequence (when the whole sequence is in view) |
| \(t\) | current position / current context length during generation |
| \(i, j\) | query position and key position inside attention |
| \(n_{\text{in}}, n_{\text{out}}\) | prompt length; number of generated tokens |
| \(B\) | batch size: number of sequences processed together |
| \(m\) | number of token vectors in one matrix product (\(m=B\) in decode, \(m=B\,n_{\text{in}}\) in prefill) |
| \((m\times k)\cdot(k\times n)\) | generic matrix-product shapes; costs \(2mkn\) FLOPs |
| \(X\) | input to a sub-layer, \(T\times d\); \(\tilde X=\operatorname{Norm}(X)\) |
| \(Q,K,V\) | queries, keys, values; \(W_Q,W_K,W_V,W_O\) their projections |
| \(S, P, O\) | attention scores, attention probabilities, attention output |
| \(M\) | causal mask (\(0\) or \(-\infty\)) |
| \(W_1,W_3,W_2\) | SwiGLU gate, up and down projections |
| \(d_c\), \(d_h^R\), \(c_t\) | MLA latent dimension (512), decoupled RoPE dimension (64), cached latent of token \(t\) (Ch. 8) |
| \(W_{DKV}, W_{UK}, W_{UV}\) | MLA down-projection and key/value up-projections (Ch. 8) |
| \(w\), \(\phi\) | attention window size; fraction of global (full-attention) layers (Ch. 8) |
| \(W_E\in\mathbb R^{\lvert\mathcal V\rvert\times d}\), \(W_U\) | embedding table; unembedding (LM head) |
| \(z\), \(\tau\) | logits; sampling temperature |
| \(v\in\mathcal V\), \(p(v)\) | a vocabulary token; its next-token probability (Ch. 4) |
| \(\kappa=1/\tau\) | inverse temperature (Ch. 4) |
| \(n_{\text{top}}, p_{\text{top}}, p_{\min}\) | top-\(k\) count; top-\(p\) mass; min-\(p\) ratio (Ch. 4) |
| \(n_{\text{beam}}\) | beam width (Ch. 4) |
| \(\rho\), \(\xi_v\) | repetition-penalty factor; Gumbel noise for token \(v\) (Ch. 4) |
| \(h_t\); \(h^{(\ell)}_j\) | recurrent hidden state (RNNs, Ch. 1); residual-stream vector of position \(j\) after layer \(\ell\) (Ch. 3 proofs) |

## Training and information
| Symbol | Meaning |
|---|---|
| \(\mathcal L\) | loss (never \(L\), which is layers) |
| \(H\), \(\mathrm{PP}\) | cross-entropy (nats or bits per token); perplexity \(e^H\) |
| \(D\), \(D_{\text{inf}}\) | training tokens; tokens processed at inference over a model's lifetime |
| \(C\) | compute in FLOPs |
| \(E, c_N, c_D, \alpha_N, \alpha_D\) | Chinchilla fit \(\mathcal L(N,D)=E+c_N N^{-\alpha_N}+c_D D^{-\alpha_D}\) |

## Hardware and performance
| Symbol | Meaning | H100 SXM (dense BF16) |
|---|---|---|
| \(\pi\) | peak compute, FLOP/s | 989 TFLOP/s |
| \(\beta\) | memory (HBM) bandwidth, bytes/s | 3.35 TB/s |
| \(M_{\text{HBM}}\) | memory capacity | 80 GB |
| \(I\) | arithmetic intensity, FLOPs per byte moved | |
| \(I^*=\pi/\beta\) | ridge point | ≈ 295 FLOP/B |
| \(b\) | bytes per element (\(b_w\) weights, \(b_{kv}\) KV cache); BF16 = 2, FP8 = 1 | |
| \(s_{kv}\) | KV-cache bytes per token \(=2\,L\,n_{kv}\,d_h\,b_{kv}\) | |
| \(t_{\text{step}}\) | time of one forward pass (decode step) | |
| \(f_{\text{clk}}\) | clock frequency (Ch. 5) | ≈ 1.83 GHz |
| \(B^{*}(t)\), \(t_{\text{crit}}\) | critical batch size for compute-bound decode at context \(t\); critical context length (Ch. 6) | 388 tokens for Llama 3 8B |
| MFU, MBU | model FLOPs utilisation; memory-bandwidth utilisation (Ch. 3, 6) | |
| \(\delta_{\text{mem}}\) | main-memory access latency (Ch. 5) | ≈ 600–800 ns |

## Serving
| Symbol | Meaning |
|---|---|
| TTFT, TPOT, ITL | time to first token; mean time per output token after the first; inter-token latency |
| \(\lambda\), \(W\), \(\bar n\) | arrival rate; mean time in system; mean number in system (\(\bar n=\lambda W\), Little's law) |
| \(t_{\text{gpu}}\), \(t_{\text{cpu}}\) | GPU time and CPU time of one engine step (Ch. 12) |
| \(G\), \(q\) | goodput: highest arrival rate at which a fraction \(q\) of requests meet both TTFT and TPOT targets (Ch. 13) |
| \(\beta_{\text{net}}\) | network (scale-out) bandwidth available per GPU, bytes/s; 400 Gb/s = 50 GB/s (Ch. 13) |
| \(\chi\) | KV-transfer ratio: time to send a prompt's KV cache divided by the time to prefill it (Ch. 13) |
| \(n_P\), \(n_D\) | numbers of prefill and decode GPUs in a disaggregated deployment (Ch. 13) |

## Quantisation (Ch. 14)
| Symbol | Meaning |
|---|---|
| \(n_b\) | bits per quantised element |
| \(n_g\) | group (block) size: number of elements sharing one scale |
| \(\Delta\), \(\zeta\) | quantisation step (scale) and zero point; \(\hat x=\Delta\,(q-\zeta)\) |
| \(\hat W\), \(\hat x\) | quantised (dequantised) weight matrix or value |
| \(r_c\) | crest factor \(\max_j\lvert x_j\rvert/\operatorname{rms}(x)\) of a block |
| \(\Sigma=XX^\top\) | second-moment (Gram) matrix of a layer's calibration inputs (GPTQ) |
| \(u_j\), \(\alpha_s\) | SmoothQuant per-channel scales and migration strength |
| \(R\) | orthogonal rotation (QuaRot, SpinQuant); \(\mathcal H_d\) a \(d\times d\) Hadamard matrix |

## Pruning and distillation (Ch. 15)
| Symbol | Meaning |
|---|---|
| \(\eta\) | sparsity: fraction of weights (or activations) that are zero |
| \(p_a\) | probability that an FFN neuron is active for a given token (contextual sparsity) |
| \(p_{\text{te}}\), \(p_{\text{st}}\) | teacher and student next-token distributions (distillation) |
| \(\mathrm{KL}(p\,\Vert\,q)\) | Kullback–Leibler divergence \(\sum_v p(v)\log\frac{p(v)}{q(v)}\) |

## Communication (Ch. 16 onwards)
| Symbol | Meaning |
|---|---|
| \(n_p\) | number of GPUs (ranks) taking part in a collective |
| \(\nu\) | bytes in each rank's buffer for a collective (the message size) |
| \(\alpha\) | per-step (per-message) latency of a communication step, seconds |
| \(\beta_{\text{link}}\) | per-GPU, per-direction link bandwidth, bytes/s (NVLink on H100: 450 GB/s) |
| \(n_{\text{TP}}, n_{\text{PP}}, n_{\text{DP}}, n_{\text{CP}}, n_{\text{EP}}\) | degrees of tensor, pipeline, data, context and expert parallelism (Ch. 17–18) |
| \(n_\mu\) | number of micro-batches in flight in a pipeline (Ch. 17) |
| \(T_c\) | tokens of a sequence held by one device under context parallelism (Ch. 17) |

## Mixture of experts (Ch. 18)
| Symbol | Meaning | DeepSeek-V3 |
|---|---|---|
| \(n_{\text{exp}}\) | routed experts per MoE layer | 256 |
| \(k_r\) | experts selected per token (top-\(k_r\)) | 8 |
| \(d_e\) | hidden width of one expert's FFN | 2048 |
| \(W_r\) | router (gate) matrix, \(d\times n_{\text{exp}}\); affinities \(s_{t,i}\) | |
| \(m_e\) | tokens routed to one expert in a step, \(\approx Bk_r/n_{\text{exp}}\) | |
| \(f_i\), \(\bar p_i\) | fraction of tokens routed to expert \(i\); mean router probability of expert \(i\) | |
| \(\omega_i\) | routing bias of expert \(i\) (auxiliary-loss-free balancing) | |

## Speculative decoding (Ch. 19)
| Symbol | Meaning |
|---|---|
| \(q(v)\) | draft model's next-token distribution (the target's is \(p(v)\)) |
| \(\gamma\) | number of draft tokens proposed per verification step |
| \(a\) | per-token acceptance probability, \(a=\sum_v\min(p(v),q(v))=1-\mathrm{TV}(p,q)\) |
| \(n_{\text{acc}}\) | tokens produced per verification step (accepted drafts plus one) |
| \(c_d\) | cost of one draft step relative to one target decode step |

## Units
- Memory sizes of tensors: binary units (KiB, MiB, GiB). Datasheet capacities and bandwidths: decimal (GB, TB/s), as vendors quote them.
- FLOPs = floating-point operations (a count); FLOP/s = rate. A multiply-add counts as 2 FLOPs.
- "Dense" throughput figures never include the 2× structured-sparsity marketing factor.
