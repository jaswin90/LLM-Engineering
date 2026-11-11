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

## Fine-tuning and parameter-efficient adaptation (Ch. 16)
| Symbol | Meaning |
|---|---|
| \(\mathcal I\) | set of response positions that carry loss in supervised fine-tuning |
| \(W_0\), \(\Delta W\) | frozen pre-trained matrix; the update fine-tuning learns |
| \(W_{\downarrow}, W_{\uparrow}\), \(r_a\) | LoRA down- and up-projections and adapter rank (as in Ch. 27) |
| \(\alpha_a\), \(s\) | LoRA scaling constant; effective scale \(s=\alpha_a/r_a\) (\(\alpha_a/\sqrt{r_a}\) in rsLoRA). Not the latency \(\alpha\) |
| \(\nabla_W\) | full-fine-tuning gradient \(X^\top G\) of a matrix (never formed in LoRA) |
| \(\eta_{\text{lr}}\) | learning rate (subscript distinguishes it from the sparsity \(\eta\)) |
| \(m\), \(\lVert V\rVert_c\) | DoRA magnitude vector; vector of column norms |
| \(l_p\) | number of learned prefix or soft-prompt positions |
| \(n_{\text{DP}}\) | data-parallel degree (ZeRO/FSDP sharding factor) |

## Communication (Ch. 17 onwards)
| Symbol | Meaning |
|---|---|
| \(n_p\) | number of GPUs (ranks) taking part in a collective |
| \(\nu\) | bytes in each rank's buffer for a collective (the message size) |
| \(\alpha\) | per-step (per-message) latency of a communication step, seconds |
| \(\beta_{\text{link}}\) | per-GPU, per-direction link bandwidth, bytes/s (NVLink on H100: 450 GB/s) |
| \(n_{\text{TP}}, n_{\text{PP}}, n_{\text{DP}}, n_{\text{CP}}, n_{\text{EP}}\) | degrees of tensor, pipeline, data, context and expert parallelism (Ch. 18–18) |
| \(n_\mu\) | number of micro-batches in flight in a pipeline (Ch. 18) |
| \(T_c\) | tokens of a sequence held by one device under context parallelism (Ch. 18) |

## Mixture of experts (Ch. 19)
| Symbol | Meaning | DeepSeek-V3 |
|---|---|---|
| \(n_{\text{exp}}\) | routed experts per MoE layer | 256 |
| \(k_r\) | experts selected per token (top-\(k_r\)) | 8 |
| \(d_e\) | hidden width of one expert's FFN | 2048 |
| \(W_r\) | router (gate) matrix, \(d\times n_{\text{exp}}\); affinities \(s_{t,i}\) | |
| \(m_e\) | tokens routed to one expert in a step, \(\approx Bk_r/n_{\text{exp}}\) | |
| \(f_i\), \(\bar p_i\) | fraction of tokens routed to expert \(i\); mean router probability of expert \(i\) | |
| \(\omega_i\) | routing bias of expert \(i\) (auxiliary-loss-free balancing) | |

## Speculative decoding (Ch. 20)
| Symbol | Meaning |
|---|---|
| \(q(v)\) | draft model's next-token distribution (the target's is \(p(v)\)) |
| \(\gamma\) | number of draft tokens proposed per verification step |
| \(a\) | per-token acceptance probability, \(a=\sum_v\min(p(v),q(v))=1-\mathrm{TV}(p,q)\) |
| \(n_{\text{acc}}\) | tokens produced per verification step (accepted drafts plus one) |
| \(c_d\) | cost of one draft step relative to one target decode step |

## Long context (Ch. 21)
| Symbol | Meaning |
|---|---|
| \(T_{\text{train}}\) | longest context seen in (pre-)training |
| \(\mu\) | context-extension factor, target length \(/\,T_{\text{train}}\) |
| \(\Lambda_r=2\pi/\theta_r\) | wavelength (in tokens) of RoPE frequency pair \(r\) |
| \(n_{\text{rot}}(r)=T_{\text{train}}/\Lambda_r\) | rotations pair \(r\) completes over the training length; \(c_{\text{lo}},c_{\text{hi}}\) YaRN ramp thresholds |
| \(k_s\) | tokens (or blocks) selected per query by sparse attention |

## Recurrent and state-space layers (Ch. 22)
| Symbol | Meaning |
|---|---|
| \(\mathcal S_t\) | matrix-valued recurrent state of a linear-attention head after token \(t\) (\(d_v\times d_k\)) |
| \(\psi(\cdot)\) | feature map of linear attention, \(\psi(q)^\top\psi(k)\approx\) similarity |
| \(f_t\), \(\eta_t\) | forget (decay) gate in \((0,1]\) and write rate of a gated / delta-rule state update |
| \(\mathbf A\), \(\mathbf b_t\), \(\mathbf c_t\) | SSM state matrix (diagonal) and input-dependent input/output vectors (Mamba's \(A,B,C\)) |
| \(\Delta_t\) | SSM discretisation step (Ch. 22 only; unrelated to the quantisation step of Ch. 14) |
| \(d_s\) | SSM state size per channel (Mamba's \(N\)) |

## Reinforcement learning (Ch. 23–25)
| Symbol | Meaning |
|---|---|
| \(p_\theta\), \(p_{\text{ref}}\), \(p_{\theta_{\text{old}}}\) | policy (the model's distribution; never \(\pi\), which is peak FLOP/s); frozen reference policy; policy that generated the rollouts |
| \(x\), \(y=(y_1,\dots,y_{n_y})\) | prompt; sampled response of \(n_y\) tokens |
| \(s_t=(x,y_{&lt;t})\), \(a_t=y_t\) | state (prefix) and action (token) of the token-level MDP |
| \(\mathcal R(x,y)\), \(\mathcal R_t\) | sequence reward; per-step reward (zero except at the end, plus KL shaping) |
| \(J(\theta)\), \(J_{\text{KL}}\) | expected reward; KL-regularised objective |
| \(V^{p}\), \(Q^{p}\), \(A^{p}\); \(V_\psi\) | state-value, action-value and advantage functions of policy \(p\); learned critic (not the attention matrices \(V\), \(Q\)) |
| \(\hat A_t\), \(\delta_t\) | advantage estimate; TD error \(\mathcal R_t+\gamma_dV(s_{t+1})-V(s_t)\) |
| \(\gamma_d\), \(\lambda_{\text{GAE}}\) | discount factor (usually 1); GAE parameter (not the draft length \(\gamma\) or arrival rate \(\lambda\)) |
| \(\beta_{\text{KL}}\) | KL-penalty coefficient (not the bandwidth \(\beta\)) |
| \(\varrho_t\), \(\epsilon_{\text{clip}}\) | importance ratio \(p_\theta/p_{\theta_{\text{old}}}\) of token \(t\); PPO clipping range |
| \(\varrho^{\text{ref}}_t\), \(k_1,k_2,k_3\) | ratio \(p_{\text{ref}}/p_\theta\); the three per-token KL estimators |
| \(p^*\), \(Z(x)\) | optimum of the KL-regularised objective, \(p^*\propto p_{\text{ref}}e^{\mathcal R/\beta_{\text{KL}}}\); its normaliser |
| \(y_w\succ y_l\) | preferred ("winning") and rejected responses of a comparison (Ch. 24) |
| \(\mathcal R_\phi\) | learned reward model with parameters \(\phi\) |
| \(\sigma(u)\) | logistic function \(1/(1+e^{-u})\) (as a function; not a singular value) |
| \(\hat{\mathcal R}_\theta=\beta_{\text{KL}}\log(p_\theta/p_{\text{ref}})\) | DPO's implicit reward |
| \(h_\theta\) | log-ratio margin \(\log\frac{p_\theta(y_w)}{p_{\text{ref}}(y_w)}-\log\frac{p_\theta(y_l)}{p_{\text{ref}}(y_l)}\) (IPO) |
| \(m_{\text{SimPO}}\) | SimPO's target reward margin |
| \(n_{y,i}\) | length of response \(i\) of a group (Ch. 25) |
| \(P_{\text{info}}(p_s,n_s)\) | probability that a group of \(n_s\) samples contains both a success and a failure |
| \(\epsilon_{\text{low}},\epsilon_{\text{high}}\) | DAPO's decoupled lower and upper clipping ranges |
| \(n_{\text{stale}}\) | policy updates by which rollouts lag the trained policy in asynchronous RL |

## Reasoning and test-time compute (Ch. 26)
| Symbol | Meaning |
|---|---|
| \(n_s\) | number of independent samples drawn for one problem (pass@\(n_s\), voting, best-of-\(n_s\)) |
| \(p_s\) | probability that a single sample solves the problem |
| \(n_{\text{think}}\) | reasoning ("thinking") tokens generated before the answer |
| \(\mathcal R_i\), \(\hat A_i\) | reward of rollout \(i\); its group-relative advantage (GRPO) |

## Serving at scale (Ch. 27)
| Symbol | Meaning |
|---|---|
| \(n_r\) | number of replicas behind a router |
| \(u\) | utilisation: offered load divided by capacity |
| \(n_c\) | replicas sampled per routing decision (power of \(n_c\) choices) |
| \(W_{\downarrow}, W_{\uparrow}\), \(r_a\) | LoRA down- and up-projections and adapter rank; \(\Delta W=W_{\downarrow}W_{\uparrow}\) |
| \(c_{\text{gpu}}\) | price of one GPU-hour, in dollars |
| \(\Phi\), \(n_g\) | tokens per second produced by a replica at its operating point; GPUs per replica |

## Units
- Memory sizes of tensors: binary units (KiB, MiB, GiB). Datasheet capacities and bandwidths: decimal (GB, TB/s), as vendors quote them.
- FLOPs = floating-point operations (a count); FLOP/s = rate. A multiply-add counts as 2 FLOPs.
- "Dense" throughput figures never include the 2× structured-sparsity marketing factor.

---

# Volume II — Cloud AI Infrastructure

Volume II uses every symbol above with the same meaning and adds the following. Where a Volume I letter is reused, it carries a subscript that keeps it distinct (for example \(P_{\text{IT}}\) is a power, never the attention-probability matrix \(P\)).

## Capacity, power and cost (Vol. II Ch. 1–2, 9–10)
| Symbol | Meaning |
|---|---|
| \(\lambda_{\text{pk}}\), \(\bar\lambda\) | peak and daily-mean arrival rates; \(\lambda_{\text{pk}}/\bar\lambda\) is the peak-to-mean ratio |
| \(r_{\text{hit}}\) | prefix-cache hit rate: fraction of input tokens served from cache |
| \(\Phi_P\), \(\Phi_D\) | prefill and decode tokens per second a replica sustains at its operating point |
| \(n_{\text{gpu}}\) | number of accelerators in a fleet |
| \(s_{\text{spare}}\) | spare-capacity fraction (failures, draining, cold-start headroom, canaries) |
| \(P_{\text{gpu}}\) | all-in IT power per accelerator (its share of server, network and storage), kW |
| \(P_{\text{IT}}\), \(P_{\text{fac}}\) | IT power; total facility power. \(\text{PUE}=P_{\text{fac}}/P_{\text{IT}}\) |
| \(e_{\text{tok}}\), \(e_{\text{tok}}^{\text{fac}}\) | energy per token at the IT level, \(n_gP_{\text{gpu}}/\Phi\); at the facility meter (Ch. 2) |
| \(a_{\text{act}}\), \(C_{\text{sw}}\), \(V\) | switching activity, switched capacitance and supply voltage in \(P_{\text{dyn}}=a_{\text{act}}C_{\text{sw}}V^2f_{\text{clk}}\) (Ch. 2) |
| \(\mu_P\), \(\sigma_P\), \(\rho_P\) | mean, standard deviation and pairwise correlation of per-server power draw (Ch. 2) |
| \(z_\epsilon\) | upper \(\epsilon\)-quantile of the standard normal distribution |
| \(\dot m\), \(c_p\), \(\vartheta\) | coolant mass flow (kg/s), specific heat (J/(kg·K)), temperature (\(\Delta\vartheta\): temperature rise) (Ch. 2) |
| \(\ell\), \(n_{\text{rt}}\) | network path length; number of sequential round trips (Ch. 2, 12, 17) |
| \(f_{\text{use}}\), \(M_{\text{act}}\) | usable fraction of device memory; memory reserved for activations, graphs and workspace (Ch. 3) |
| \(B_{\max}\), \(\bar t\), \(t^{*}\) | largest batch that fits; mean context length of resident sequences; TPOT target (Ch. 3) |
| \(\beta_{\text{eff}}\) | effective bandwidth of a replica, \(n_g\cdot\text{MBU}\cdot\beta\) (Ch. 3) |
| \(\upsilon_i\) | fraction of time tenant \(i\) is active on a shared device (Ch. 3) |

## Networks (Vol. II Ch. 4)
| Symbol | Meaning |
|---|---|
| \(k_{\text{sw}}\) | switch radix (ports per switch) |
| \(H\), \(r\) | number of hosts on a fabric; link rate per host |
| \(n_f\), \(n_{\text{path}}\), \(c_{\max}\) | flows in a collective step; equal-cost paths; largest number of flows on one path |

## Storage and cold starts (Vol. II Ch. 5)
| Symbol | Meaning |
|---|---|
| \(t_{\text{cold}}\) | cold-start time of a replica \(=t_{\text{node}}+t_{\text{sched}}+t_{\text{image}}+t_{\text{weights}}+t_{\text{init}}+t_{\text{warm}}\) |
| \(W_{\text{bytes}}\) | bytes of model weights, \(Nb_w\) (distinct from the time in system \(W\) of Little's law) |
| \(n_{\text{conn}}\), \(\beta_{\text{conn}}\), \(\beta_{\text{NIC}}\), \(\beta_{\text{svc}}\) | concurrent storage requests; throughput of one; client NIC bandwidth; service-side limit |
| \(c\) | chunk size of a pipelined transfer or broadcast (bytes) |

## Orchestration (Vol. II Ch. 6–7)
| Symbol | Meaning |
|---|---|
| \(\lambda_f\) | failure rate of one node (or device), per hour |
| \(g\) | (Ch. 6–7 only, as a count) nodes in a multi-node replica group; elsewhere \(g\) is the GQA group size |
| \(t_{\text{rec}}\) | recovery time of a failed replica |
| \(\Gamma\), \(\Gamma_{\text{res}}\) | generation time of a request; residual (remaining) time of an in-flight request (not \(S\), the attention scores) |
| \(n_b\), \(t_{\text{drain}}\) | replicas replaced per rollout batch; time to drain a replica |
| \(c_j\), \(a_i\), \(n_{\text{res}}\) | free-capacity vector of node \(j\); demand vector of pod \(i\); number of resource types (Ch. 7) |
| \(\tau_c\), \(\delta_c\), \(M_{\text{bf}}\) | checkpoint interval; cost of one checkpoint; mean time between interruptions (Ch. 7) |
| \(z\), \(n_{\text{req}}\) | number of failure domains a tier is spread over; replicas required (Ch. 7, 13) |

## Serving platforms (Vol. II Ch. 8)
| Symbol | Meaning |
|---|---|
| \(n_{\text{q},j}\), \(n_{\text{hit},j}\) | prefill tokens queued at replica \(j\); length of the request's prefix already cached there |
| \(u^{\text{kv}}_j\), \(t_{\text{kv}}\), \(t_{\text{lora}}\) | KV-cache utilisation of replica \(j\); time penalties for KV pressure and adapter loading |
| \(m\), \(m^{*}\) | autoscaling metric per replica and its target |
| \(c_{\text{ded}}\), \(c_{\text{sls}}\), \(o\) | dedicated price per GPU-hour; serverless price per busy GPU-hour; cold-start overhead fraction |

## Capacity and procurement (Vol. II Ch. 9)
| Symbol | Meaning |
|---|---|
| \(D(\tau)\), \(T_{\text{per}}\) | GPU demand at time of day \(\tau\); length of the planning period |
| \(R\), \(c_r\), \(c_o\) | committed capacity (GPUs); committed and on-demand prices per GPU-hour |
| \(p_{\text{na}}\), \(c_{\text{short}}\), \(c_{\text{idle}}\), \(c_{\text{over}}\) | probability on-demand is unavailable; cost of a GPU-hour of shortfall; value of lending an idle GPU-hour; cost of an unused GPU-hour |
| \(\ell_q\), \(q\) | pinball loss; forecast quantile |
| \(\lambda_p\), \(L\) (as a job length, Ch. 9 only) | preemption rate per hour; uninterrupted work a job needs (elsewhere \(L\) is layers) |
| \(w\), \(t_{\text{node}}\), \(\dot\lambda\) | warm-pool size (GPUs; elsewhere \(w\) is the attention window); node provisioning time; demand growth rate |
