# Editorial self-review log

This file records weaknesses found in the drafts and what was done about them. Every revision pass adds an entry. Rules that come out of a review go into `CLAUDE.md` so they apply to every future chapter.

---

## Review 1 — Draft 1 of Chapters 1–2 (Nov 2026)

### Weaknesses found

| # | Weakness | Evidence in Draft 1 | Severity |
|---|---|---|---|
| W1 | **Reads like lecture notes, not a book.** Roughly 40% of each chapter was boxes, tables and bullet lists. Paragraphs were 2–4 sentences. Ideas were stated without being developed. | Ch 1 §1.10 was almost entirely a table. Ch 2 §2.2 was three bullets and a table. | High |
| W2 | **Concepts used before they were motivated or defined.** | Ch 1 §1.12 used "ridge point" and "FLOPs per byte" before defining them. Ch 1 §1.9 used "prefill" and "decode" four sections before defining them. Ch 2 brought in the roofline (a hardware idea) in a chapter about the model, without first saying what HBM or a tensor core is. | High |
| W3 | **Inconsistent and colliding notation.** | β meant HBM bandwidth, the RoPE base *and* a Chinchilla exponent. P meant peak FLOP/s *and* the attention-probability matrix. L meant layers, loss *and* Little's-law occupancy. B meant batch size *and* a Chinchilla constant. M meant the causal mask *and* the GEMM row count. m, n meant RoPE positions *and* matrix dimensions. h meant the RNN state *and* the normalised input. T and t were used interchangeably for sequence length. | High |
| W4 | **The history was compressed and the narrative broke off.** | n-grams, neural LMs and RNNs got one paragraph each. After §1.9 the "evolution" turned into tables, and the reader never saw *why* each systems idea followed from the last. | Medium |
| W5 | **Chapter 2 was overloaded.** | It covered the architecture, generation, the KV cache, the roofline, a latency model and code, so no topic got the space it needed. | High |
| W6 | **No running example.** | Numbers jumped between Llama 2 7B/70B, Llama 3 8B, Mistral and DeepSeek. The reader could not build intuition about one model. | Medium |
| W7 | **No chapter openings or closing bridges.** | Chapters began straight at §x.1 and ended at the summary, with no "why this chapter" and no hand-off to the next. | Medium |
| W8 | **Some claims were unsourced.** | ChatGPT's user numbers and the output/input price ratio had no citation. | Low |
| W9 | **Figures were captioned but not discussed.** | Several figures were only referenced through their captions. The body text never walked the reader through them. | Medium |
| W10 | **Quizzes were not tied to sections.** | MCQs were grouped only at the end, with no pointer to the section they test. | Low |

### Research done for the revision
- Search summaries of: Aleksa Gordić, *Inside vLLM: Anatomy of a High-Throughput LLM Inference System* (2025), which goes broad first and then layers in detail, built around a running example script; Horace He, *Making Deep Learning Go Brrrr From First Principles* (2022), which splits runtime into three regimes (compute, memory bandwidth, overhead) and explains operator fusion; Lilian Weng, *Large Transformer Model Inference Optimization* (2023); kipply, *Transformer Inference Arithmetic* (2022), which teaches by counting from few principles; *How to Scale Your Model*, Ch. 7 "All About Transformer Inference" (2025), which builds up from sampling one token to serving across many accelerators; and the BentoML LLM inference handbook.
- Direct page fetches of these sites were blocked by the build sandbox's network policy, so the lessons above come from search summaries and prior familiarity with the texts.

### Rectification
1. **`CLAUDE.md`** now holds standing writing instructions: book-style prose, a motivation-first arc for every section, prose kept above 75% of the text, and the rules for chapter openings, bridges and figures.
2. **`NOTATION.md`** is the single source of truth for symbols, and the site has a matching notation page. All collisions above are resolved: hardware peak is π, bandwidth is β, the RoPE base is Θ, loss is 𝓛, the Chinchilla constants are c_N, c_D with exponents α_N, α_D, the GEMM row count is m, attention positions are i and j, the normalised input is X̃, Little's-law occupancy is n̄, and T vs t is defined (T = sequence length, t = current position).
3. **Restructure.** Old Ch 2 is split into **Ch 2 *The Transformer, Dissected*** (what is computed) and **Ch 3 *Running the Model: Prefill, Decode and the KV Cache*** (what it costs to run). The book now has 26 chapters.
4. **Running examples** are fixed for the whole book. Models: Llama 3 8B (dense, GQA), Llama 3 70B (large dense) and DeepSeek-V3 (MoE + MLA). Hardware: H100 SXM, with A100, H200 and B200 for comparison.
5. **Chapter 1 rewritten.** The history is expanded into continuous narrative; hardware intuition comes before any FLOPs-per-byte argument; prefill and decode are introduced informally when first needed; the systems history is told as a causal chain.
6. **Chapters 2 and 3 rewritten** to the new standard. Horace He's compute/memory/overhead lens now motivates arithmetic intensity, and a complete numerical "life of a request" ties Chapter 3 together.
7. Every chapter now opens with a motivation section and closes with a "Looking ahead" bridge. Every figure is discussed in the body text. Quizzes point back to their sections.

---

## Review 2 — Correction found while writing Chapter 7 (Nov 2026)
- **Error:** Chapter 5's tiling box claimed that shared-memory tiling alone raises a GEMM's HBM intensity "comfortably above" the H100 ridge. A \(t\times t\) block tile gives intensity ≈ \(t/b\) (≈ 85 FLOP/B for 128 × 256 BF16 tiles), which is below the ridge. Fast GEMMs also rely on L2 reuse across concurrently scheduled blocks (rasterisation/swizzling).
- **Fix:** rewrote the Ch 5 box and interview answer ch05-qa-08 to give the correct two-level argument; Ch 7 now derives it with a chart.
- **Rule added to practice:** recompute every quantitative claim from the formula before publishing, even "obvious" ones.

## Review 3 — Part IV notation and figure pass (Chapters 12–13)
- **Notation collision (Ch 12).** §12.2 used \(g\) and \(c\) for GPU and CPU step times; \(g\) is the GQA group size. Renamed to \(t_{\text{gpu}}\), \(t_{\text{cpu}}\) throughout the chapter and its questions, and added them to `NOTATION.md`, with the Chapter 13 symbols \(G\), \(q\), \(\beta_{\text{net}}\), \(\chi\), \(n_P\), \(n_D\).
- **Figure legibility.** Labels drawn on saturated bars (Figures 12.3, 13.1, 13.4, 13.9) were low-contrast; added `t-on` / `t-on-dark` text classes in `book.css`. KaTeX does not render inside SVG `<text>`; Figure 13.2 uses `<tspan>` subscripts instead. Rule: never put `\( … \)` inside an SVG.
- **Honest framing of disaggregation (Ch 13).** Headline speed-ups from Splitwise and DistServe are reported with their baselines, and §13.6 proves the colocation bound so that readers see which assumptions disaggregation exploits.
- **SVG fill classes (Ch 14).** `fill-s1` does not exist in `book.css` and rendered black; use `fill-acc` for the primary colour. Wide tables must sit inside `div.table-wrap` (a five-column table overflowed at 390 px).
- **Structural check (Ch 16).** A scripted replacement of a theorem box matched the first `</div>` (the box title's) and left a stray closing tag, which pushed later figures outside the article column at desktop width. Added a desktop check that every `figure` stays inside `article`; long inline formulas moved to display maths so their hidden MathML does not widen the page.

## Review 4 — component minimums in early chapters (2026-11-06)

A count against CLAUDE.md §4 found early chapters below the per-chapter minimums: Chapters 4 and 5 had no drills, Chapters 2, 7 and 8 had one, and Chapters 7, 9 and 10 had no worked examples. Added:
- Ch2: Drill 2.2 (parameter count of Llama 3 70B).
- Ch4: Drill 4.1 (temperature and top-p by hand), Drill 4.2 (cost of grammar masks).
- Ch5: Drill 5.1 (peak FLOP/s from SM count), Drill 5.2 (HBM bandwidth from bus width).
- Ch7: worked examples on uncoalesced reads and launch overhead; Drill 7.2 (tile arithmetic intensity).
- Ch8: Drill 8.2 (MLA cache vs MHA per token for DeepSeek-V3).
- Ch9: worked examples on standard-attention HBM traffic and split-KV decoding.
- Ch10: worked examples on max-length reservation waste and prefix-cache savings.
Whole-book Playwright check re-run: no errors, 856 MCQs, 427 QAs. Added Figure 6.8 (roofline floor to measured decode time) and Figure 11.7 (request lifecycle state machine), so every chapter now has at least four SVG diagrams.
- Second worked examples for the eight chapters that had one: Ch3 (Llama 3 70B on 8×H100), Ch4 (cost of sampling on the GPU), Ch8 (FP8 cache at 32k), Ch11 (choosing the chunk size), Ch16 (MoE all-to-all for DeepSeek-V3), Ch20 (which RoPE pairs Llama 3.1 rescales), Ch24 (sample size for a p99), Ch26 (plausibility of a $0.10/M price). Each was cross-checked against the conventions of its chapter (80% bandwidth, 50% MFU, $2.50 per GPU-hour, α–β values of Ch16). Every chapter now meets the §4 minimums of two worked examples, two drills and four SVG diagrams.
- Third worked examples for the six chapters closest to the 9,000-word floor: Ch17 (a 512k-token prompt and context parallelism), Ch19 (prompt lookup for a code edit), Ch21 (checkpoint spacing for recurrent-state prefix caching), Ch22 (RL weight synchronisation), Ch23 (cache-affinity routing), Ch25 (an MoE model streamed from flash). Self-check caught one arithmetic slip before publishing (0.95⁹ gives 7.4 tokens per step, not 7.0).

## Review 5 — post-training added to the flow (2026-11-06)

A gap was found: the book covered serving LoRA adapters (§27.4) and named GRPO and PPO (§26.7), but never explained fine-tuning, parameter-efficient methods, reinforcement learning, RLHF, DPO or GRPO, all of which AI-lab interviews ask about. Four chapters are being inserted where the story needs them, not appended:
- **Ch 16, Fine-Tuning and Parameter-Efficient Adaptation**, ends Part V (renamed "Making Models Smaller and Adapting Them"): QLoRA builds on Ch 14, and LoRA sets up multi-LoRA serving in Ch 27.
- **New Part VIII, Post-Training**: Ch 23 (RL for LLMs, to PPO), Ch 24 (preferences: RLHF, DPO), Ch 25 (verifiable rewards: GRPO), followed by Reasoning Models, now Ch 26, because reasoning models are the product of that RL. Production becomes Part IX (Ch 27–30).

Renumbering rules: only displayed numbers change. File names (`chNN.html`), question `data-id`s and element ids are unchanged, so URLs and readers' saved progress survive; the new chapters use files ch27–ch30. Every cross-reference form (§x.y, Figure/Drill x.y, headings, "Chapter(s) …" lists, "Ch N", Part labels) was rewritten by one scripted pass over the sources, then all chapters were rebuilt and diffed against the same pass applied to the published pages (identical except Chapter 1, whose roadmap figure and plan paragraph were rewritten for nine parts). The hand-offs into the new chapters (end of Ch 15 and Ch 22) were rewritten.
- Ch 16 (Fine-Tuning and Parameter-Efficient Adaptation) written: SFT and loss masking, the 16-byte bill, ZeRO/FSDP, the low-rank hypothesis, LoRA with its gradient derived as projected descent, rank/targets/learning rates, QLoRA, the other PEFT methods with their inference costs, and multi-adapter serving economics. Self-checks fixed one arithmetic slip (pre-training FLOPs of Llama 3 8B are 7.2×10²³, five orders above the fine-tune, not 1.2×10²⁴), five wrong section references, two chart labels colliding with lines, and one long inline formula moved to display.
