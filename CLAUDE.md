# CLAUDE.md — authoring instructions for *Introduction to LLM Engineering*

This repository is a **book**, published as a static website from `docs/` (GitHub Pages, auto-deployed by `.github/workflows/pages.yml` on every push that touches `docs/`). Read this file, `NOTATION.md`, `BOOK_SKELETON.md` and `REVIEW.md` before writing or editing any chapter.

## 1. Reader and promise
- Reader: PhD in mathematics, ~5 years of data science, little software or systems engineering. Assume fluency in linear algebra, probability and optimisation. Assume no knowledge of GPUs, operating systems, networking or C++.
- Goal: the reader can derive the cost of serving any LLM on any hardware, explain every major inference technique and why it appeared when it did, and pass FAANG / AI-lab ML-systems and inference-engineer interviews.

## 2. It must read like a book
1. **Prose first.** At least 75% of each chapter is continuous prose. Boxes, tables and bullet lists *support* the prose; they never replace it. A section that is mostly a table or a list is not finished.
2. **Paragraphs develop ideas.** A typical paragraph is 4–8 sentences: claim → reasoning → example → consequence.
3. **Every section follows the arc: motivation → idea → mathematics → worked example → consequences → bridge.**
   - *Motivation*: open with the problem, question or failure that makes the idea necessary. Never introduce a concept "because it exists".
   - *Bridge*: end with the new question the idea raises, which the next section answers.
4. **Never use a concept before it is motivated and defined.** If a forward reference is unavoidable, give an informal one-sentence definition and say which chapter makes it precise.
5. **Chapter opening.** Each chapter starts with an unnumbered section that motivates it: where we are in the story, the question this chapter answers, what the reader will be able to do afterwards (learning objectives), and what earlier chapters it uses.
6. **Chapter closing.** End with a "Looking ahead" section that hands off to the next chapter by naming the open problem it will solve. Then the summary, MCQs, interview questions and references.
7. **Tell the evolution.** Present techniques as a causal chain: bottleneck → idea → new bottleneck. Give dates. Say who did it and why it was needed *then*.
8. **Figures are discussed, not just captioned.** The body text must refer to every figure ("Figure 3.2 shows…") and walk the reader through what to notice.
9. **Software-engineering bridges.** When a systems concept appears (cache, process, kernel, queue, page table), explain it in mathematical language in a "For the mathematician" box.
10. **Tone.** Precise, warm, direct. Use "we" for derivations. No hype, no filler, no unexplained jargon. Use British spelling (quantisation, normalisation), as the existing chapters do.

## 3. Consistency
- **Notation:** use only the symbols in `NOTATION.md`. Add new symbols there before using them, and check for collisions. In particular: \(L\) = layers, \(\mathcal L\) = loss, \(\pi\) = peak FLOP/s, \(\beta\) = bandwidth, \(\Theta\) = RoPE base, \(T\) = sequence length, \(t\) = current position, \(m\) = rows of a matrix product.
- **Running examples** (use these unless a different model is the point): **Llama 3 8B** (dense, GQA), **Llama 3 70B** (large dense), **DeepSeek-V3** (MoE + MLA). Hardware: **H100 SXM** primary, plus A100 / H200 / B200 for comparison. Reuse the numbers already derived: Llama 3 8B has \(N\approx8.03\)B, 16.06 GB of BF16 weights, and \(s_{kv}=128\) KiB per token.
- Cross-reference sections as "§3.4" and chapters as "Chapter 5". Figures are numbered per chapter (Figure 3.2).

## 4. Required components per chapter
- Learning objectives; an opening motivation; numbered sections; at least 4 inline SVG diagrams and 2 data charts where the topic allows; at least 2 worked examples and 2 drills (`details.drill`); "Looking ahead"; summary.
- **MCQs:** 30 or more per chapter (`div.mcq` with `data-id`, `data-topic` naming the section such as "§3.5 Roofline", options in `ol.opts`, correct options marked `data-correct`, and an explanation in `.explain`).
- **Interview questions:** 15 or more per chapter (`details.qa` with `data-id`, a level pill `warm|core|deep|design`, and a model answer in `.answer`).
- **References:** primary sources with arXiv links at the end of the chapter, cited inline with `<span class="cite">`. Add new key sources to `docs/references.html`.
- **Length:** 9,000–14,000 words per chapter including questions (≈ 18–28 printed pages).

## 5. Technical conventions
- Chapters live in `docs/chapters/chNN.html` and are registered in the `CHAPTERS` array in `docs/assets/js/book.js` (set `ready: true` and `file`).
- Math: KaTeX (vendored in `docs/assets/vendor/katex`). Use `\( … \)` for inline and `\[ … \]` for display. Do **not** use `$` delimiters, because `$` is reserved for prices. Escape `<` as `&lt;` inside math.
- Diagrams: inline `<svg class="dg">`, using the CSS classes in `book.css` (`bx`, `bx2`…, `ln`, `t-sm`…) so they work in light and dark themes. Charts: `figure.chart` with a JSON `script.chart-spec` (see `docs/assets/js/charts.js`). Use colour slots `s1`, `s2`, `s3` first.
- Never rename existing `data-id`s of questions that have been published, because readers' saved progress is keyed on them. Moving a question to another chapter is the exception: give it a new id.

## 6. Before pushing
1. Serve `docs/` locally (`python3 -m http.server`) and run a Playwright check on every changed page: no page errors, no `.katex-error`, no horizontal overflow at 390 px width, and all charts rendered. Look at screenshots of new figures for label collisions. Keep screenshots in the scratchpad, never in the repo.
2. Update `BOOK_SKELETON.md` (status table) and add a `REVIEW.md` entry for any editorial pass.
3. Commit, then push to the working branch. The Pages workflow deploys automatically, and the site is at https://jaswin90.github.io/LLM-Engineering/.
