# Introduction to LLM Engineering

A free, interactive, web-native textbook on **LLM inference engineering**: KV caches, attention kernels, GPU architecture, batching and scheduling, quantisation, distributed and disaggregated serving, speculative decoding, and ML-systems design. It is written for readers with a strong mathematical background (PhD level) and data-science experience but little systems or software-engineering background, and it aims at FAANG / AI-lab interview depth.

**Read it:** https://jaswin90.github.io/LLM-Engineering/ (after enabling Pages; see [SETUP.md](SETUP.md))

- 📐 Equations rendered with KaTeX, diagrams as inline SVG, charts rendered from data in the page. No images.
- ✅ Interactive MCQs with explanations, plus graded interview questions with model answers.
- 🔐 Optional Google sign-in (Firebase) keeps each reader's history across devices.
- 📚 Primary-source references in every chapter and a consolidated [bibliography](docs/references.html).

## Repository layout

| Path | Contents |
|---|---|
| `BOOK_SKELETON.md` | Full plan: 25 chapters in 8 parts plus appendices, with each chapter's objectives, sections, diagrams, MCQ themes, interview questions and references |
| `docs/` | The website (GitHub Pages root) |
| `docs/chapters/ch01.html` | Ch. 1: From Shannon to Serving (history, scaling, inference economics, metrics) |
| `docs/chapters/ch02.html` | Ch. 2: The Transformer, Dissected for Inference (shapes, parameters, FLOPs, KV cache, roofline) |
| `docs/assets/js/book.js` | Reader engine: TOC, theme, MCQ and Q&A components, progress store |
| `docs/assets/js/auth.js` | Google sign-in and Firestore sync |
| `docs/assets/js/charts.js` | Dependency-free SVG chart renderer |
| `.github/workflows/pages.yml` | Deploys `docs/` to GitHub Pages |

## Writing status

Chapters 1–2 are drafted (about 40 printed pages, 67 MCQs, 34 interview questions). Chapters 3–25 follow the skeleton.
