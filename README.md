# Introduction to LLM Engineering

A free, interactive, web-native textbook on **LLM inference engineering**: KV caches, attention kernels, GPU architecture, batching and scheduling, quantisation, distributed and disaggregated serving, speculative decoding, and ML-systems design. It is written for readers with a strong mathematical background (PhD level) and data-science experience but little systems or software-engineering background, and it aims at FAANG / AI-lab interview depth.

**Read it:** https://jaswin90.github.io/LLM-Engineering/

- 📐 Equations rendered with KaTeX, diagrams as inline SVG, charts rendered from data in the page. No images.
- ✅ Interactive MCQs with explanations, plus graded interview questions with model answers.
- 🔐 Optional Google sign-in (Firebase) keeps each reader's history across devices.
- 📚 Primary-source references in every chapter and a consolidated [bibliography](docs/references.html).

## Repository layout

| Path | Contents |
|---|---|
| `BOOK_SKELETON.md` | The plan: 26 chapters in 8 parts, with each chapter's motivating question, arc and references, and a status table |
| `CLAUDE.md`, `NOTATION.md`, `REVIEW.md` | Authoring rules, the book-wide notation table, and the editorial review log |
| `docs/` | The website (GitHub Pages root) |
| `docs/chapters/ch01.html` … `ch26.html` | The 26 chapters (I Foundations; II GPUs; III Attention and memory; IV Serving systems; V Smaller models; VI Distributed inference; VII Faster decoding and longer contexts; VIII Production) |
| `docs/assets/js/book.js` | Reader engine: TOC, theme, MCQ and Q&A components, progress store |
| `docs/assets/js/auth.js` | Google sign-in and Firestore sync |
| `docs/assets/js/charts.js` | Dependency-free SVG chart renderer |
| `.github/workflows/pages.yml` | Deploys `docs/` to GitHub Pages |

## Writing status

All 26 chapters are written (roughly 230,000 words including questions, more than 110 inline SVG diagrams and about 60 data charts, 856 MCQs and 427 interview questions with model answers). Google sign-in needs a Firebase project configured in `docs/assets/js/firebase-config.js` (see [SETUP.md](SETUP.md)); without it, progress is kept in the browser's local storage.
