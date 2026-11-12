# Introduction to LLM Engineering

A free, interactive, web-native textbook on **LLM inference engineering**: KV caches, attention kernels, GPU architecture, batching and scheduling, quantisation, distributed and disaggregated serving, speculative decoding, and ML-systems design. It is written for readers with a strong mathematical background (PhD level) and data-science experience but little systems or software-engineering background, and it aims at FAANG / AI-lab interview depth.

**Read it:** https://jaswin90.github.io/LLM-Engineering/

- 📐 Equations rendered with KaTeX, diagrams as inline SVG, charts rendered from data in the page. No images.
- ✅ Interactive MCQs with explanations, plus graded interview questions with model answers.
- 🔐 Progress is saved in the browser; a no-account **sync link** (or export/import file) moves it between devices, and optional Google sign-in (Firebase, see `SETUP.md`) syncs it automatically.
- 📚 Primary-source references in every chapter and a consolidated [bibliography](docs/references.html).

## Repository layout

| Path | Contents |
|---|---|
| `BOOK_SKELETON.md` | The plan: 30 chapters in 9 parts, with each chapter's motivating question, arc and references, and a status table |
| `CLAUDE.md`, `NOTATION.md`, `REVIEW.md` | Authoring rules, the book-wide notation table, and the editorial review log |
| `docs/` | The website (GitHub Pages root) |
| `docs/chapters/ch01.html` … `ch30.html` | The 30 chapters (I Foundations; II GPUs; III Attention and memory; IV Serving systems; V Smaller and adapted models; VI Distributed inference; VII Faster decoding and longer contexts; VIII Post-training: RL, alignment and reasoning; IX Production). File names are stable identifiers, not display numbers: the four post-training chapters added later live in `ch27`–`ch30` (displayed as Chapters 16 and 23–25), and the order is defined by the `CHAPTERS` array in `docs/assets/js/book.js` |
| `docs/cloud/ci01.html` … `ci20.html` | Volume II, *Cloud AI Infrastructure*: 20 chapters in 6 parts (cover at `docs/cloud/index.html`) |
| `docs/assets/js/book.js` | Reader engine: TOC, theme, MCQ and Q&A components, progress store |
| `docs/assets/js/auth.js` | Google sign-in and Firestore sync |
| `docs/assets/js/charts.js` | Dependency-free SVG chart renderer |
| `.github/workflows/pages.yml` | Deploys `docs/` to GitHub Pages |

## Writing status

All 30 chapters are written (roughly 290,000 words including questions, 131 inline SVG diagrams, 71 data charts, 985 MCQs and 491 interview questions with model answers). Readers can sign in with an email and password (confirmed by email) or with Google, through the Firebase project configured in `docs/assets/js/firebase-config.js` (see [SETUP.md](SETUP.md)); without signing in, progress is kept in the browser and can be moved between devices with a sync link.

Volume II, *Cloud AI Infrastructure*, is written in a first version: 20 chapters, about 175,000 words, 81 diagrams, 42 charts, 623 MCQs and 320 interview questions. Several of its chapters are still below the 9,000-word floor (see `REVIEW.md`, Review 6).
