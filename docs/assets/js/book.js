/* Introduction to LLM Engineering — core reader engine.
 * Classic script (works from file:// too). Provides:
 *   - chapter registry, header/sidebar/TOC, theme
 *   - progress store (localStorage) with a sync hook used by auth.js
 *   - interactive MCQs and interview Q&A with self-rating
 *   - KaTeX rendering
 */
(function () {
  'use strict';

  const CHAPTERS = [
    { id: 'ch01', n: 1, part: 'I', title: 'From Shannon to Serving', sub: 'The history of language modelling as a chain of bottlenecks, and why inference became the hard problem', file: 'chapters/ch01.html', ready: true },
    { id: 'ch02', n: 2, part: 'I', title: 'The Transformer, Dissected', sub: 'Tokenisation, attention, RoPE, normalisation, SwiGLU; counting parameters, FLOPs and bytes', file: 'chapters/ch02.html', ready: true },
    { id: 'ch03', n: 3, part: 'I', title: 'Running the Model: Prefill, Decode and the KV Cache', sub: 'The KV cache, compute vs memory vs overhead, the roofline, and latency models for TTFT and TPOT', file: 'chapters/ch03.html', ready: true },
    { id: 'ch04', n: 4, part: 'I', title: 'Decoding: From Logits to Text', sub: 'Greedy, beam, temperature, top-k/top-p/min-p, GPU sampling, structured output, determinism', file: 'chapters/ch04.html', ready: true },
    { id: 'ch05', n: 5, part: 'II', title: 'GPU Architecture from First Principles', sub: 'SIMT, SMs, tensor cores, HBM, NVLink; Volta → Blackwell → Rubin; TPUs and others', file: 'chapters/ch05.html', ready: true },
    { id: 'ch06', n: 6, part: 'II', title: 'The Roofline Model and Performance Analysis', sub: 'Hierarchical rooflines, critical batch size, MFU/MBU, profiling', file: 'chapters/ch06.html', ready: true },
    { id: 'ch07', n: 7, part: 'II', title: 'GPU Programming for Mathematicians', sub: 'CUDA, Triton, tiling, online softmax, fusion, CUDA graphs', file: 'chapters/ch07.html', ready: true },
    { id: 'ch08', n: 8, part: 'III', title: 'The KV Cache: Shrinking It', sub: 'MQA, GQA, MLA, sliding windows, KV quantisation, eviction', file: 'chapters/ch08.html', ready: true },
    { id: 'ch09', n: 9, part: 'III', title: 'IO-Aware Attention: FlashAttention 1 → 4', sub: 'Tiling, online softmax, FlashDecoding, FlashInfer, FlashMLA', file: 'chapters/ch09.html', ready: true },
    { id: 'ch10', n: 10, part: 'III', title: 'Memory Management: PagedAttention and Prefix Caching', sub: 'Block tables, copy-on-write, radix trees, KV hierarchies', file: 'chapters/ch10.html', ready: true },
    { id: 'ch11', n: 11, part: 'IV', title: 'Batching and Scheduling', sub: 'Continuous batching, chunked prefill, preemption, SLO-aware scheduling', file: 'chapters/ch11.html', ready: true },
    { id: 'ch12', n: 12, part: 'IV', title: 'Inside an Inference Engine', sub: 'vLLM, SGLang, TensorRT-LLM, TGI, llama.cpp', file: 'chapters/ch12.html', ready: true },
    { id: 'ch13', n: 13, part: 'IV', title: 'Disaggregated Serving', sub: 'Splitwise, DistServe, Mooncake, Dynamo, KV transfer', file: 'chapters/ch13.html', ready: true },
    { id: 'ch14', n: 14, part: 'V', title: 'Quantisation', sub: 'FP8, INT4, MXFP4/NVFP4, GPTQ, AWQ, SmoothQuant, rotations', file: 'chapters/ch14.html', ready: true },
    { id: 'ch15', n: 15, part: 'V', title: 'Pruning, Sparsity and Distillation', sub: 'SparseGPT, Wanda, 2:4, Minitron, distillation', file: 'chapters/ch15.html', ready: true },
    { id: 'ch27', n: 16, part: 'V', title: 'Fine-Tuning and Parameter-Efficient Adaptation', sub: 'Full fine-tuning, LoRA, QLoRA, DoRA, adapters', file: 'chapters/ch27.html', ready: true },
    { id: 'ch16', n: 17, part: 'VI', title: 'Distributed Computing Fundamentals', sub: 'Collectives, the α–β model, ring all-reduce, NVLink, InfiniBand, NCCL', file: 'chapters/ch16.html', ready: true },
    { id: 'ch17', n: 18, part: 'VI', title: 'Parallelism Strategies for Inference', sub: 'Tensor, pipeline, sequence and context parallelism', file: 'chapters/ch17.html', ready: true },
    { id: 'ch18', n: 19, part: 'VI', title: 'Mixture of Experts at Inference', sub: 'Routing, expert parallelism, all-to-all, the DeepSeek-V3 deployment', file: 'chapters/ch18.html', ready: true },
    { id: 'ch19', n: 20, part: 'VII', title: 'Speculative Decoding', sub: 'Exactness proof, Medusa, EAGLE 1–3, MTP, tree verification', file: 'chapters/ch19.html', ready: true },
    { id: 'ch20', n: 21, part: 'VII', title: 'Long-Context Inference', sub: 'RoPE scaling, sparse attention, context parallelism', file: 'chapters/ch20.html', ready: true },
    { id: 'ch21', n: 22, part: 'VII', title: 'Beyond Attention: SSMs and Hybrids', sub: 'Linear attention, Mamba, RWKV, hybrid models', file: 'chapters/ch21.html', ready: true },
    { id: 'ch28', n: 23, part: 'VIII', title: 'Reinforcement Learning for Language Models', sub: 'Policy gradients, advantages, KL control, PPO', file: 'chapters/ch28.html', ready: true },
    { id: 'ch29', n: 24, part: 'VIII', title: 'Learning from Human Preferences', sub: 'Reward models, RLHF, DPO and its variants', file: 'chapters/ch29.html', ready: true },
    { id: 'ch30', n: 25, part: 'VIII', title: 'Reinforcement Learning with Verifiable Rewards', sub: 'GRPO, DAPO, process rewards, rollout systems', file: 'chapters/ch30.html', ready: true },
    { id: 'ch22', n: 26, part: 'VIII', title: 'Reasoning Models and Test-Time Compute', sub: 'o1/R1, inference scaling, long-output serving', file: 'chapters/ch22.html', ready: true },
    { id: 'ch23', n: 27, part: 'IX', title: 'Serving at Scale', sub: 'Routing, autoscaling, multi-LoRA, cost modelling', file: 'chapters/ch23.html', ready: true },
    { id: 'ch24', n: 28, part: 'IX', title: 'Benchmarking and Observability', sub: 'Methodology, MLPerf, tracing, quality regressions', file: 'chapters/ch24.html', ready: true },
    { id: 'ch25', n: 29, part: 'IX', title: 'On-Device and Edge Inference', sub: 'llama.cpp, GGUF, MLX, NPUs', file: 'chapters/ch25.html', ready: true },
    { id: 'ch26', n: 30, part: 'IX', title: 'The ML-Systems Design Interview', sub: 'Worked case studies and rubrics', file: 'chapters/ch26.html', ready: true }
  ];
  const PARTS = { I: 'Foundations: What We Run and Why It Is Expensive', II: 'The Machine: Hardware for Mathematicians', III: 'Attention and Memory', IV: 'Serving Systems', V: 'Making Models Smaller and Adapting Them', VI: 'Distributed Inference', VII: 'Faster Decoding and Longer Contexts', VIII: 'Post-Training: Reinforcement Learning, Alignment and Reasoning', IX: 'Production' };


  /* Volume II: Cloud AI Infrastructure. Ids use the prefix "ci" so that saved progress never collides with Volume I. */
  const CLOUD_CHAPTERS = [
    { id: 'ci01', n: 1, part: 'I', title: 'Why Serving Became a Cloud Problem', sub: 'The stack from silicon to region, capacity as the scarce resource, and the map of this volume', file: 'cloud/ci01.html', ready: true },
    { id: 'ci02', n: 2, part: 'I', title: 'The AI Datacenter: Power, Cooling and Space', sub: 'Watts per token, PUE, liquid cooling, rack density and the grid', file: 'cloud/ci02.html', ready: true },
    { id: 'ci03', n: 3, part: 'I', title: 'Accelerators as Cloud Products', sub: 'Instance families, GPU sharing (MIG, MPS), TPUs and Trainium, price–performance', file: 'cloud/ci03.html', ready: true },
    { id: 'ci04', n: 4, part: 'I', title: 'Datacenter Networks for AI', sub: 'Front-end and back-end fabrics, fat trees, rail-optimised designs, RDMA, congestion', file: 'cloud/ci04.html', ready: true },
    { id: 'ci05', n: 5, part: 'I', title: 'Storage, Weights and Cold Starts', sub: 'Object storage, safetensors, weight streaming, caching tiers, peer-to-peer distribution', file: 'cloud/ci05.html', ready: true },
    { id: 'ci06', n: 6, part: 'II', title: 'Containers and Kubernetes for Accelerators', sub: 'Processes, cgroups, pods, device plugins, DRA, the GPU operator', file: 'cloud/ci06.html', ready: true },
    { id: 'ci07', n: 7, part: 'II', title: 'Scheduling a GPU Cluster', sub: 'Bin packing, gang and topology-aware scheduling, quotas, preemption, fragmentation', file: 'cloud/ci07.html', ready: true },
    { id: 'ci08', n: 8, part: 'II', title: 'Serving Platforms and Control Planes', sub: 'KServe, Ray Serve, Triton, Dynamo, llm-d, managed endpoints and serverless GPUs', file: 'cloud/ci08.html', ready: true },
    { id: 'ci09', n: 9, part: 'III', title: 'Capacity, Procurement and Autoscaling', sub: 'Reserved, on-demand and spot capacity, forecasting, capacity mix as an optimisation', file: 'cloud/ci09.html', ready: true },
    { id: 'ci10', n: 10, part: 'III', title: 'The Economics of Inference', sub: 'TCO, build versus buy, pricing tokens, FinOps and cost attribution', file: 'cloud/ci10.html', ready: true },
    { id: 'ci11', n: 11, part: 'IV', title: 'The Front Door: Gateways, Streaming and Rate Limits', sub: 'HTTP/2, SSE, gRPC, WebSockets, distributed rate limiting, admission control', file: 'cloud/ci11.html', ready: false },
    { id: 'ci12', n: 12, part: 'IV', title: 'Multi-Region Serving and Disaster Recovery', sub: 'Global load balancing, data residency, failover, capacity across regions', file: 'cloud/ci12.html', ready: false },
    { id: 'ci13', n: 13, part: 'IV', title: 'Reliability Engineering for AI Services', sub: 'SLOs and error budgets, GPU failure modes, MTBF at scale, safe rollouts, incidents', file: 'cloud/ci13.html', ready: false },
    { id: 'ci14', n: 14, part: 'IV', title: 'Security, Isolation and Multi-Tenancy', sub: 'Threat models, confidential computing, cache side channels, weight protection, supply chain', file: 'cloud/ci14.html', ready: false },
    { id: 'ci15', n: 15, part: 'V', title: 'Serving Multimodal Models', sub: 'Vision encoders, image tokens, encoder caching, speech and real-time audio, video', file: 'cloud/ci15.html', ready: false },
    { id: 'ci16', n: 16, part: 'V', title: 'Embeddings, Retrieval and Vector Search at Scale', sub: 'Embedding serving, HNSW, IVF-PQ, sharded indexes, RAG latency budgets', file: 'cloud/ci16.html', ready: false },
    { id: 'ci17', n: 17, part: 'V', title: 'Agents and Compound AI Systems', sub: 'Tool calls, MCP, code sandboxes, session state, program-aware scheduling', file: 'cloud/ci17.html', ready: false },
    { id: 'ci18', n: 18, part: 'V', title: 'Guardrails and Moderation Infrastructure', sub: 'Safety classifiers, streaming moderation, guard-model placement, audit', file: 'cloud/ci18.html', ready: false },
    { id: 'ci19', n: 19, part: 'V', title: 'Batch, Offline and Rollout Inference', sub: 'Batch APIs, throughput-optimal pipelines, spot-tolerant jobs, RL rollouts', file: 'cloud/ci19.html', ready: false },
    { id: 'ci20', n: 20, part: 'VI', title: 'The Cloud Infrastructure Design Interview', sub: 'Worked case studies, estimation drills and rubrics', file: 'cloud/ci20.html', ready: false }
  ];
  const CLOUD_PARTS = { I: 'The Physical Cloud', II: 'Orchestration', III: 'Elasticity and Economics', IV: 'The Global Request Path', V: 'Beyond Chat: Workloads Volume I Left Open', VI: 'The Interview' };
  const BOOKS = {
    llm: { key: 'llm', vol: 'Volume I', brand: 'Introduction to <span>LLM Engineering</span>', name: 'Introduction to LLM Engineering', index: 'index.html', chapters: CHAPTERS, parts: PARTS },
    cloud: { key: 'cloud', vol: 'Volume II', brand: 'Cloud AI <span>Infrastructure</span>', name: 'Cloud AI Infrastructure', index: 'cloud/index.html', chapters: CLOUD_CHAPTERS, parts: CLOUD_PARTS }
  };
  const ALL_CHAPTERS = CHAPTERS.map(c => Object.assign({ book: 'llm' }, c)).concat(CLOUD_CHAPTERS.map(c => Object.assign({ book: 'cloud' }, c)));
  const bookKey = () => (document.body && document.body.getAttribute('data-book')) || 'llm';
  const BOOK = () => BOOKS[bookKey()] || BOOKS.llm;

  /* ------------------------------------------------------------------ store */
  const KEY = 'llmeng-progress-v1';
  function safeGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function safeSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } }
  function blank() { return { v: 1, items: {}, chapters: {}, updatedAt: 0 }; }

  const store = {
    data: blank(),
    listeners: [],
    load() {
      const raw = safeGet(KEY);
      if (raw) { try { const d = JSON.parse(raw); if (d && d.items) this.data = Object.assign(blank(), d); } catch (e) {} }
    },
    save() { this.data.updatedAt = Date.now(); safeSet(KEY, JSON.stringify(this.data)); },
    get(id) { return this.data.items[id]; },
    set(id, rec) {
      rec.ts = Date.now();
      this.data.items[id] = rec; this.save();
      this.emit({ kind: 'item', id, rec });
    },
    del(id) { delete this.data.items[id]; this.save(); this.emit({ kind: 'item', id, rec: null }); },
    chapter(ch) { return this.data.chapters[ch] || {}; },
    setChapter(ch, patch) {
      const c = Object.assign({}, this.data.chapters[ch] || {}, patch, { ts: Date.now() });
      this.data.chapters[ch] = c; this.save();
      this.emit({ kind: 'chapter', id: ch, rec: c });
    },
    onChange(fn) { this.listeners.push(fn); },
    emit(ev) { this.listeners.forEach(fn => { try { fn(ev); } catch (e) { console.error(e); } }); },
    /* Merge a remote snapshot: newest timestamp wins per record. Returns merged data. */
    merge(remote) {
      if (!remote) return this.data;
      const out = blank();
      ['items', 'chapters'].forEach(k => {
        const a = this.data[k] || {}, b = remote[k] || {};
        new Set([...Object.keys(a), ...Object.keys(b)]).forEach(id => {
          const x = a[id], y = b[id];
          out[k][id] = !x ? y : !y ? x : ((y.ts || 0) > (x.ts || 0) ? y : x);
        });
      });
      this.data = out; this.save();
      this.emit({ kind: 'reload' });
      return out;
    },
    reset() { this.data = blank(); this.save(); this.emit({ kind: 'reload' }); }
  };
  store.load();

  /* ------------------------------------------------------------------ helpers */
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  function el(tag, attrs, html) {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) { if (k === 'class') e.className = attrs[k]; else e.setAttribute(k, attrs[k]); }
    if (html != null) e.innerHTML = html;
    return e;
  }
  let toastTimer;
  function toast(msg) {
    let t = $('.toast'); if (!t) { t = el('div', { class: 'toast', role: 'status' }); document.body.appendChild(t); }
    t.textContent = msg; t.classList.add('show');
    clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), 2400);
  }
  const root = () => document.body.getAttribute('data-root') || './';
  const chapterId = () => document.body.getAttribute('data-chapter');
  function snippet(node, n) { const s = (node.textContent || '').replace(/\s+/g, ' ').trim(); return s.length > n ? s.slice(0, n - 1) + '…' : s; }

  /* ------------------------------------------------------------------ theme */
  function setTheme(t) {
    if (t) document.documentElement.setAttribute('data-theme', t); else document.documentElement.removeAttribute('data-theme');
    safeSet('llmeng-theme', t || '');
  }
  function currentTheme() {
    const t = document.documentElement.getAttribute('data-theme');
    if (t) return t;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  /* ------------------------------------------------------------------ header */
  function buildHeader() {
    const r = root(), bk = BOOK(), other = bk.key === 'llm' ? BOOKS.cloud : BOOKS.llm;
    const bar = el('header', { class: 'topbar' });
    bar.innerHTML =
      '<button class="btn icon-btn" id="toc-toggle" aria-label="Contents">☰</button>' +
      '<a class="brand" href="' + r + bk.index + '">' + bk.brand + '</a>' +
      '<span class="spacer"></span>' +
      '<nav><a href="' + r + bk.index + '#contents">Contents</a><a href="' + r + other.index + '" title="' + other.name + '">' + other.vol + '</a><a href="' + r + 'notation.html">Notation</a><a href="' + r + 'progress.html">My progress</a><a href="' + r + 'references.html">Bibliography</a></nav>' +
      '<button class="btn icon-btn" id="theme-btn" aria-label="Toggle dark mode" title="Toggle dark mode">◐</button>' +
      '<span id="auth-slot"><button class="btn" id="auth-btn" title="Sign in to sync your progress">Sign in</button></span>';
    document.body.insertBefore(bar, document.body.firstChild);
    $('#theme-btn').addEventListener('click', () => setTheme(currentTheme() === 'dark' ? 'light' : 'dark'));
    $('#toc-toggle').addEventListener('click', () => { const s = $('.sidebar'); if (s) s.classList.toggle('open'); });
    $('#auth-btn').addEventListener('click', () => {
      if (window.BookAuth && window.BookAuth.configured) window.BookAuth.signIn();
      else toast('Accounts are not enabled yet. Your progress is saved in this browser; use My progress → Copy sync link to move it to another device.');
    });
  }
  function renderAuth(user, syncing) {
    const slot = $('#auth-slot'); if (!slot) return;
    if (!user) {
      slot.innerHTML = '<button class="btn" id="auth-btn" title="Sign in or create an account to keep your progress on every device">Sign in</button>';
      $('#auth-btn').addEventListener('click', () => window.BookAuth && window.BookAuth.signIn());
      return;
    }
    const name = (user.displayName || user.email || 'Reader').split(' ')[0];
    slot.innerHTML = '<span class="user-chip">' + (user.photoURL ? '<img alt="" referrerpolicy="no-referrer" src="' + user.photoURL + '">' : '') +
      '<span class="sync-dot ' + (syncing ? 'on' : '') + '" title="' + (syncing ? 'Progress synced to your account' : 'Syncing…') + '"></span>' +
      '<span class="small">' + name.replace(/[<>&]/g, '') + '</span>' +
      '<button class="btn small" id="signout-btn">Sign out</button></span>';
    $('#signout-btn').addEventListener('click', () => window.BookAuth.signOut());
  }

  /* ------------------------------------------------------------------ TOC */
  function slug(s) { return s.toLowerCase().replace(/<[^>]+>/g, '').replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 60); }
  function buildToc() {
    const side = $('.sidebar'), art = $('article'); if (!side || !art) return;
    const hs = $$('h2, h3', art);
    const ol = el('ol');
    hs.forEach(h => {
      if (!h.id) h.id = slug(h.textContent);
      const li = el('li', { class: h.tagName.toLowerCase() });
      const a = el('a', { href: '#' + h.id }); a.textContent = h.textContent;
      a.addEventListener('click', () => side.classList.remove('open'));
      li.appendChild(a); ol.appendChild(li);
    });
    side.appendChild(el('h4', null, 'On this page'));
    side.appendChild(ol);
    if ('IntersectionObserver' in window) {
      const links = new Map($$('a', ol).map(a => [a.getAttribute('href').slice(1), a]));
      const io = new IntersectionObserver(entries => {
        entries.forEach(e => {
          if (e.isIntersecting) { links.forEach(a => a.classList.remove('active')); const a = links.get(e.target.id); if (a) a.classList.add('active'); }
        });
      }, { rootMargin: '-10% 0px -80% 0px' });
      hs.forEach(h => io.observe(h));
    }
  }

  /* ------------------------------------------------------------------ MCQ */
  const mcqs = [];
  function initMcqs() {
    const ch = chapterId();
    const chN = (BOOK().chapters.find(c => c.id === ch) || {}).n || '';
    $$('.mcq').forEach((box, i) => {
      const id = box.getAttribute('data-id') || (ch + '-mcq-' + (i + 1));
      box.setAttribute('data-id', id); if (!box.id) box.id = id;
      const lis = $$('ol.opts > li', box);
      const correct = lis.map((li, j) => li.hasAttribute('data-correct') ? j : -1).filter(j => j >= 0);
      const multi = correct.length > 1;
      const qEl = $('.q', box);
      const head = el('div', { class: 'q-head' },
        '<span class="q-num">Q' + chN + '.' + (i + 1) + '</span>' + (box.dataset.topic ? '<span>' + box.dataset.topic + '</span>' : '') + '<span class="q-status"></span>');
      box.insertBefore(head, box.firstChild);
      if (multi) qEl.insertAdjacentHTML('beforeend', ' <span class="multi-hint">(select all that apply)</span>');
      const buttons = lis.map((li, j) => {
        const b = el('button', { type: 'button' }); b.innerHTML = li.innerHTML; li.innerHTML = ''; li.appendChild(b);
        b.addEventListener('click', () => choose(state, j)); return b;
      });
      const actions = el('div', { class: 'q-actions' });
      const checkBtn = el('button', { class: 'btn small primary', type: 'button' }, 'Check answer');
      const retryBtn = el('button', { class: 'btn small', type: 'button' }, 'Try again');
      const meta = el('span');
      if (multi) actions.appendChild(checkBtn);
      actions.appendChild(retryBtn); actions.appendChild(meta);
      const ex = $('.explain', box);
      box.insertBefore(actions, ex);
      const state = { id, box, buttons, correct, multi, sel: new Set(), done: false, head, meta, retryBtn, checkBtn, title: snippet(qEl, 140) };
      retryBtn.style.display = 'none';
      checkBtn.addEventListener('click', () => grade(state));
      retryBtn.addEventListener('click', () => reset(state));
      mcqs.push(state);
      restore(state);
    });
  }
  function choose(s, j) {
    if (s.done) return;
    if (!s.multi) { s.sel = new Set([j]); grade(s); return; }
    if (s.sel.has(j)) s.sel.delete(j); else s.sel.add(j);
    s.buttons.forEach((b, k) => b.classList.toggle('sel', s.sel.has(k)));
  }
  function paint(s, chosen) {
    s.buttons.forEach((b, k) => {
      b.classList.remove('sel', 'right', 'wrong'); b.disabled = true;
      if (s.correct.includes(k)) b.classList.add('right');
      else if (chosen.includes(k)) b.classList.add('wrong');
    });
    s.box.classList.add('answered'); s.done = true; s.retryBtn.style.display = '';
    if (s.multi) s.checkBtn.style.display = 'none';
  }
  function isRight(s, chosen) { return chosen.length === s.correct.length && chosen.every(c => s.correct.includes(c)); }
  function grade(s) {
    const chosen = Array.from(s.sel).sort();
    if (!chosen.length) { toast('Pick at least one option.'); return; }
    const ok = isRight(s, chosen);
    const prev = store.get(s.id) || {};
    const attempts = (prev.attempts || 0) + 1;
    store.set(s.id, { t: 'mcq', ch: chapterId(), chosen, ok, attempts, firstOk: prev.attempts ? !!prev.firstOk : ok, title: s.title });
    paint(s, chosen); status(s);
  }
  function status(s) {
    const r = store.get(s.id); const st = $('.q-status', s.head);
    if (!r) { st.textContent = ''; st.className = 'q-status'; s.meta.textContent = ''; return; }
    st.textContent = r.ok ? '✓ correct' : '✗ incorrect'; st.className = 'q-status ' + (r.ok ? 'ok' : 'no');
    s.meta.textContent = 'Attempts: ' + r.attempts + (r.attempts > 1 ? (r.firstOk ? ' · right first time' : ' · missed first time') : '');
  }
  function reset(s) {
    s.done = false; s.sel = new Set(); s.box.classList.remove('answered'); s.retryBtn.style.display = 'none';
    if (s.multi) s.checkBtn.style.display = '';
    s.buttons.forEach(b => { b.disabled = false; b.classList.remove('sel', 'right', 'wrong'); });
  }
  function restore(s) {
    const r = store.get(s.id);
    reset(s);
    if (r && r.chosen) paint(s, r.chosen);
    status(s);
  }

  /* ------------------------------------------------------------------ Interview Q&A */
  const qas = [];
  function initQas() {
    const ch = chapterId();
    const chN = (BOOK().chapters.find(c => c.id === ch) || {}).n || '';
    $$('details.qa').forEach((d, i) => {
      const id = d.getAttribute('data-id') || (ch + '-qa-' + (i + 1));
      d.setAttribute('data-id', id); if (!d.id) d.id = id;
      const sum = $('summary', d);
      const txt = el('span', { class: 'qa-text' });
      Array.from(sum.childNodes).forEach(n => { if (!(n.nodeType === 1 && n.classList.contains('lvl'))) txt.appendChild(n); });
      sum.appendChild(txt);
      sum.insertAdjacentHTML('afterbegin', '<span class="qa-num">I' + chN + '.' + (i + 1) + '</span>');
      const stateEl = el('span', { class: 'qa-state' }); sum.appendChild(stateEl);
      const ans = $('.answer', d) || d;
      const rate = el('div', { class: 'qa-rate' }, '<span>How did you do?</span>');
      const b1 = el('button', { class: 'btn small', type: 'button' }, '✓ I nailed it');
      const b2 = el('button', { class: 'btn small', type: 'button' }, '↻ Review again');
      const b3 = el('button', { class: 'btn small', type: 'button' }, 'Clear');
      rate.append(b1, b2, b3); ans.appendChild(rate);
      const s = { id, d, stateEl, title: snippet(sum, 140) };
      b1.addEventListener('click', () => { store.set(id, { t: 'qa', ch, rating: 'got', title: s.title }); paintQa(s); toast('Marked as mastered'); });
      b2.addEventListener('click', () => { store.set(id, { t: 'qa', ch, rating: 'review', title: s.title }); paintQa(s); toast('Added to your review list'); });
      b3.addEventListener('click', () => { store.del(id); paintQa(s); });
      qas.push(s); paintQa(s);
    });
  }
  function paintQa(s) {
    const r = store.get(s.id);
    s.stateEl.className = 'qa-state' + (r ? (r.rating === 'got' ? ' ok' : ' rev') : '');
    s.stateEl.textContent = r ? (r.rating === 'got' ? '✓ mastered' : '↻ review') : 'reveal answer ▾';
  }

  /* ------------------------------------------------------------------ chapter score + completion */
  function chapterStats(ch) {
    const items = Object.values(store.data.items).filter(r => r && r.ch === ch);
    const m = items.filter(r => r.t === 'mcq'), q = items.filter(r => r.t === 'qa');
    return { mcqDone: m.length, mcqOk: m.filter(r => r.ok).length, qaGot: q.filter(r => r.rating === 'got').length, qaRev: q.filter(r => r.rating === 'review').length };
  }
  function buildScore() {
    const side = $('.sidebar'); const ch = chapterId(); if (!side || !ch) return;
    const box = el('div', { class: 'chapter-score' });
    side.appendChild(box);
    const paintScore = () => {
      const st = chapterStats(ch), tot = mcqs.length || 1;
      box.innerHTML = '<b>Your progress</b><div class="bar"><i style="width:' + Math.round(100 * st.mcqDone / tot) + '%"></i></div>' +
        '<div>MCQs: ' + st.mcqDone + '/' + mcqs.length + ' answered · ' + st.mcqOk + ' correct</div>' +
        '<div>Interview Qs: ' + st.qaGot + ' mastered · ' + st.qaRev + ' to review</div>' +
        '<div class="muted small" id="sync-note">' + (window.BookAuth && window.BookAuth.user ? 'Synced to your Google account' : 'Saved in this browser') + '</div>';
    };
    paintScore();
    store.onChange(paintScore);
    store.setChapter(ch, Object.assign(store.chapter(ch), { totalMcq: mcqs.length, totalQa: qas.length, visited: true }));
  }
  function buildCompletion() {
    const art = $('article'); const ch = chapterId(); if (!art || !ch) return;
    const row = el('div', { class: 'complete-row' });
    const btn = el('button', { class: 'btn', type: 'button' });
    const paintBtn = () => { const done = !!store.chapter(ch).complete; btn.textContent = done ? '✓ Chapter completed' : 'Mark chapter as completed'; btn.className = 'btn' + (done ? ' primary' : ''); };
    btn.addEventListener('click', () => { store.setChapter(ch, { complete: !store.chapter(ch).complete }); paintBtn(); });
    row.appendChild(btn); row.appendChild(el('span', { class: 'muted' }, 'Your answers and ratings are saved automatically.'));
    const nav = $('.chap-nav', art);
    art.insertBefore(row, nav || null);
    paintBtn(); store.onChange(paintBtn);
  }
  function buildChapNav() {
    const art = $('article'); const ch = chapterId(); if (!art || !ch || $('.chap-nav', art)) return;
    const list = BOOK().chapters, idx = BOOK().index;
    const i = list.findIndex(c => c.id === ch); const r = root();
    const nav = el('nav', { class: 'chap-nav' });
    const prev = list[i - 1], next = list[i + 1];
    nav.innerHTML = (prev && prev.ready ? '<a href="' + r + prev.file + '"><small>← Previous</small>' + prev.n + '. ' + prev.title + '</a>' : '<a href="' + r + idx + '"><small>← Back</small>Contents</a>') +
      (next && next.ready ? '<a class="next" href="' + r + next.file + '"><small>Next →</small>' + next.n + '. ' + next.title + '</a>' : '<a class="next" href="' + r + idx + '#contents"><small>Coming next</small>' + (next ? next.n + '. ' + next.title : 'Contents') + '</a>');
    art.appendChild(nav);
  }

  /* ------------------------------------------------------------------ math */
  function renderMath(node) {
    if (typeof window.renderMathInElement !== 'function') return;
    window.renderMathInElement(node || document.body, {
      delimiters: [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }],
      throwOnError: false, ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    });
  }

  /* ------------------------------------------------------------------ boot */
  function boot() {
    buildHeader();
    buildToc();
    initMcqs();
    initQas();
    buildScore();
    buildChapNav();
    buildCompletion();
    renderMath();
    if (window.BookCharts) window.BookCharts.renderAll();
    store.onChange(ev => { if (ev.kind === 'reload') { mcqs.forEach(restore); qas.forEach(paintQa); } });
    document.dispatchEvent(new CustomEvent('book:ready'));
  }
  window.Book = { get CHAPTERS() { return BOOK().chapters; }, get PARTS() { return BOOK().parts; }, ALL_CHAPTERS, BOOKS, get BOOK() { return BOOK(); }, store, toast, renderAuth, chapterStats, renderMath, el };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
