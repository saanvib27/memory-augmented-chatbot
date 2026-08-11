# Memory-Augmented Chatbot — Complete Concepts & Study Guide

> A deep, from-first-principles explanation of **every** technology and design decision in this project.
> Written so you can walk into a meeting and confidently explain *what* each piece is, *why* it exists,
> *why we chose it*, *where else it's used*, and *what its limits are*.

---

## How to use this guide

Each concept follows the same template so you can revise fast:

- **In one line** — the elevator pitch.
- **The problem it solves** — why it exists at all.
- **How it works** — the mechanics, with an example.
- **Why preferred / alternatives** — what we compared it against.
- **Where it's used (sectors)** — real-world context.
- **Limitations** — where it breaks or costs you.
- **Why we chose it here** — the decision for *this* project.

Read Section 1 first — it gives you the mental model that everything else hangs off.

---

## Table of Contents

1. [The Big Picture — mental model & request flow](#1-the-big-picture)
2. [Large Language Models (LLMs)](#2-large-language-models-llms)
3. [Groq + LLaMA 3.3 70B — our LLM choice](#3-groq--llama-33-70b)
4. [Tokens & Embeddings](#4-tokens--embeddings)
5. [Vector Databases & FAISS](#5-vector-databases--faiss)
6. [Retrieval-Augmented Generation (RAG)](#6-retrieval-augmented-generation-rag)
7. [Chunking](#7-chunking)
8. [Knowledge Graphs (spaCy + networkx)](#8-knowledge-graphs)
9. [Hybrid RAG — vectors + graph together](#9-hybrid-rag)
10. [Memory: short-term vs long-term](#10-memory)
11. [Automatic Memory Extraction](#11-automatic-memory-extraction)
12. [LangGraph & Agent Orchestration](#12-langgraph--agent-orchestration)
13. [Supabase (Postgres + Auth)](#13-supabase)
14. [Authentication: OAuth & Guest mode](#14-authentication)
15. [FastAPI Backend](#15-fastapi-backend)
16. [The Frontend (3-panel UI)](#16-the-frontend)
17. [Evaluation: relevance, recall & cosine similarity](#17-evaluation)
18. [Week 8 — Single-Agent Pipeline concepts](#18-week-8-single-agent-pipeline)
19. [Why this whole stack? (cost, offline, trade-offs)](#19-why-this-stack)
20. [Likely meeting questions & crisp answers](#20-likely-meeting-questions)
21. [Glossary](#21-glossary)

---

## 1. The Big Picture

### The mental model

A plain LLM (like the base LLaMA model) is a **very well-read person with no notes and no memory of you**. It knows a lot from training, but:
- It can't cite *your* documents (it may hallucinate).
- It forgets everything the moment the conversation ends.
- Its knowledge is frozen at its training cutoff.

This project fixes all three by wrapping the LLM in three "augmentations":

| Augmentation | Fixes | In this project |
|---|---|---|
| **Retrieval (RAG)** | Hallucination, stale knowledge | FAISS search over 15 ML/AI Wikipedia articles |
| **Knowledge Graph** | "How are concepts related?" | spaCy-extracted triples in networkx |
| **Memory** | Forgetting the user | Supabase stores facts + chat history |

So the name **"Memory-Augmented Chatbot with Knowledge Graph and Hybrid RAG"** literally describes the three augmentations bolted onto the LLM.

### End-to-end request flow (what happens when you hit "Send")

```
You type: "Explain GANs"
      │
      ▼
[FastAPI /chat endpoint]  ← receives {user_id, session_id, message}
      │
      ▼
[LangGraph StateGraph]  ← orchestrates 4 nodes in order:
      │
      ├─(1) memory_node  → pull your saved facts + recent chat history from Supabase
      ├─(2) rag_node     → embed the question, FAISS finds 5 similar text chunks
      ├─(3) graph_node   → look up related triples in the Knowledge Graph
      └─(4) model_node   → stuff all of the above into one prompt → Groq LLaMA 3.3 → answer
      │                    → then auto-extract new facts about you → save to Supabase
      ▼
[Response] answer + sources + kg_triples + relevance/recall metrics
      │
      ▼
[Browser UI] renders the answer, animates the pipeline, shows sources & metrics
```

Everything below is just a deep-dive into each box in that diagram.

---

## 2. Large Language Models (LLMs)

**In one line:** A neural network trained to predict the next token (word-piece) over enormous text, which as a side effect learns grammar, facts, reasoning patterns, and style.

**The problem it solves:** We want a machine that can understand a natural-language question and produce a fluent, relevant natural-language answer — without hand-coding rules for every possible sentence.

**How it works (example):**
- Text is broken into **tokens** (≈ word pieces). "Generative" might be one token; "GANs" might be two.
- The model is trained to predict the next token: given "Paris is the capital of", it learns to output "France".
- Stack this billions of times over the internet's text and the model absorbs facts and patterns.
- At inference, it generates one token at a time, feeding its own output back in ("autoregression").
- **Transformers** (the architecture, from *"Attention Is All You Need"*, 2017) make this feasible: the **self-attention** mechanism lets every token look at every other token to decide what's relevant. This is exactly one of the topics in your knowledge base.

**Why preferred / alternatives:**
| Approach | Why weaker than LLMs |
|---|---|
| Rule-based chatbots (if/else, regex) | Brittle, can't generalize, huge maintenance |
| Classic ML (bag-of-words + SVM) | No real language understanding, no generation |
| RNN/LSTM seq2seq | Struggle with long-range context; slow (sequential) |
| **Transformers/LLMs** | Parallelizable, capture long-range context, emergent reasoning |

**Where it's used (sectors):** customer support, coding assistants, search, healthcare triage, legal drafting, education, content generation — essentially any task involving language.

**Limitations:**
- **Hallucination** — will confidently invent facts. (RAG mitigates this.)
- **Knowledge cutoff** — doesn't know events after training. (RAG mitigates.)
- **No inherent memory** across sessions. (Our memory layer fixes this.)
- **Cost & latency** — big models are expensive to run.
- **Context window limit** — can only "see" a fixed number of tokens at once.

**Why we chose it here:** The whole task ("understand ML questions, answer fluently, personalize") is a language task — LLM is the only sensible engine. The augmentations exist precisely to patch the LLM's limitations above.

---

## 3. Groq + LLaMA 3.3 70B

**In one line:** We run the open-weights **LLaMA 3.3 70B** model on **Groq**, a cloud provider whose custom **LPU** hardware serves tokens extremely fast, for free on their tier.

**The problem it solves:** We need a *high-quality* LLM but have *no budget* and *no GPU*. We must not run a 70-billion-parameter model on a laptop (impossible), and we don't want to pay OpenAI per token.

**Key distinctions to be crystal-clear on in the meeting:**
- **LLaMA 3.3 70B** = the *model* (the brain / the weights), made by Meta, open-weights.
- **Groq** = the *inference provider* (the hardware + API that runs the model). **Not** to be confused with "Grok" (xAI's model).
- **"70B"** = 70 billion parameters — the tunable numbers in the network. More parameters ≈ more capability (and more compute).
- The model runs **in the cloud**, not locally. Your machine only sends the prompt and receives text.

**How it works:** Your backend calls `groq.chat.completions.create(model="llama-3.3-70b-versatile", messages=[...])`. Groq runs the model on their LPUs and streams back the answer, usually in 1–2 seconds.

**Why preferred / alternatives:**
| Option | Verdict for this project |
|---|---|
| OpenAI GPT-4o | Excellent but **paid** per token |
| Google Gemini free tier | Good, but stricter rate limits; we standardized on Groq |
| Local LLaMA via Ollama | **Free** but needs a strong GPU; too slow/heavy for a laptop |
| Hugging Face Inference API | Needs auth, rate-limited, slower for big models |
| **Groq + LLaMA 3.3 70B** | Free tier, 70B quality, **very fast** (LPU), simple API ✅ |

**Where it's used (sectors):** Groq is used where **low latency** matters — real-time assistants, voice agents, high-throughput chat. Open models like LLaMA are favored where **data privacy / no vendor lock-in / cost control** matter (finance, healthcare, on-prem enterprise).

**Limitations:**
- Free tier has **rate limits** (requests/min, tokens/min).
- You depend on Groq's uptime (a hosted service).
- 70B is strong but not GPT-4-class on the hardest reasoning.
- Model IDs get **deprecated** — we already hit this once (`llama3-70b-8192` was retired → switched to `llama-3.3-70b-versatile`). Good story to tell: shows you handle real-world breakage.

**Why we chose it here:** Best **quality-per-dollar-per-latency** for a student project. Free, fast, 70B-quality, and a drop-in OpenAI-style API so the code stays simple.

---

## 4. Tokens & Embeddings

These two are often confused — be precise.

### Tokens
**In one line:** The chunks of text an LLM actually reads/writes (word-pieces). Billing and context limits are measured in tokens.
**Example:** "unbelievable" → `un`, `believ`, `able` (3 tokens). Rough rule: **1 token ≈ 0.75 words**.

### Embeddings
**In one line:** A way to turn a piece of text into a **list of numbers (a vector)** that captures its *meaning*, so that texts with similar meaning have similar vectors.

**The problem it solves:** Computers can't compare *meaning* directly. "car" and "automobile" share zero letters but mean the same thing. Embeddings place them close together in a numeric space so we can measure semantic similarity with math.

**How it works (example):**
- We use the model **`all-MiniLM-L6-v2`** (from sentence-transformers). It maps any sentence to a **384-dimensional** vector.
- "What is RAG?" → `[0.12, -0.04, 0.88, … ]` (384 numbers).
- A chunk about retrieval-augmented generation gets a *nearby* vector.
- **Similarity** is measured with **cosine similarity** — the cosine of the angle between two vectors. 1.0 = identical meaning, 0 = unrelated.

**Why preferred / alternatives:**
| Text representation | Weakness |
|---|---|
| Keyword match (TF-IDF, BM25) | No understanding of synonyms/meaning; "car" ≠ "automobile" |
| One-hot / bag-of-words | Huge, sparse, no semantics |
| Word2Vec / GloVe (word-level) | No *sentence* context; "bank" (river vs money) ambiguous |
| **Sentence embeddings (MiniLM)** | Whole-sentence meaning, dense, fast, small ✅ |

**Where it's used (sectors):** semantic search, recommendation systems, deduplication, clustering, RAG, plagiarism detection, anomaly detection.

**Limitations:**
- Quality is capped by the embedding model; `all-MiniLM-L6-v2` is small (fast but not the most accurate).
- Embeddings can miss nuance/negation ("not good" vs "good").
- Domain mismatch: a model trained on general text may embed niche jargon poorly.

**Why we chose `all-MiniLM-L6-v2`:** It's **tiny (runs on CPU/laptop), fast, free, and 384-dim** (small index, quick search) while still good enough for a focused ML/AI corpus. It runs *locally* — the only heavy compute that happens on your machine.

---

## 5. Vector Databases & FAISS

**In one line:** A specialized store that holds embeddings and can find the *nearest* vectors to a query vector in milliseconds. We use **FAISS** (Facebook AI Similarity Search).

**The problem it solves:** Once every chunk is an embedding, "find the most relevant chunk to this question" = "find the nearest vectors." Doing that by comparing against every vector one by one is slow at scale. Vector DBs make it fast (approximate nearest-neighbor search).

**How it works (example):**
1. Offline: embed all 163 chunks → store 163 vectors in a FAISS index on disk (`data/embeddings/faiss.index`).
2. At query time: embed "Explain GANs" → ask FAISS for the **top-5 nearest** chunk vectors.
3. FAISS returns the chunk IDs; we fetch their text and feed it to the LLM.

**Why preferred / alternatives:**
| Vector store | Trade-off |
|---|---|
| Pinecone / Weaviate (managed) | Great at scale, but **paid / cloud dependency** |
| Chroma | Nice dev UX; heavier than we need |
| pgvector (Postgres) | Good if already in Postgres; extra setup |
| Brute-force numpy loop | Fine for tiny data; doesn't scale |
| **FAISS (local)** | **Free, local, extremely fast, battle-tested** ✅ |

**Where it's used (sectors):** any semantic search / RAG system — e-commerce search, support-ticket routing, legal/medical document retrieval, image similarity (FAISS also does image vectors).

**Limitations:**
- FAISS is a **library, not a full database** — no built-in persistence layer, metadata filtering, or multi-user access control (you manage that yourself; we keep a separate metadata `.pkl`).
- Approximate search can occasionally miss the true nearest neighbor (tunable).
- In-memory index — very large corpora need sharding/other index types.

**Why we chose FAISS:** Our corpus is small (163 chunks), we want **zero cost and zero cloud dependency**, and FAISS is the industry-standard local vector search. Perfect fit.

---

## 6. Retrieval-Augmented Generation (RAG)

**In one line:** Before the LLM answers, **retrieve** relevant documents and **insert them into the prompt** so the model answers *from those facts* instead of from memory alone.

**The problem it solves:** LLMs hallucinate and go stale. RAG grounds answers in a trusted, up-to-date knowledge source *you* control — and lets you **cite sources**.

**How it works (example):**
1. User asks "What is RAG?"
2. Retrieve top-5 chunks about RAG from FAISS.
3. Build a prompt: *"Using this context: [chunk1…chunk5], answer: What is RAG?"*
4. LLM answers grounded in those chunks, and we show the user which sources were used.

This is why, in your screenshots, answers say things like *"(Source: [Retrieval-augmented generation])"* and show **"5 sources used."**

**Why preferred / alternatives:**
| Alternative to RAG | Why RAG wins |
|---|---|
| Fine-tuning the model on your data | Expensive, slow to update, still can hallucinate, needs GPUs |
| Bigger context window (paste everything) | Costs tokens, hits limits, model gets "lost in the middle" |
| Plain LLM | Hallucinates, can't cite, stale |
| **RAG** | Cheap, updatable (just re-index), grounded, cite-able ✅ |

**Where it's used (sectors):** enterprise Q&A over internal docs, customer support bots, legal/medical/finance research assistants, coding assistants over a codebase — RAG is the **dominant pattern** for production LLM apps in 2024–2026.

**Limitations:**
- **Garbage in, garbage out** — if retrieval fetches irrelevant chunks, the answer suffers.
- Retrieval quality depends on embeddings + chunking.
- Adds latency (an extra search step).
- Doesn't help with reasoning the docs don't contain.

**Why we chose RAG:** It's the correct, modern way to make a *domain-specific, grounded* assistant without fine-tuning. It also gives us the "sources used" and evaluation metrics that make the project demonstrable and trustworthy.

---

## 7. Chunking

**In one line:** Splitting long documents into smaller passages ("chunks") before embedding, so retrieval returns *focused* pieces rather than whole articles.

**The problem it solves:** You can't embed a 5,000-word Wikipedia article as one vector — it would blur many topics into one "average" meaning, and you can't fit it into the prompt. Chunking keeps each vector about *one* idea.

**How it works (example):** The Wikipedia "Transformer" article is split into ~500-token passages. Each passage becomes one embedding. A question about "attention" then matches the specific passage about attention, not the whole article. Our pipeline produced **163 chunks** from 15 articles.

**Why preferred / trade-offs (chunk size is a real dial):**
| Chunk size | Effect |
|---|---|
| Too small (1 sentence) | Loses context; retrieval fragments |
| Too large (whole article) | Diluted meaning; wastes prompt tokens |
| **~300–500 tokens** | Sweet spot: one coherent idea per chunk ✅ |

**Where it's used:** every RAG system. Advanced variants: overlapping chunks, semantic chunking, recursive splitting by headings.

**Limitations:** A fact split across two chunks can be missed; fixed-size splits can cut mid-idea (overlap mitigates this).

**Why we chose ~500-token chunks:** Balances context vs precision for encyclopedic text, and keeps the prompt small enough for fast, cheap LLM calls.

---

## 8. Knowledge Graphs

**In one line:** A structured web of **entities** (nodes) connected by **relationships** (edges), stored as **(subject → relation → object) triples**, e.g. `Transformer —[uses]→ attention`.

**The problem it solves:** Vector search finds *similar text* but doesn't understand *explicit relationships*. A KG answers "how is X connected to Y?" and gives structured, explainable facts.

**How it works (example) — our offline pipeline:**
1. **spaCy** reads each chunk and does **NLP parsing**:
   - **NER (Named Entity Recognition)** — finds entities ("BERT", "Google").
   - **Dependency parsing** — finds grammatical structure: subject → verb → object.
   - From "RNNs model continuous time," it extracts `RNN —[model]→ continuous time`.
2. These **762 triples** are loaded into **networkx**, a Python graph library (1,200 nodes, 760 edges), saved as `graph.json`.
3. At query time, `graph_node` looks up triples related to the question's keywords and adds them to the prompt. That's what fills the **Knowledge Graph** panel in the UI (e.g. `origin RNN —[be]→ neuroscience`).

**Why preferred / alternatives:**
| Option | Trade-off |
|---|---|
| **Neo4j** (real graph DB) | Powerful (Cypher queries) but needs **Docker/a server** |
| RDF triple stores | Standards-heavy, overkill here |
| **networkx** (in-memory) | **Zero infra, pure Python, saves to JSON** ✅ |

**Where it's used (sectors):** Google's Knowledge Panel, fraud detection (linking accounts), recommendation engines, drug discovery (protein/gene relationships), enterprise data integration, search.

**Limitations:**
- Extraction quality: rule-based (dependency parsing) triples can be noisy — you saw this (we filtered out LaTeX/math junk triples).
- Doesn't scale to billions of edges in-memory (that's where Neo4j earns its keep).
- Building/maintaining a high-quality KG is labor-intensive.

**Why we chose spaCy + networkx:** We wanted a KG **without standing up a database server**. spaCy extracts triples offline; networkx holds the graph in memory and serializes to JSON. **No Docker, no Neo4j, runs anywhere** — matching the project's "zero-infra, free" philosophy.

---

## 9. Hybrid RAG

**In one line:** Combine **two retrieval methods** — vector search (FAISS) **and** the Knowledge Graph — feeding *both* into the LLM prompt.

**The problem it solves:** Each method has a blind spot. Vectors are great at "find text about this topic" but weak at explicit relations. Graphs are great at "how are these connected" but weak at fuzzy semantic matching. Using both covers each other's gaps.

**How it works (example):** For "How do transformers relate to attention?":
- **FAISS** returns passages *describing* transformers and attention.
- **KG** returns the explicit edge `Transformer —[uses]→ attention`.
- The LLM sees both the prose and the structured relationship → a richer, better-grounded answer.

**Why preferred / alternatives:** Pure vector RAG is the common baseline; adding graph context is a step up in **explainability** and **relational** questions. (Other hybrids combine vector + keyword/BM25 search — also valid; we chose vector + graph because the KG was a project requirement.)

**Where it's used (sectors):** advanced enterprise search, biomedical QA, financial intelligence — anywhere both *semantic* and *relational* precision matter.

**Limitations:** More moving parts to build and maintain; if the KG is noisy it can add distracting context (hence our triple-filtering).

**Why we chose it:** The assignment explicitly asked for **Knowledge Graph + Hybrid RAG**, and it genuinely produces more grounded, explainable answers than vector-only RAG.

---

## 10. Memory

This is the "Memory-Augmented" in the title. There are **two kinds** — don't conflate them.

### (a) Short-term memory = chat history
**In one line:** The recent turns of the *current* conversation, so the bot can handle follow-ups like "explain *it* more."
- Stored in Supabase `chat_history` (session_id, role, content).
- Injected into the prompt as "Recent conversation: …".
- **Analogy:** working memory / scratchpad.

### (b) Long-term memory = user facts & preferences
**In one line:** Durable facts about *you* that persist across *all* sessions and days — e.g. "User is a research student at Amity University."
- Stored in Supabase `user_memory` (facts + preferences JSON, keyed by user_id).
- Injected as "Known facts about user: …".
- **Analogy:** long-term memory / a CRM profile.

**The problem it solves:** A vanilla LLM forgets you instantly. Memory makes the assistant *personal* and *continuous* — the difference you demonstrated when a fresh GANs answer said *"As a research student … at Amity University, you may find GANs…"* using a fact from an earlier session.

**Why preferred / alternatives:**
| Option | Trade-off |
|---|---|
| Stuff entire history into every prompt | Blows the context window + token cost |
| Fine-tune per user | Absurdly expensive |
| **Store facts in a DB, inject the relevant few** | Cheap, scalable, persistent ✅ |

**Where it's used (sectors):** ChatGPT's "memory" feature, personalized tutoring, healthcare (patient context), CRM assistants, any product that should "remember" a user.

**Limitations:**
- **Privacy/consent** — storing personal facts needs care (GDPR etc.).
- Extraction can save wrong/irrelevant "facts" (needs review/curation).
- Memory can become stale or contradictory over time.

**Why we chose Supabase-backed memory:** Persistent, multi-user, keyed by a real auth identity (Google UUID) or a guest id — so memory follows the user across devices and sessions. Exactly what "memory-augmented" requires.

---

## 11. Automatic Memory Extraction

**In one line:** After each answer, a **second, quick LLM call** reads the exchange and pulls out durable facts about the user, saving them to `user_memory`.

**The problem it solves:** Manual "remember this" buttons are clunky. Auto-extraction makes memory *build itself* as you chat (you watched the chips fill in live).

**How it works (example):**
- You say: *"hi im a data science intern at celebal tech."*
- The extractor prompt: *"Extract personal facts about the USER as a JSON array."*
- Returns: `["User works at Celebal Tech", "User is a data science intern"]`
- We `upsert` these into `user_memory` (deduplicated). Best-effort: wrapped in try/except so a failure never blocks the main answer.

**Why preferred / alternatives:** Regex/keyword extraction ("I am a ___") is brittle; an LLM understands phrasing variety. Trade-off: an extra API call per turn.

**Where it's used:** ChatGPT memory, personalized assistants, note-taking copilots.

**Limitations:** Can extract noise or over-personal data; adds latency/cost; needs guardrails on *what* to store.

**Why we chose it:** It's what makes the "memory" feel magical in a demo, and it's cheap because the ML/AI corpus keeps conversations focused.

---

## 12. LangGraph & Agent Orchestration

**In one line:** **LangGraph** lets us define the pipeline as a **graph of steps (nodes)** with shared state, instead of one tangled function. Our graph is **memory → RAG → KG → LLM**.

**The problem it solves:** A real assistant does several steps per query (fetch memory, retrieve, look up graph, generate, save). Hand-wiring that becomes spaghetti. LangGraph gives a clean, inspectable **StateGraph** where each node reads/updates a shared `state` dict and passes it on.

**How it works (example):** `state = {query, user_id, session_id}` flows through:
- `memory_node` adds `state["memory_context"]` and `state["chat_history"]`
- `rag_node` adds `state["rag_context"]`
- `graph_node` adds `state["graph_context"]`
- `model_node` reads all of it, calls the LLM, writes `state["answer"]`, saves history + memory.

This maps 1:1 to the animated **Pipeline** panel in your UI.

**Why preferred / alternatives:**
| Option | Trade-off |
|---|---|
| Plain Python function calls | Works, but hard to extend, branch, or visualize |
| **LangChain "chains"** | Linear; less flexible for branching/loops |
| **LangGraph** | Graph structure → supports branching, loops, conditional routing, state ✅ |
| Full agent frameworks (AutoGPT-style) | Overkill, unpredictable for a scoped task |

**Where it's used (sectors):** production LLM agents — multi-step research assistants, tool-using agents, customer-support workflows, anything with conditional logic ("if the query is math, route to calculator").

**Limitations:** Another dependency/abstraction to learn; for a strictly linear 4-step flow it's arguably more than needed — but it makes the architecture clean, extensible, and easy to explain (a plus for the assignment).

**Why we chose LangGraph:** The assignment asked for a **LangGraph agent**, and it genuinely gives a tidy, extensible orchestration with shared state — and a story that maps directly onto the UI's pipeline visualization.

> **Note on Week 8:** the *separate* Week 8 notebook is a **single-agent pipeline built from scratch** (no LangGraph) to show you understand the *underlying* routing logic manually. See Section 18.

---

## 13. Supabase

**In one line:** A hosted **PostgreSQL** database + **Auth** + auto-generated APIs — "open-source Firebase." We use it for **all persistence** (memory, history, sessions) and **Google login**.

**The problem it solves:** We need a real, multi-user, cloud database and authentication — without running our own server or writing a backend for auth.

**How it works (example):** Three tables:
- `user_memory` (facts, preferences per user_id)
- `chat_history` (every turn: session_id, user_id, role, content)
- `sessions` (id, title, timestamps — powers the sidebar list)

Plus **Supabase Auth** handles Google OAuth and gives each user a stable UUID we key memory to. Your Supabase screenshots prove all of this with real rows.

**Why preferred / alternatives:**
| Option | Trade-off |
|---|---|
| **MongoDB** (our original plan) | NoSQL; needs hosting or Atlas; we wanted SQL + auth in one |
| Firebase | Great, but proprietary; NoSQL; Google lock-in |
| Raw Postgres on a VM | You manage servers, backups, auth yourself |
| SQLite (local file) | No multi-user/cloud sync |
| **Supabase** | Postgres + Auth + REST, generous **free tier**, dashboard ✅ |

**Where it's used (sectors):** startups & MVPs needing a quick full backend, SaaS apps, hackathons, anything wanting Postgres + auth without DevOps.

**Limitations:** Free-tier limits (rows, pauses on inactivity); cloud dependency; the anon key is public (so **Row-Level Security** matters in real production — ours is a demo).

**Why we chose Supabase:** One service gives us **SQL persistence + Google OAuth + a dashboard** on a free tier — replacing what would otherwise be MongoDB + a custom auth server. Huge time saver, and SQL fits the relational data (users↔sessions↔messages) perfectly.

---

## 14. Authentication

**In one line:** Two ways in — **Google OAuth** (real identity, cross-device memory) and **Guest mode** (instant, local, no login).

**The problem it solves:** Memory must attach to *a person*. We need identity — but we also don't want to force login on someone just trying a demo.

**How it works:**
- **OAuth (Open Authorization):** you click "Continue with Google," Google verifies you and hands Supabase a token; Supabase gives us a stable **user UUID**. We never see your password. That UUID keys your memory (the `98abe39e-…` in your screenshots).
- **Guest mode:** we generate a random `guest_xxx` id in the browser's `localStorage`. Memory works, but only on that device.

**Why preferred / alternatives:**
| Option | Trade-off |
|---|---|
| Email+password (roll your own) | You store passwords → security burden |
| **OAuth (Google)** | No passwords stored, trusted provider, 1-click ✅ |
| No auth at all | Can't do per-user persistent memory |

**Where it's used (sectors):** virtually every consumer app ("Sign in with Google/Apple/GitHub").

**Limitations:** OAuth setup is fiddly (consent screen, redirect URLs, verification status — you lived through publishing the app). Depends on the provider. Guest data is device-bound and not synced.

**Why we chose this combo:** Google OAuth gives frictionless, secure, cross-device identity for real users; guest mode keeps the demo instantly usable for an evaluator. Best of both.

---

## 15. FastAPI Backend

**In one line:** A modern, fast Python web framework that exposes our logic as HTTP **endpoints** (`/chat`, `/sessions`, `/memory`) the frontend calls.

**The problem it solves:** The browser UI needs to talk to the Python ML pipeline. FastAPI is the bridge — it receives JSON requests, runs the pipeline, returns JSON responses.

**How it works (example):** `POST /chat {message, user_id, session_id}` → runs LangGraph → returns `{answer, sources, kg_triples, context_relevance, context_recall}`. FastAPI also auto-generates interactive API docs at `/docs`.

**Why preferred / alternatives:**
| Framework | Trade-off |
|---|---|
| **Flask** | Simple but synchronous by default, no built-in validation/docs |
| Django | Batteries-included but heavy for an API |
| **Streamlit** | Fast to demo but *not* a real API; couples UI+logic, reruns whole script — we deliberately avoided it |
| **FastAPI** | Async, **Pydantic validation**, auto docs, high performance ✅ |

**Where it's used (sectors):** ML model serving, microservices, high-throughput APIs — it's the default for Python ML backends today.

**Limitations:** Python concurrency limits (GIL) for CPU-bound work; you write your own frontend (unlike Streamlit). For us that's a feature — a real API + custom UI is more impressive and production-like.

**Why we chose FastAPI over Streamlit:** A real backend/frontend split looks and behaves like a production system: structured JSON responses, multiple endpoints, a custom ChatGPT-style UI, and clean separation of concerns. (You explicitly decided this — see the deployment discussion.)

---

## 16. The Frontend

**In one line:** A hand-built **3-panel** single-page app (vanilla HTML/CSS/JS) — left = sessions + memory, center = chat, right = pipeline + metrics + KG.

**The problem it solves:** Users need to *see* the system working, not just get text. The UI makes the RAG/KG/memory pipeline visible and the experience feel like ChatGPT.

**Key features (and why each exists):**
- **Pipeline animation** — teaches/shows the 4 stages running.
- **Sources drawer + relevance/recall bars** — trust & transparency (you can see what grounded the answer).
- **KG panel** — surfaces the structured relations.
- **Session sidebar (new/search/rename/delete)** — conversation management like ChatGPT.
- **Streaming (typewriter)** — perceived speed + familiar UX.
- **Export to Markdown, copy, regenerate, feedback** — practical polish.
- **Light/dark theme** — accessibility & preference.

**Why vanilla JS / alternatives:** React/Vue would be typical for bigger apps but add build tooling. For a single self-contained `index.html` served by FastAPI, **vanilla JS keeps it dependency-free and instantly runnable**.

**Limitations:** Vanilla JS gets unwieldy for very large UIs (no components/state framework). Fine at this size.

**Why we chose it:** Zero build step, one file, easy to serve, and it still delivers a polished 3-panel experience.

---

## 17. Evaluation

**In one line:** We quantify retrieval quality with two **cosine-similarity** metrics — **Context Relevance** and **Context Recall** — computed locally, no external API.

**The problem it solves:** "Is the RAG actually retrieving good context?" You need numbers, not vibes, to prove and improve the system.

**How it works (example):**
- **Context Relevance** = cosine similarity between the **query** and the **retrieved chunks**. High = we fetched on-topic material. (Avg **0.553** across 7 test queries.)
- **Context Recall** = cosine similarity between the **generated answer** and the **retrieved chunks**. High = the answer actually used the retrieved context. (Avg **0.639**.)
- Both shown live in the UI's "Retrieval Quality" bars per answer.

**Why preferred / alternatives:**
| Option | Trade-off |
|---|---|
| **RAGAS** (LLM-as-judge framework) | Powerful but needs an **API key + costs**; we avoided the dependency |
| Human eval | Gold standard but slow/subjective |
| **Cosine-similarity metrics** | Free, instant, deterministic, good enough to show retrieval health ✅ |

**Where it's used (sectors):** every serious RAG deployment needs eval; frameworks include RAGAS, TruLens, DeepEval, plus classic IR metrics (precision@k, recall@k, MRR, nDCG).

**Limitations:** Cosine similarity is a **proxy** — it measures semantic overlap, not factual correctness or hallucination. It won't catch a fluent-but-wrong answer. RAGAS/human eval go deeper.

**Why we chose cosine metrics:** They give **honest, reproducible numbers with zero cost/keys**, and they map directly onto the live UI bars — perfect for a demonstrable student project.

---

## 18. Week 8 — Single-Agent Pipeline

*(This is the separate Week 8 assignment, distinct from the Final Project chatbot.)*

**In one line:** A **single agent** that reads a query, **detects intent**, and **routes** to the right **tool** (calculator, keyword extractor, etc.), returning **structured JSON**.

**Core concepts to explain:**

**1. Intent detection / routing** — decide *what the user wants* and send it to the right handler.
- **Example:** "calculate 20 + 5" → intent `calculation` → calculator tool.
- We used **regex pattern matching** (bonus: more robust than plain keyword `in` checks — catches "20 + 5", "what is 15*4?" without the word "calculate").

**2. Tool use** — the agent doesn't do everything itself; it calls specialized functions ("tools"):
- Calculator, Keyword Extractor (required) + Summarizer, Word Counter, Unit Converter (bonus).
- This is the **core idea behind modern "agents"**: an LLM/router deciding which tool to invoke.

**3. Structured output (JSON)** — every response is `{"type": ..., "result": ...}` so a program (not just a human) can consume it reliably.

**4. Error handling & logging (bonus)** — empty/invalid input returns a clean error; a logger records each query, detected intent, and result.

**Why preferred / where used:** Intent-routing + tools is exactly how real assistants (Siri, Alexa, LLM tool-calling) work — classify the request, dispatch to a capability, return structured data an app can act on.

**Limitations:** Rule/regex routing is brittle vs. an LLM classifier; a fixed tool set can't handle unforeseen requests. (The Final Project shows the LLM-driven evolution of the same idea.)

**Why this design:** The assignment asked for a *single-agent* pipeline with routing, tools, structured output, and error handling — and we added logging + extra tools + regex routing as the bonus.

---

## 19. Why This Stack?

The single thread tying every choice together: **maximum capability at zero cost, minimal infrastructure, runs on a laptop.**

| Need | Cheap/heavy option we avoided | What we used | Why |
|---|---|---|---|
| LLM | Paid GPT-4 / local GPU | Groq LLaMA 3.3 70B | Free, fast, 70B quality |
| Embeddings | Paid embedding API | all-MiniLM-L6-v2 (local) | Free, runs on CPU |
| Vector search | Pinecone (paid) | FAISS (local) | Free, fast, standard |
| Knowledge Graph | Neo4j (Docker/server) | spaCy + networkx | Zero infra, offline |
| Memory/DB/Auth | MongoDB + custom auth | Supabase | Postgres + OAuth + free tier |
| Eval | RAGAS (API key) | Cosine similarity | Free, instant |
| UI | React (build tooling) | Vanilla JS | One file, no build |

**The narrative for your meeting:** *"Every component was chosen to be free, laptop-friendly, and dependency-light, while still using industry-standard patterns (RAG, KG, agent orchestration, OAuth). The result runs end-to-end with just a Groq key and a Supabase project."*

---

## 20. Likely Meeting Questions

**Q: Is the model running locally?**
A: No. The LLM (LLaMA 3.3 70B) runs in the cloud on Groq. Only the small embedding model runs locally. My laptop mostly orchestrates and stores.

**Q: How do you prevent hallucination?**
A: RAG — I retrieve real passages from my knowledge base and make the model answer from them, and I show the sources + retrieval metrics. It's grounded, not free-form.

**Q: Difference between RAG and fine-tuning?**
A: Fine-tuning bakes knowledge into weights (expensive, static, still hallucinates). RAG injects fresh, external knowledge at query time — cheap, updatable, cite-able. I chose RAG.

**Q: Why a Knowledge Graph *and* vector search?**
A: Hybrid RAG. Vectors capture fuzzy semantic similarity; the graph captures explicit relationships. Together they answer both "about this topic" and "how these connect."

**Q: How does memory work / how is it personalized?**
A: Two layers — short-term chat history and long-term user facts in Supabase, keyed to the Google user ID. After each turn a quick LLM call extracts new facts. Future answers inject them, so responses personalize (demoed with the "Amity University" answer).

**Q: Why Groq, not OpenAI?**
A: Free tier, 70B open model, and Groq's LPU hardware is very low-latency. Quality-per-dollar-per-latency was best for this project.

**Q: Why FastAPI over Streamlit?**
A: I wanted a real API + custom UI (production-like), not a coupled script. FastAPI gives structured JSON endpoints, validation, and auto docs; the custom frontend shows the pipeline.

**Q: How do you evaluate it?**
A: Two cosine-similarity metrics — context relevance (query↔retrieved) and recall (answer↔retrieved) — shown live and averaged over test queries. No paid eval API. I'm honest that these are proxies, not hallucination detectors.

**Q: What are the biggest limitations?**
A: Small embedding model, noisy rule-based KG triples, cosine metrics don't catch fluent-but-wrong answers, free-tier rate limits, and the demo Supabase setup would need Row-Level Security for production.

**Q: How would you productionize it?**
A: Add RLS + auth hardening, move to a managed/hardened vector DB if the corpus grows, add proper eval (RAGAS/human), rate-limit handling/retries, streaming from the server, observability/logging, and deploy on Railway/Render/Fly.

---

## 21. Glossary

- **LLM** — Large Language Model; next-token predictor trained on huge text.
- **Transformer** — the neural architecture behind LLMs; uses self-attention.
- **Self-attention** — mechanism letting each token weigh every other token's relevance.
- **Token** — a word-piece; unit of LLM input/output and billing.
- **Parameter** — a learned number in the network ("70B" = 70 billion of them).
- **Embedding** — a vector capturing text meaning.
- **Cosine similarity** — angle-based measure of vector closeness (1 = same meaning).
- **Vector database** — store optimized for nearest-neighbor vector search (FAISS).
- **RAG** — Retrieval-Augmented Generation; retrieve docs, then generate.
- **Chunk** — a passage a document is split into before embedding.
- **Knowledge Graph** — entities + relationships as (subject, relation, object) triples.
- **Triple** — one KG fact: subject → relation → object.
- **NER** — Named Entity Recognition; find entities in text.
- **Dependency parsing** — grammatical structure analysis (subject/verb/object).
- **Hybrid RAG** — combining vector search + another retriever (here, the KG).
- **Agent** — a system that decides which steps/tools to use to fulfill a request.
- **LangGraph** — framework to build agents as stateful graphs of nodes.
- **Orchestration** — coordinating the multi-step pipeline.
- **OAuth** — delegated login without sharing passwords (Sign in with Google).
- **Supabase** — hosted Postgres + Auth + APIs ("open-source Firebase").
- **FastAPI** — modern async Python web framework for APIs.
- **Context window** — max tokens an LLM can consider at once.
- **Hallucination** — model stating false info confidently.
- **Inference** — running a trained model to get outputs.
- **LPU** — Groq's Language Processing Unit; hardware optimized for LLM inference.

---

*Prepared as a study companion for the Memory-Augmented Chatbot (Celebal Technologies internship). Read Section 1 and Section 20 the night before; skim the rest by topic.*
