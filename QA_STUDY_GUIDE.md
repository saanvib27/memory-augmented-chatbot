# Course Q&A Study Guide — Deep Learning → Generative AI → Agentic AI

> Interview / viva-style **question-and-answer** revision covering every curriculum topic:
> Autoencoders, VAEs, GANs, Optimizers, GenAI, RAG (basic + advanced), Prompt Engineering,
> LLM Internals, Fine-Tuning (LoRA/PEFT), LLM Evaluation, ReAct, and Week-8 Agentic AI
> (LangGraph, AutoGen / Tool Use / Function Calling, Agent Evaluation).
>
> Each answer gives the *what*, *why*, *how*, an *example*, *alternatives/limits*, and where relevant, how it ties to your projects.

---

## Table of Contents
1. [Autoencoders](#1-autoencoders)
2. [Variational Autoencoders (VAE)](#2-variational-autoencoders-vae)
3. [GANs — Generator, Discriminator, Training](#3-gans)
4. [Optimizers — SGD+Momentum, AdaGrad, RMSprop, Adam](#4-optimizers)
5. [Intro to Generative AI](#5-intro-to-generative-ai)
6. [RAG with LangChain](#6-rag-with-langchain)
7. [Advanced RAG](#7-advanced-rag)
8. [Prompt Engineering](#8-prompt-engineering)
9. [LLM Internals](#9-llm-internals)
10. [Fine-Tuning (LoRA / PEFT)](#10-fine-tuning-lora--peft)
11. [LLM Evaluation](#11-llm-evaluation)
12. [ReAct Framework](#12-react-framework)
13. [Week-8 Agentic AI: LangGraph](#13-langgraph)
14. [AutoGen, Tool Use & Function Calling](#14-autogen-tool-use--function-calling)
15. [Agent Evaluation](#15-agent-evaluation)
16. [Assignment Walkthroughs](#16-assignment-walkthroughs)
17. [Rapid-fire one-liners](#17-rapid-fire)

---

## 1. Autoencoders

**Q1. What is an autoencoder?**
A neural network trained to **reconstruct its own input**. It has two halves: an **encoder** that compresses the input into a small **latent vector (bottleneck)**, and a **decoder** that rebuilds the input from that vector. It learns by minimizing **reconstruction loss** (e.g., MSE between input and output).

**Q2. Why compress to a bottleneck — what's the point of copying the input?**
The bottleneck forces the network to keep only the **most important features** and discard noise. So the latent vector becomes a compact, meaningful representation — useful for compression, denoising, and feature learning.

**Q3. What is it used for (sectors/use cases)?**
- **Denoising** (your assignment: clean noisy images) — feed noisy input, train to output the clean version.
- **Dimensionality reduction** (a non-linear alternative to PCA).
- **Anomaly detection** — train on normal data; anything it reconstructs badly is an anomaly (fraud, defects, network intrusion).
- **Representation/feature learning** for downstream tasks.

**Q4. How does a *denoising* autoencoder work (your assignment)?**
You deliberately add noise to the input image, but compute the loss against the **original clean** image. The network learns to map noisy → clean, i.e., to remove noise rather than merely copy.

**Q5. Autoencoder vs PCA?**
PCA is **linear**; autoencoders use non-linear activations, so they capture **non-linear** structure PCA can't. PCA is simpler/faster and more interpretable.

**Q6. Limitations?**
- A plain autoencoder's latent space is **not organized/continuous** — you can't sample it to generate *new* realistic data (that's what VAEs fix).
- Can overfit / just memorize if the bottleneck is too large.
- Reconstruction (MSE) loss on images tends to produce **blurry** outputs.

---

## 2. Variational Autoencoders (VAE)

**Q1. What is a VAE and how is it different from a normal autoencoder?**
A VAE is a **generative** autoencoder. Instead of encoding an input to a single point, it encodes to a **probability distribution** (a mean μ and variance σ). You then **sample** from that distribution and decode. This makes the latent space **continuous and smooth**, so you can sample new points and generate *new* data.

**Q2. Why encode a distribution instead of a point?**
It regularizes the latent space so nearby points decode to similar, meaningful outputs — enabling **generation** and smooth **interpolation** (e.g., morphing one face into another).

**Q3. What is the "reparameterization trick"?**
Sampling is random and you can't backpropagate through randomness. The trick rewrites the sample as `z = μ + σ · ε`, where `ε` is random noise drawn separately. Now μ and σ are deterministic and **differentiable**, so gradients can flow and the network trains.

**Q4. What is the VAE loss?**
Two terms: **reconstruction loss** (rebuild the input well) + **KL divergence** (keep the learned distribution close to a standard normal). The KL term is what forces a clean, sample-able latent space.

**Q5. VAE vs GAN?**
- VAEs are **stable to train** and give a structured latent space, but outputs are often **blurrier**.
- GANs produce **sharper, more realistic** images but are **harder to train** (instability, mode collapse).

**Q6. Where used?** Image/molecule generation, anomaly detection, data augmentation, learning smooth latent representations.

---

## 3. GANs

**Q1. What is a GAN?**
A **Generative Adversarial Network**: two networks compete in a **zero-sum game**. The **Generator** creates fake data from random noise; the **Discriminator** tries to tell real from fake. They train together until the generator's fakes fool the discriminator. (Introduced by Ian Goodfellow, 2014.)

**Q2. What exactly do the Generator and Discriminator do?**
- **Generator (G):** input = random noise vector `z`; output = synthetic sample (e.g., an image). Goal: fool D.
- **Discriminator (D):** input = a sample (real or fake); output = probability it's real. Goal: catch fakes.
It's like a **counterfeiter (G) vs a detective (D)** — both improve by competing.

**Q3. How is a GAN trained (the loop)?**
1. G makes fake samples from noise.
2. D is shown real + fake samples and updated to classify them correctly (maximize its accuracy).
3. G is updated to make D *wrong* (maximize the probability D labels G's fakes as real).
4. Repeat. This is a **minimax** game — D maximizes, G minimizes the same objective.

**Q4. What is "mode collapse"?**
When G finds a few outputs that reliably fool D and produces **only those**, ignoring the diversity of real data (e.g., generates the same face repeatedly). A key GAN failure mode.

**Q5. Why are GANs hard to train?**
The two networks must stay **balanced**. If D gets too strong, G's gradients vanish; if G gets too strong, D can't teach it. Leads to instability, oscillation, non-convergence.

**Q6. What is a Conditional GAN (cGAN)?**
G and D also receive a **label** (a condition). So you can generate a *specific* class — e.g., "generate the digit 7" or "day→night" image translation. (This is in your knowledge-base corpus.)

**Q7. How are GANs evaluated?**
- **Inception Score (IS):** quality + diversity via a pretrained classifier.
- **Fréchet Inception Distance (FID):** distance between feature distributions of real vs generated images (lower = better). Standard metric.

**Q8. Where used?** Image synthesis, super-resolution, style transfer, data augmentation, deepfakes, art, medical image synthesis.

---

## 4. Optimizers

**Q1. What is an optimizer in deep learning?**
The algorithm that **updates the model weights** using gradients from backpropagation to minimize the loss. It decides *how far* and *in what direction* to step.

**Q2. What is plain SGD, and its problem?**
**Stochastic Gradient Descent** updates weights using the gradient from a mini-batch: `w = w − lr · gradient`. Problems: it can be **noisy**, slow in ravines, and gets stuck oscillating; the learning rate is hard to tune.

**Q3. What does Momentum add?**
It accumulates a **velocity** (an exponential moving average of past gradients) and steps in that direction — like a **ball rolling downhill** gaining speed. It smooths oscillations and speeds up convergence in consistent directions.

**Q4. What is AdaGrad?**
An **adaptive learning rate** per parameter — it divides the step by the accumulated sum of squared past gradients. Parameters with big gradients get smaller steps. Great for **sparse features**. **Problem:** the accumulator only grows, so the learning rate **shrinks to near-zero** and learning stalls.

**Q5. How does RMSprop fix AdaGrad?**
It uses a **moving average** of squared gradients instead of a growing sum, so the effective learning rate **doesn't collapse**. Good for non-stationary/online problems and RNNs.

**Q6. What is Adam and why is it the default?**
**Adam = Momentum + RMSprop.** It keeps a moving average of gradients (1st moment, like momentum) *and* of squared gradients (2nd moment, like RMSprop), plus a bias correction. It's **adaptive, fast, and robust** with little tuning — hence the go-to default.

**Q7. Quick comparison table.**

| Optimizer | Idea | Strength | Weakness |
|---|---|---|---|
| SGD | Raw gradient step | Simple, generalizes well | Noisy, slow, LR-sensitive |
| SGD+Momentum | Adds velocity | Faster, smoother | Still one global LR |
| AdaGrad | Per-param adaptive LR | Good for sparse data | LR decays to zero |
| RMSprop | Moving avg of sq. grads | LR doesn't collapse | Needs LR tuning |
| **Adam** | Momentum + RMSprop | Fast, robust, default | Can generalize slightly worse than tuned SGD |

**Q8. Limitation to mention:** Adam sometimes **generalizes slightly worse** than well-tuned SGD+Momentum on some vision tasks — which is why SGD is still used in some SOTA training.

---

## 5. Intro to Generative AI

**Q1. What is Generative AI?**
AI that **creates new content** (text, images, audio, code) rather than just classifying/predicting labels. It learns the underlying data distribution and samples from it.

**Q2. Discriminative vs Generative models?**
- **Discriminative** learns the boundary `P(y|x)` — "is this spam?" (classification).
- **Generative** learns the data `P(x)` — "produce a realistic email." Generative can synthesize; discriminative can only judge.

**Q3. Main families of generative models?**
- **Autoencoders/VAEs** — latent-space generation.
- **GANs** — adversarial generation.
- **Diffusion models** — iteratively denoise random noise into data (behind DALL·E/Stable Diffusion).
- **Transformers/LLMs** — autoregressive text (and multimodal) generation.

**Q4. Why did GenAI explode recently?**
Transformers + massive data + compute + scaling laws → models that generate coherent long-form content. The **transformer** architecture (attention) was the unlock.

**Q5. Limitations/risks?** Hallucination, bias, copyright/ethics, compute cost, factual staleness — which is exactly why techniques like **RAG** and **evaluation** matter.

---

## 6. RAG with LangChain

**Q1. What is RAG?**
**Retrieval-Augmented Generation:** retrieve relevant external documents and put them in the prompt so the LLM answers from **grounded facts**, not just memory. Reduces hallucination, allows citations, and keeps knowledge current.

**Q2. What are the stages of a RAG pipeline?**
1. **Load** documents. 2. **Split** into chunks. 3. **Embed** chunks into vectors. 4. **Store** in a vector DB. 5. **Retrieve** top-k for a query. 6. **Generate** an answer with the LLM using the retrieved context.

**Q3. What is LangChain and why use it for RAG?**
A framework that provides ready-made **components** (document loaders, text splitters, embeddings, vector-store wrappers, retrievers, chains) so you assemble a RAG pipeline quickly instead of coding each part. It standardizes the "glue."

**Q4. What is a "retriever" and "vector store" in LangChain terms?**
- **Vector store** — holds embeddings + does similarity search (FAISS, Chroma, Pinecone).
- **Retriever** — an interface that, given a query, returns the most relevant documents (usually wrapping the vector store).

**Q5. Example (your Week-7 Document QA):** Load PDFs → split into chunks → embed → store in FAISS → user asks a question → retrieve top chunks → LLM answers with sources.

**Q6. RAG vs fine-tuning?**
RAG injects knowledge **at query time** (cheap, updatable, cite-able). Fine-tuning bakes it into weights (expensive, static, still hallucinates). Use RAG for **knowledge**, fine-tuning for **behavior/style/format**.

**Q7. Limitations?** Retrieval errors propagate; depends on chunking + embedding quality; adds latency; can't reason beyond retrieved text.

---

## 7. Advanced RAG

**Q1. What problems does "naive" RAG have that advanced RAG solves?**
Naive RAG (embed → top-k → stuff) can retrieve **irrelevant or redundant** chunks, miss context split across chunks, and get "lost in the middle." Advanced RAG improves **retrieval precision** and **context quality**.

**Q2. Key advanced-RAG techniques (know a few by name):**
- **Re-ranking** — retrieve many candidates, then a cross-encoder re-scores them for precision.
- **Hybrid search** — combine **semantic (vector)** + **keyword (BM25)** retrieval.
- **Query transformation** — rewrite/expand the query, or **HyDE** (generate a hypothetical answer, embed *that*).
- **Multi-query** — issue several query variants and merge results.
- **Metadata filtering** — restrict retrieval by tags/date/source.
- **Sentence-window / parent-document** retrieval — retrieve small, expand to surrounding context.
- **Reciprocal Rank Fusion (RRF)** — merge multiple ranked lists.

**Q3. What is a cross-encoder re-ranker vs a bi-encoder?**
- **Bi-encoder** (embeddings): encodes query and doc *separately* — fast, used for first-stage retrieval.
- **Cross-encoder**: feeds query+doc *together* into a model for a precise relevance score — slow but accurate, used to **re-rank** the top candidates.

**Q4. How does your project relate?**
Your Final Project uses **Hybrid RAG** in a different sense — **vector search + Knowledge Graph** — which is itself an advanced-RAG pattern (fusing semantic retrieval with structured relations).

**Q5. Limitations?** More components = more latency, cost, and tuning. Diminishing returns if the base corpus/embeddings are weak.

---

## 8. Prompt Engineering

**Q1. What is prompt engineering?**
Designing the input text to an LLM to reliably get the desired output — via clear instructions, context, examples, and formatting constraints.

**Q2. Key techniques (name them):**
- **Zero-shot** — just ask.
- **Few-shot** — include a few input→output examples to show the pattern.
- **Chain-of-Thought (CoT)** — "think step by step" to improve reasoning.
- **Role/system prompting** — "You are an expert ML tutor…".
- **Output formatting** — "Respond in JSON with keys …".
- **Delimiters & context** — clearly separate instructions from data.

**Q3. Why does few-shot / CoT help?**
Examples anchor the format and task; CoT gives the model "space" to reason through intermediate steps, which improves accuracy on math/logic/multi-step tasks.

**Q4. Example from your project:** The system prompt tells the model its **domain** (ML/AI), its **sources** (Wikipedia corpus), and to **cite sources** — a form of role + grounding prompt engineering.

**Q5. Limitations?** Brittle (small wording changes shift output), model-specific, doesn't add new knowledge (that's RAG), and long prompts cost tokens. Prompt injection is a security risk.

---

## 9. LLM Internals

**Q1. What is the core architecture of an LLM?**
The **Transformer** (2017, "Attention Is All You Need"). Key parts: token **embeddings + positional encodings**, stacked layers of **multi-head self-attention** + **feed-forward networks**, with residual connections and layer norm.

**Q2. What is self-attention, intuitively?**
For each token, attention computes how much every *other* token matters to it, using **Query, Key, Value** vectors: `softmax(Q·Kᵀ / √d) · V`. This lets the model capture **long-range context** and word relationships (e.g., resolve what "it" refers to).

**Q3. What is multi-head attention?**
Run attention **several times in parallel** with different learned projections ("heads"), each capturing different relationships (syntax, coreference, etc.), then concatenate. More expressive than single attention.

**Q4. Why positional encodings?**
Attention is order-agnostic by itself, so we **inject position information** so the model knows word order ("dog bites man" ≠ "man bites dog").

**Q5. Encoder vs decoder architectures?**
- **Encoder-only (BERT):** bidirectional, great for understanding/classification.
- **Decoder-only (GPT/LLaMA):** autoregressive, great for **generation** (predict next token).
- **Encoder-decoder (T5):** translation/summarization.

**Q6. What are parameters and why does "70B" matter?**
Parameters are the learned weights. More parameters ≈ more capacity/capability (and more compute/memory). "70B" = 70 billion — the size of the LLaMA model in your project.

**Q7. What is the context window?**
The max number of tokens the model can attend to at once. Limits how much you can put in a prompt — a key reason RAG (retrieve *relevant* bits) beats "paste everything."

**Q8. Key limitations from internals?** Quadratic attention cost with sequence length, fixed context window, no built-in memory, and hallucination from probabilistic generation.

---

## 10. Fine-Tuning (LoRA / PEFT)

**Q1. What is fine-tuning?**
Continuing training of a pretrained model on a **smaller, task-specific dataset** to adapt its behavior, style, or domain.

**Q2. What is the problem with *full* fine-tuning of an LLM?**
Updating **all** 70B parameters needs huge GPU memory, is slow/expensive, and produces a full copy of the model per task. Impractical for most teams.

**Q3. What is PEFT?**
**Parameter-Efficient Fine-Tuning** — freeze most of the model and train only a **small number of extra parameters**. You get most of the benefit at a fraction of the cost.

**Q4. What is LoRA and how does it work?**
**Low-Rank Adaptation.** Instead of updating a big weight matrix `W`, LoRA freezes `W` and learns two tiny **low-rank matrices A and B** whose product `A·B` is added to `W`. Only A and B are trained (often <1% of params). At inference you add the small delta. Massive memory/compute savings.

**Q5. What is QLoRA?**
LoRA on top of a **4-bit quantized** base model — lets you fine-tune very large models on a single consumer GPU.

**Q6. Fine-tuning vs RAG — when to use which?**
- **RAG** → inject **knowledge/facts** (updatable, cite-able).
- **Fine-tuning** → teach **behavior, format, tone, or a skill** the model lacks.
- They're **complementary** (RAG for what to know, fine-tuning for how to act).

**Q7. Limitations?** Needs quality labeled data; risk of **catastrophic forgetting**; can bake in bias; still needs GPUs (less than full FT). Doesn't keep knowledge fresh like RAG.

---

## 11. LLM Evaluation

**Q1. Why is evaluating LLMs hard?**
Outputs are **open-ended** — there's rarely a single correct string. Fluent text can be **factually wrong**. So classic accuracy doesn't fully apply.

**Q2. Categories of evaluation?**
- **Reference-based** (compare to a gold answer): BLEU, ROUGE, exact match.
- **Semantic**: embedding/cosine similarity (what your project uses).
- **LLM-as-judge**: use a strong LLM to score answers (RAGAS, G-Eval).
- **Human evaluation**: gold standard but slow/subjective.
- **Task benchmarks**: MMLU, HellaSwag, TruthfulQA, etc.

**Q3. How do you evaluate a *RAG* system specifically?**
Separate **retrieval** vs **generation** quality:
- **Context Relevance / Precision** — are retrieved chunks on-topic?
- **Context Recall** — did retrieval get all needed info?
- **Faithfulness/Groundedness** — does the answer stick to the retrieved context (no hallucination)?
- **Answer Relevance** — does it actually answer the question?
(Frameworks: **RAGAS**, TruLens, DeepEval.)

**Q4. What does *your* project use, and its honest limitation?**
Two **cosine-similarity** metrics — context relevance (query↔chunks) and recall (answer↔chunks) — computed locally, no API. **Limitation:** cosine similarity measures **semantic overlap, not factual correctness** — it won't catch a fluent-but-wrong answer. RAGAS/human eval go deeper.

**Q5. What is BLEU/ROUGE?**
N-gram overlap metrics for translation (BLEU) and summarization (ROUGE) — cheap but shallow (penalize valid paraphrases).

---

## 12. ReAct Framework

**Q1. What is ReAct?**
**Reason + Act** — a prompting pattern where the LLM **interleaves reasoning ("Thought")** with **actions (tool calls, "Action")** and **observations** of the tool results, looping until it can answer.

**Q2. What does the loop look like?**
```
Thought: I need the current weather.
Action: call weather_api("Delhi")
Observation: 34°C, sunny
Thought: Now I can answer.
Answer: It's 34°C and sunny in Delhi.
```
The model **thinks**, **acts** (uses a tool), **observes**, and repeats.

**Q3. Why is ReAct powerful?**
It combines **reasoning** (chain-of-thought) with **tool use** (accessing live/external info), so the agent can solve tasks it can't do from memory alone — and its reasoning trace is **inspectable/debuggable**.

**Q4. ReAct vs plain Chain-of-Thought?**
CoT only *reasons* internally; ReAct also *acts* on the world (search, calculator, APIs) and grounds its reasoning in real observations — reducing hallucination.

**Q5. Where used / relation to your work?**
It's the conceptual backbone of tool-using agents. Your **Week-8 single-agent pipeline** is a simplified version: detect intent → act via a tool (calculator, etc.) → return result.

**Q6. Limitations?** Can loop or get stuck; more LLM calls = more latency/cost; needs good tool descriptions; error handling matters.

---

## 13. LangGraph

**Q1. What is LangGraph?**
A framework to build LLM applications as a **graph (state machine)**: **nodes** are steps (functions/LLM calls), **edges** define flow, and a shared **state** object is passed and updated through the graph. Built for **stateful, multi-step, possibly cyclic** agent workflows.

**Q2. Why LangGraph over plain LangChain chains?**
LangChain chains are mostly **linear**. LangGraph supports **branching, loops, conditional routing, and persistent state** — needed for real agents (e.g., "if the tool fails, retry"; "loop until done").

**Q3. Core concepts?**
- **State** — a shared dict all nodes read/write.
- **Nodes** — units of work.
- **Edges / conditional edges** — control flow, including cycles.

**Q4. Example (your Final Project):** A 4-node StateGraph — `memory → rag → graph → model` — where each node enriches a shared `state` dict, and the final node calls the LLM. Maps 1:1 to the UI's pipeline animation.

**Q5. When is LangGraph overkill?**
For a strictly linear, simple flow, plain function calls suffice. LangGraph shines when you need **branching/loops/state/human-in-the-loop**.

**Q6. Limitations?** Extra abstraction/learning curve; debugging graph state can be tricky.

---

## 14. AutoGen, Tool Use & Function Calling

**Q1. What is "function calling" / "tool use"?**
The ability of an LLM to **decide to call an external function** and return **structured arguments** (usually JSON) for it, instead of answering in prose. The app runs the function and feeds the result back. This connects the LLM to calculators, APIs, databases, search, etc.

**Q2. How does function calling actually work?**
You give the model a list of tools with **name, description, and a JSON schema of parameters**. The model outputs which tool to call and the arguments. Your code executes it and returns the result; the model then produces the final answer.
*Example:* user asks "weather in Delhi?" → model emits `get_weather(city="Delhi")` → you call the API → feed result back → model answers.

**Q3. What is AutoGen?**
A **multi-agent** framework (Microsoft) where multiple LLM "agents" with different roles **converse to solve a task** — e.g., a "Coder" agent and a "Reviewer"/"Executor" agent talking to each other, using tools, until the task is done.

**Q4. Single-agent vs multi-agent (AutoGen)?**
- **Single-agent** (your Week-8 assignment): one agent routes to tools.
- **Multi-agent** (AutoGen): several specialized agents collaborate/debate, which can solve more complex tasks but adds cost and coordination complexity.

**Q5. Why structured (JSON) output matters?**
So a **program** can reliably parse and act on the response (not just a human reading prose). Your Week-8 pipeline returns `{"type": ..., "result": ...}` for exactly this reason.

**Q6. Limitations?** The model can call the wrong tool or hallucinate arguments; needs validation, error handling, and good tool descriptions; multi-agent setups can loop, get expensive, or drift off-task.

---

## 15. Agent Evaluation

**Q1. Why is evaluating *agents* different from evaluating a single LLM answer?**
An agent takes **multiple steps** (reason → tool → observe → …). You must evaluate the **whole trajectory**, not just the final text: Did it pick the **right tool**? Pass **correct arguments**? Recover from errors? Reach the goal **efficiently**?

**Q2. What do you actually measure?**
- **Task success rate** — did it achieve the goal?
- **Tool-selection accuracy** — right tool for the intent?
- **Argument correctness** — valid parameters?
- **Steps / efficiency / cost** — did it loop needlessly?
- **Robustness** — does it handle bad/empty input? (Your Week-8 empty-query → clean error is exactly this.)
- **Trajectory/faithfulness** — were the reasoning steps sound?

**Q3. How can you evaluate them in practice?**
- **Golden test sets** — queries with expected tool + output (your Week-8 test cases).
- **LLM-as-judge** — a strong model scores the trajectory.
- **Component metrics** — intent-classification accuracy, tool-call success.
- **Human review** for complex tasks.

**Q4. Relation to your Week-8 assignment?**
You evaluated with a **golden set of test queries**, checking that each routed to the correct intent/tool and returned the right structured result, plus logging for observability and an empty-input error case.

**Q5. Limitations?** Hard to build comprehensive test sets; non-determinism makes results vary; LLM-judges have their own biases; multi-step failures are hard to attribute.

---

## 16. Assignment Walkthroughs

**A. Autoencoder for Image Denoising**
- **Goal:** remove noise from images.
- **How:** add noise to inputs, train encoder-decoder to output the **clean** image (loss vs clean target).
- **Key ideas:** bottleneck forces feature learning; denoising AE learns noise-robust representations.

**B. Document Question Answering System (RAG)**
- **Goal:** answer questions over your own documents.
- **Pipeline:** load docs → chunk → embed → FAISS → retrieve top-k → LLM answers with sources.
- **Key ideas:** grounding, chunking, embeddings, retrieval, citing sources — the core RAG loop.

**C. Build an Agentic AI Pipeline (Week-8)**
- **Goal:** a single agent that routes queries to tools and returns structured JSON.
- **How:** intent detection (regex/keywords) → route to tool (calculator, keyword extractor, +bonus summarizer/word-count/unit-convert) → `{"type","result"}` output, with **logging** and **error handling**.
- **Key ideas:** intent routing, tool use, function-calling pattern, structured output, agent evaluation via test cases. This is the manual, from-scratch version of the ReAct/LangGraph concepts.

---

## 17. Rapid-Fire

- **Autoencoder** — reconstruct input via a bottleneck; learns compact features.
- **VAE** — probabilistic autoencoder; smooth latent space → can generate.
- **Reparameterization trick** — `z = μ + σ·ε` to allow backprop through sampling.
- **GAN** — generator vs discriminator adversarial game.
- **Mode collapse** — GAN generates limited variety.
- **FID** — GAN image-quality metric (lower = better).
- **SGD** — basic gradient step; **Momentum** — adds velocity; **AdaGrad** — per-param LR (decays); **RMSprop** — moving avg fixes decay; **Adam** — momentum + RMSprop (default).
- **Generative vs discriminative** — model `P(x)` vs `P(y|x)`.
- **RAG** — retrieve then generate; grounds the LLM.
- **Chunking** — split docs so each vector is one idea.
- **Embedding** — text → meaning vector; compare via cosine similarity.
- **Advanced RAG** — re-ranking, hybrid search, HyDE, multi-query.
- **Prompt engineering** — zero/few-shot, chain-of-thought, roles, formats.
- **Transformer** — attention-based architecture behind LLMs.
- **Self-attention** — Q·Kᵀ softmax · V; captures token relationships.
- **BERT** = encoder (understanding); **GPT/LLaMA** = decoder (generation).
- **LoRA** — train tiny low-rank matrices instead of all weights (PEFT).
- **QLoRA** — LoRA on a 4-bit quantized model.
- **RAG vs fine-tuning** — knowledge vs behavior.
- **LLM eval** — reference (BLEU/ROUGE), semantic (cosine), LLM-judge (RAGAS), human.
- **RAG eval** — context relevance, recall, faithfulness, answer relevance.
- **ReAct** — Thought → Action → Observation loop (reason + tools).
- **LangGraph** — agents as stateful graphs (branching/loops).
- **Function calling** — LLM emits structured tool calls (JSON args).
- **AutoGen** — multi-agent conversational framework.
- **Agent evaluation** — task success, tool-selection accuracy, efficiency, robustness.

---

*Study tip: For the meeting, be able to give the one-line answer first, then a 20-second "why/how + example." The rapid-fire section is your night-before refresher.*
