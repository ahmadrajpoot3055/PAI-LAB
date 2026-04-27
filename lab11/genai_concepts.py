"""
================================================================
1. LangChain
================================================================
ANALOGY: LangChain is the Express.js of AI apps — a lightweight
framework that wires together LLMs, tools, and data so you don't
have to build from scratch every time.
What LangChain solves:
  Without LangChain: Manually concatenate prompts, manage API calls,
  parse outputs, retry on errors, track conversation history.
  With LangChain: Declare a chain with components; the framework
  orchestrates the flow automatically.
Key abstractions:
  PromptTemplate  – Parametrized prompt with placeholders
  LLMChain        – Prompt → LLM → Output Parser in one object
  SequentialChain – Chain A output becomes Chain B input
  RouterChain     – Dispatch to specialized sub-chains by topic
  ReAct Agent     – "Reason → Act → Observe" loop using tools
Real-world scenario: A legal assistant that:
  1. Accepts a contract PDF
  2. Splits + embeds it into ChromaDB
  3. For each user question, retrieves relevant clauses
  4. Passes clauses + question to GPT → returns legal summary
================================================================
2. LLMs (Large Language Models)
================================================================
ANALOGY: An LLM is like a distilled model of human thought,
trained by reading the equivalent of millions of books.
What makes an LLM "large":
  • Parameters: GPT-3=175B, GPT-4≈1T, LLaMA-3-70B=70B
  • Training tokens: GPT-3 trained on 300B tokens; LLaMA-3 on 15T
  • Compute: GPT-4 training estimated at $100M+
Transformer self-attention (the core mechanism):
  For each token, compute Attention = softmax(QK^T / √d_k) × V
  This lets every token "attend" to every other token in context.
Emergent abilities (only appear at scale):
  • Multi-step reasoning (chain-of-thought prompting)
  • In-context learning (learn from examples in the prompt)
  • Tool use (call functions, browse web)
  • Code generation (explain → write → debug → test)
================================================================
3. RAG (Retrieval-Augmented Generation)
================================================================
ANALOGY: RAG is an open-book exam for the LLM.
  Closed-book LLM: "I'll answer from memory, but I might hallucinate."
  RAG-enabled LLM: "Let me look it up in the documents first."
Detailed chunk strategy:
  Document → split into 512-token chunks with 50-token overlap
  Why overlap? Avoids splitting a key sentence across chunk boundary.
RAG failure modes and fixes:
  Problem: Retrieved chunks are irrelevant
  Fix: Better embedding model / re-ranking with cross-encoder
  Problem: Answer ignores retrieved context
  Fix: More explicit prompt: "Answer ONLY using the provided context."
  Problem: Hallucination still occurs
  Fix: Add citation requirement → LLM must quote the source
Hybrid RAG (best practice):
  Dense retrieval (embedding similarity) + Sparse retrieval (BM25)
  → Re-rank combined results with cross-encoder
  → Feed top-3 to LLM
================================================================
4. FAISS (Facebook AI Similarity Search)
================================================================
ANALOGY: FAISS is like a GPS for a 768-dimensional space.
Given your location (query vector), find the 5 nearest cities
(document vectors) out of millions — in milliseconds.
How IVF (Inverted File Index) works:
  1. Training: K-means cluster all vectors into nlist cells
  2. Indexing: Assign each vector to its nearest cell centroid
  3. Querying: Embed query → find nprobe nearest centroids
               → search only vectors in those cells → return top-k
Speed vs accuracy tradeoff:
  nprobe=1  → fastest but may miss relevant results
  nprobe=50 → slower but much more accurate
  nprobe=nlist → equivalent to exact search (IndexFlatL2)
GPU acceleration:
  FAISS supports GPU (faiss-gpu) for 10-100x speedup,
  enabling billion-scale vector search in real-time.
================================================================
5. Vector
================================================================
ANALOGY: A vector is the fingerprint of a piece of content.
Just as a fingerprint uniquely identifies a person, an embedding
vector uniquely represents the semantic meaning of text.
Contextual vs static embeddings:
  Word2Vec: "bank" → same vector regardless of context
  BERT:     "bank" (financial) → different vector than
            "bank" (river) because BERT sees full sentence context
Dimensionality and meaning:
  Each dimension in the vector captures some latent concept.
  Dimension 42 might encode "formality level".
  Dimension 156 might encode "technical vs. casual register".
  (Dimensions are not human-interpretable individually, but
   collectively encode rich semantic structure.)
Embedding model sizes:
  Model                    | Dimensions | Best for
  -------------------------|------------|------------------
  MiniLM-L6-v2             | 384        | Fast, lightweight
  all-mpnet-base-v2        | 768        | High quality
  text-embedding-ada-002   | 1536       | OpenAI API
  text-embedding-3-large   | 3072       | Best accuracy
================================================================
6. VectorDB (Vector Database)
================================================================
ANALOGY: A VectorDB organizes books by topic similarity, not
alphabetically. Ask "find me books about AI safety" and it returns
books whose content is semantically closest to your query.
Architecture internals:
  Ingestion:  text → embedding model → float32[dim] → ANN index
  Storage:    Vectors + metadata (JSON) + optional payload filters
  Query:      query_vector → ANN(top-k) → filter metadata → return
Hybrid search (modern approach):
  Score = α × dense_score + (1−α) × sparse_score
  dense_score : cosine similarity of embeddings
  sparse_score: BM25 keyword relevance score
  α is tuned per use case (0.5 = balanced)
Choosing a VectorDB:
  Prototype / small data   → ChromaDB (zero config, local)
  Production / cloud       → Pinecone, Weaviate
  High performance / self-hosted → Qdrant, Milvus
  Already have NumPy data  → FAISS (fastest, in-memory)
================================================================
7. Generative AI (GenAI)
================================================================
ANALOGY: Traditional AI recognizes a cat in a photo.
GenAI draws the cat from scratch, given a text description.
How diffusion models work (DALL-E, Stable Diffusion):
  Training: Add Gaussian noise to real images, step by step
  (Forward process: image → noise over T=1000 steps)
  Train a U-Net to predict / reverse each noise step.
  Inference: Start from pure noise → denoise T times
  → Use text embedding to guide each denoising step (CLIP guidance)
Generative AI risks:
  • Deepfakes: AI-synthesized video/audio of real people
  • Misinformation: Convincing fake news at scale
  • Copyright: Models trained on copyrighted content
  • Job disruption: Writers, artists, coders impacted
Responsible GenAI practices:
  • Watermarking AI-generated content
  • Filtering training data for harmful content
  • Constitutional AI / RLHF alignment
  • Transparency about AI involvement
================================================================
8. GANs (Generative Adversarial Networks)
================================================================
ANALOGY: The Generator is a master counterfeiter printing fake
banknotes. The Discriminator is a detective trained to spot fakes.
As the detective improves, so does the counterfeiter.
Training dynamics:
  Early: G produces blurry blobs; D easily identifies fakes
  Mid:   G learns basic shapes/colors; D must look harder
  Late:  G produces photorealistic outputs; D is at ~50% accuracy
  (50% = D can't tell real from fake = Nash equilibrium)
Mode collapse (the main failure mode):
  G discovers one output that always fools D,
  and keeps producing only that output (no diversity).
  Fix: Mini-batch discrimination, Wasserstein loss (WGAN)
Modern GAN applications still relevant in 2025:
  • StyleGAN3 for synthetic training data generation
  • GANs for super-resolution (ESRGAN)
  • Voice conversion (voice cloning from seconds of audio)
  • Data augmentation for rare medical images
"""
print(__doc__)
import random, math
print("=== Toy GAN Simulation ===")
print("Target: Generate numbers near 7.0\n")
class Generator:
    def __init__(self): self.mean = random.uniform(0, 10)
    def generate(self): return random.gauss(self.mean, 1.2)
    def update(self, fb): self.mean += 0.4 * fb
class Discriminator:
    def score(self, x): return 1 / (1 + math.exp(abs(x - 7.0) - 1))
G, D = Generator(), Discriminator()
TARGET = 7.0
print(f"Generator starting mean: {G.mean:.2f}")
for epoch in range(1, 11):
    fake = G.generate()
    d_score = D.score(fake)
    fb = (1 - d_score) if fake < TARGET else -(1 - d_score)
    G.update(fb)
    print(f"  Epoch {epoch:2d} | fake={fake:5.2f} | D(fake)={d_score:.3f} | gen_mean={G.mean:.2f}")
print(f"\nFinal generator mean: {G.mean:.2f} (target: {TARGET})")
