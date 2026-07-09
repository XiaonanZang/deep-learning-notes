---
title: Graph Attention Networks (GAT)
---

# Graph Attention Networks (GAT)

A GAT updates each node in a graph by **attending to its neighbours** — it is self-attention with
the graph's adjacency matrix as its mask. But the mechanism is the easy part. The useful questions
are *why a graph at all*, how this connects to the graph methods you already know, and how the
learned embeddings actually get used to predict. This note goes in that order:

**why graphs → from classical graph algorithms to learned message passing → the GAT mechanism +
code → how the embeddings get used → predicting on graphs you have never seen → two real systems.**

Everything here runs. It uses a dense adjacency matrix for clarity (real libraries use sparse edge
lists, but the math is identical).

---

## 1. Why a graph? Architecture follows data structure

The choice of architecture follows the **structure of the data** — this is the whole reason GNNs
exist as a separate family:

| Architecture | Assumes the data is a... | Examples |
|---|---|---|
| **CNN / UNet** | **grid** — regular, every pixel has the same 3×3 neighbourhood | images, segmentation, video |
| **RNN / Transformer** | **sequence** — ordered, or a set with positions | text, audio, time series |
| **GNN / GAT** | **arbitrary graph** — entities with irregular relationships | everything below |

A CNN works on an image because an image **is** a grid: fixed neighbours, a canonical order, a
kernel you can slide. A graph has none of that — each node has a *different number* of neighbours,
there is *no ordering*, no grid. So you cannot slide a kernel over it. GNNs generalise "aggregate
information from your neighbours" to **irregular neighbourhoods**. That is the entire point.

**The test for "is this a GNN problem?":** *are the relationships between entities part of the data,
and not a regular grid or sequence?* If yes, it is a graph.

**What is actually a graph:** social networks (users / friendships), molecules (atoms / bonds),
recommendation (users / items), knowledge graphs and GraphRAG (entities / relations), citation
networks (papers / citations), road networks (intersections / roads), fraud rings, physics meshes.

**Three task shapes** on a graph:
- **Node-level** — predict a property of each node (is this user a bot? what topic is this paper?).
- **Edge-level (link prediction)** — predict missing / future edges (friend recommendation,
  drug–target interaction, knowledge-graph completion).
- **Graph-level** — predict a property of the whole graph (is this molecule toxic?).

---

## 2. From classical graph algorithms to learned message passing

You have likely met graphs before as **algorithms that run on a graph with hand-designed structure
and weights**:

| Classical method | What it does | The "intelligence" is... |
|---|---|---|
| **Dijkstra / A\*** | shortest path given edge weights | the search algorithm; weights are given |
| **Graph cut** (min-cut / max-flow) | energy minimisation → image segmentation (pixels = nodes) | the hand-designed energy |
| **HMM** (Viterbi, forward–backward) | posterior over hidden states | the transition / emission probabilities you specify |
| **Particle filter** | sequential state estimation | the motion / observation models you write |

Common thread: **you design the model, then run a fixed inference or optimisation algorithm.** No
learning from data — the graph and its weights are given.

A **GNN flips exactly one thing**: instead of hand-designing the weights and running a fixed
algorithm, it **learns node representations from data** via *message passing*. This is the same
shift as CNN vs SIFT — hand-crafted features gave way to learned ones. Classical graph algorithms
give way to learned graph representations.

**The bridge you already own.** HMM forward–backward is a **message-passing** algorithm: each node
passes "messages" (beliefs) to its neighbours, iteratively, until they converge. That is *belief
propagation*. A GNN is a **learned, neural version of that**:

> **GNN message passing = belief propagation where the messages are learned instead of hand-derived.**

You already understand the mechanism (pass information along the edges, update each node from what
arrives). A GNN just replaces the fixed probabilistic messages with **learned neural transformations**.

### 2b. The objective stayed still; the encoder evolved

Zoom out and the last decade has one stable **objective** — *learn embeddings so related things have a
high dot product* — carried by a succession of **encoders**, each matched to the structure of the data:

| Data structure | Encoder that fits it | Era |
|---|---|---|
| sequence / co-occurrence | skip-gram (**word2vec** 2013 → **node2vec** 2014–16, transductive) | 2013–16 |
| grid (images) | **CNN** metric learning (FaceNet 2015, SimCLR 2020) | 2012– |
| set / sequence with learned relevance | **attention → Transformer** | 2015–17 |
| arbitrary graph | **GAT** = attention over neighbours | 2018 |

So contrastive embedding "by connection" is *old* (node2vec is literally word2vec on random walks over
a graph), and GAT is not a new objective — it is the **attention mechanism applied to graph
neighbourhood aggregation**. The GNN aggregator itself got progressively more expressive along the
same line: **GCN** (fixed degree-normalised mean) → **GraphSAGE** (learned transform + fixed pooling,
inductive + sampling) → **GAT** (learned, *input-dependent* per-neighbour weighting). That last step is
the flexibility attention buys: a fixed mean cannot tell a meaningful neighbour from a noisy one;
learned attention can.

---

## 3. The mechanism: message passing

One GNN layer, for every node *i*, does three things:

1. **Message** — each neighbour *j* sends a transformed version of its feature vector.
2. **Aggregate** — node *i* combines the incoming messages (sum / mean / attention).
3. **Update** — node *i* forms its new vector from the aggregate.

Stack **K** layers and each node sees its **K-hop** neighbourhood — one hop per layer. The
difference between GNN variants is *only* step 2, how neighbours are combined:

- **GCN** — a fixed, degree-normalised **mean** of neighbours.
- **GraphSAGE** — sample a fixed number of neighbours, then mean / max / pool.
- **GAT** — a **learned, attention-weighted** sum. ← this note.

---

## 4. GAT: attention over neighbours

GAT makes step 2 an attention: node *i* learns **how much to listen to each neighbour** instead of
averaging them equally. For an edge from *j* to *i*:

```
e_ij = LeakyReLU( aᵀ [ W h_i ‖ W h_j ] )      # unnormalised score for the pair (i, j)
α_ij = softmax_j( e_ij )   over j ∈ N(i)       # normalise over i's neighbours
h_i' = σ( Σ_j α_ij · W h_j )                    # attention-weighted sum of neighbours
```

`W` is a shared linear transform, `a` is a learned attention vector, `‖` is concatenation. This is
**additive (Bahdanau-style) attention** — GAT scores a *pair* of nodes with a small MLP, rather than
the `QᵀK` dot product of a transformer.

Algebra trick used in the code: since `aᵀ[Wh_i ‖ Wh_j] = a_srcᵀ(W h_i) + a_dstᵀ(W h_j)`, split `a`
into a **source** half and a **destination** half, compute each node's contribution once, and
broadcast to form all pairs.

**The adjacency matrix is the mask.** Non-edges get `-inf` before the softmax, so they receive
exactly zero weight — the same `masked_fill(-inf)` trick as the causal mask in the
[decoder note](https://xiaonanzang.github.io/deep-learning-notes/transformer-decoder.html). That
single line is what makes it a *graph* attention: **tokens = nodes, mask = the graph.**

---

## 5. From-scratch implementation

```python
import torch
import torch.nn as nn

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=1, concat=True, alpha=0.2):
        super().__init__()
        self.n_heads, self.out_dim, self.concat = n_heads, out_dim, concat
        # shared linear transform W, producing n_heads independent projections at once
        self.W = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        # the attention vector a, split into its source and destination halves (see §4)
        self.a_src = nn.Parameter(torch.empty(n_heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(n_heads, out_dim))
        self.leaky = nn.LeakyReLU(alpha)     # the nonlinearity inside the GAT score
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        # h:   (N, in_dim)  node features
        # adj: (N, N)       1 where an edge exists (include self-loops so a node attends to itself)
        N = h.shape[0]
        Wh = self.W(h).view(N, self.n_heads, self.out_dim)      # (N, H, out) transformed nodes

        # additive score split into source term (per i) and destination term (per j):
        #   e_ij = LeakyReLU( a_src·Wh_i + a_dst·Wh_j )
        s = (Wh * self.a_src).sum(-1)                           # (N, H)  source contribution
        d = (Wh * self.a_dst).sum(-1)                           # (N, H)  dest   contribution
        e = self.leaky(s[:, None, :] + d[None, :, :])           # (N, N, H) score for every pair (i,j)

        # the GRAPH is the mask: non-edges -> -inf -> zero weight after softmax
        e = e.masked_fill((adj == 0)[:, :, None], float('-inf'))
        alpha = torch.softmax(e, dim=1)                         # normalise over neighbours j

        # aggregate: h_i' = Σ_j alpha_ij · Wh_j   (contract over j)
        out = torch.einsum('ijh,jhd->ihd', alpha, Wh)           # (N, H, out)

        if self.concat:                                          # hidden layers: concatenate heads
            return out.reshape(N, self.n_heads * self.out_dim)   # (N, H*out)
        return out.mean(dim=1)                                    # final layer: average heads -> (N, out)

# smoke test on a 5-node ring graph (0-1-2-3-4-0) with self-loops
N, in_dim, out_dim, H = 5, 4, 8, 2
h = torch.randn(N, in_dim)
adj = torch.tensor([[1,1,0,0,1],
                    [1,1,1,0,0],
                    [0,1,1,1,0],
                    [0,0,1,1,1],
                    [1,0,0,1,1]], dtype=torch.float)
out = GATLayer(in_dim, out_dim, n_heads=H)(h, adj)
print(out.shape)                          # torch.Size([5, 16])  = (N, H*out)
```

Verified: attention rows sum to 1 over each node's neighbours, **non-edges receive exactly zero
weight**, and node 0 attends only to {0, 1, 4} (its ring neighbours + itself).

---

## 6. GAT vs transformer attention, and vs GCN / GraphSAGE

| | Transformer self-attention | GAT |
|---|---|---|
| Who attends to whom | every position ↔ every position (full) | only along **graph edges** (adjacency = mask) |
| Score function | **dot-product** `QᵀK / √dₖ` | **additive** `LeakyReLU(aᵀ[Wh_i ‖ Wh_j])` |
| Normalisation | softmax over all keys | softmax over a node's **neighbours** |
| Multi-head | yes | yes (identical idea) |
| "Positions" | tokens / patches / timesteps | **nodes** |

A fully-connected graph makes GAT ≈ standard self-attention; a sparse graph restricts it — the same
"attention is attention" theme as ViT (tokens = patches) and the decoder (mask = causality).

Against **GCN / GraphSAGE**, GAT's edge is the *learned, input-dependent* weighting: GCN uses a fixed
degree-based mean, GraphSAGE a fixed pooling, while GAT lets each node **decide which neighbours
matter** for the current features. (GCN/GraphSAGE scale via `scatter` / `index_add_` over sparse
edge lists — how real libraries reach millions of edges.)

---

## 7. How the embeddings get used: dot-product similarity

A GNN layer's *output* is a learned embedding per node. How does that become a prediction? The most
common answer — and the one that ties this note to the rest — is **dot-product similarity**.

Take **recommendation**. The graph is bipartite: users and items are nodes, an edge means "this user
interacted with this item." Message passing profiles each node — a user's embedding aggregates the
items they liked; an item's embedding aggregates the users who liked it (so *"users like me liked
this"* is baked in by the graph). Then the affinity score is simply:

```
score(user, item) = embedding_user · embedding_item      # dot product
```

High score → recommend. Ranking all items by this score is **nearest-neighbour search in a learned
embedding space**, and predicting a high-scoring pair *is* link prediction.

This is the same primitive you have now seen four times:

| Domain | "Correlate" = dot product between... |
|---|---|
| Attention | query · key |
| CLIP | image embedding · text embedding |
| Retrieval / RAG | query · document |
| **Recommendation (GNN)** | **user · item** |

All four: **learn embeddings so related things have a high dot product, unrelated things a low one.**
The embedding space is shaped by a **contrastive loss** (the InfoNCE idea from the
[Foundations note](https://xiaonanzang.github.io/deep-learning-notes/activations-losses-optimizers.html)):
pull a user toward the items they actually interacted with (positives), push away from sampled
negatives.

### At scale: you never compute all pairs

Scoring a user against *every* item is `O(M)` per query — impossible at millions of items. Neither
training nor serving does it:

- **Training — negative sampling.** For each positive `(user, item)` pair, sample a handful of random
  negatives instead of scoring all items. That is exactly what InfoNCE does — cross-entropy over one
  positive and a few sampled negatives, not the whole catalogue.
- **Serving — Approximate Nearest Neighbour (ANN) search.** Finding the top-k items by dot product is
  a nearest-neighbour problem, and it has mature **sublinear** index structures: graph-based **HNSW**
  (the current default), **LSH** hashing, and **IVF + product quantization** (what **FAISS** is built
  on). These power vector databases (FAISS, ScaNN, Pinecone, Milvus).

The separation of concerns is the point: **your model learns the embeddings; an off-the-shelf library
builds the index and serves retrieval.** The interface is just an `(M, d)` matrix:

```python
index = faiss.IndexHNSWFlat(d)         # or IndexIVFPQ for compression at scale
index.add(item_embeddings)             # BUILD (the library's job)
scores, ids = index.search(user_emb, k=200)   # RETRIEVE top-k, sublinear
```

Industrially this is **two-stage retrieve → rank**: ANN prunes millions to a few hundred candidates
cheaply, then a heavier ranker scores just those precisely. And it composes with §8's inductive
property — a **new item** gets an embedding from the inductive encoder (no retraining) and is simply
`index.add()`-ed, immediately retrievable. (Match the index metric to the training similarity: an
inner-product index for dot product, or normalise first for cosine.)

The same machinery *is* retrieval / RAG — embed the query, ANN-search a vector DB, feed the top-k to
an LLM. Recommendation, retrieval, and GraphRAG are one pattern: **learn embeddings, index them,
retrieve by similarity.**

---

## 8. Predicting on graphs you have never seen: transductive vs inductive

A subtle but production-defining question: the embeddings above were learned on **one known graph**.
What happens when a **new node** appears (a new user signs up) — or the whole graph is new?

- **Transductive** (original GCN, matrix factorisation, node2vec): learns a **fixed embedding per
  node**. A new node has no embedding → you must **retrain**. Fine for a static graph, useless when
  it grows.
- **Inductive** (GraphSAGE, **GAT**): learns the **aggregation *function*** — the weights `W` and the
  attention `a` — **not** per-node embeddings. A new node → gather its features and neighbours →
  apply the learned function → embedding, **no retraining**.

Look back at the code: the only parameters are `W`, `a_src`, `a_dst` — **functions of features**,
zero per-node lookup. Feed the layer *any* node with a neighbourhood and it produces an embedding.
That is why GAT is **inductive**, and why it works for streaming new users and for graphs it never
saw in training.

**How a new node actually gets its embedding — you never *initialize* it, you *supply and compute*
it.** A transductive model initialises a random per-node vector and *trains* it; an inductive model
does neither. The new node's **features** come from its own content/attributes (a new user's profile
or first content; a new item's description through a text encoder; a new document through the
encoder), not a learned per-node vector. Its **edges** come from observed interactions (the user's
first few clicks, the map's lane connectivity). You then gather both and apply the shared `W`, `a` →
its embedding, with no retraining. Cold-start edge case: a node with features but no edges still
embeds through its **self-loop**, sharpening as real neighbours arrive; a featureless, edgeless node
cannot be embedded, because the function has nothing to consume.

---

## 9. Two systems in practice

**(a) Large-scale recommendation / knowledge graphs (Huzefa's world).**
Industrial graphs — recommendation, fraud, and knowledge graphs — are exactly where GNNs earn their
keep (frameworks like **GraphStorm** and **GraphRAG** target enterprise-scale graphs). Take
recommendation: hundreds of millions of user and item nodes, billions of interaction edges. A GAT
layer profiles each node by message passing, and **attention decides which interactions matter** —
a user's one purchase of a laptop should weigh more than fifty idle clicks. New users arrive every
second, so the model *must* be inductive: compute the new user's embedding from their first few
edges and score items by dot product immediately. In a knowledge graph / GraphRAG setting, the same
machinery does **link prediction** — completing missing relations, or retrieving the relevant
subgraph to ground an LLM's answer.

**(b) The lane graph in autonomous driving (map topology).**
An HD map is naturally a graph: **lane segments (centrelines) are nodes**, and edges encode
connectivity — *successor*, *predecessor*, *left-neighbour*, *right-neighbour*. A GAT over this lane
graph lets each lane node attend to the lanes it connects to, encoding **map topology**: where a
lane leads, which lanes merge, which are legal to change into. This map encoding (LaneGCN /
VectorNet represent the road exactly this way) becomes the context a planner or motion-prediction
model reads to know where a vehicle *can* go. Crucially, **every map region is a different graph** —
so this is inductive by necessity: the model learns the lane-aggregation *function* and applies it
fresh to each new stretch of road, never a fixed per-lane embedding.

Both examples share the shape of the whole note: entities become nodes, relationships become edges,
attention learns which neighbours matter, and — because real graphs are always new — the model
learns a *function* over neighbourhoods rather than memorising a fixed graph.

---

## 10. Two practical gotchas: over-smoothing and scaling

Two questions come up constantly about real GNNs, and both are worth having an answer to.

**Over-smoothing — why GNNs are shallow.** Each layer mixes a node with its neighbours. Stack too
many and every node has aggregated an overlapping K-hop neighbourhood; in the limit, all nodes see
the whole graph and their embeddings **converge to the same vector** — the very thing you need to
tell apart washes out. So unlike CNNs and transformers (dozens of layers), GNNs are usually **2–3
layers deep**. Mitigations when you do need depth: **residual / skip connections**, **jumping
knowledge** (combine every layer's output, not just the last), and normalisation (e.g. PairNorm).
The one-liner: *GNNs go wide (big neighbourhoods per layer), not deep (few layers), because depth
causes over-smoothing.*

**Scaling — you cannot hold the whole graph in memory.** The dense `(N, N)` adjacency in this note is
`O(N²)` — fine for a demo, impossible for a billion-node graph. Two things fix it:
- **Sparse edge lists + scatter/gather** instead of a dense matrix (message passing only along real
  edges), which is how every production library (PyG, DGL, GraphStorm) is built.
- **Neighbour sampling** (GraphSAGE): for each node, sample a *fixed* number of neighbours per layer
  (say 25 then 10), so per-node compute is bounded and you can **mini-batch** — train on subgraphs
  instead of the whole graph at once. This is what makes inductive GNNs trainable at industrial
  scale, and it pairs naturally with the inductive property from §8 (a learned function applied to
  sampled neighbourhoods).

**How big is the model, really?** Typical sizes: hidden dim **64–256** (up to ~1024 at the high end),
**2–3 layers**, **4–8 heads**. All the message-passing weights together (every `W` and `a`) total only
**single-digit millions of parameters** — a `W` is roughly `in_dim × out_dim × heads`, e.g. 256×256×8
≈ 0.5M per layer. The rule stays small because it encodes a **function**, not the nodes: parameter
count is independent of `N`, so it does not grow with the graph. When you hear a production recommender
quoted at "billions of parameters," that mass is almost never the aggregation rule — it is the
**transductive ID-embedding tables** (one learned vector per user-ID and item-ID). Industrial systems
are often **hybrid**: a huge ID table (the memory hog) plus a small inductive GAT function on top. The
expressivity, meanwhile, comes from stacking + heads + already-rich input features, not from one `W`;
there is even a theoretical ceiling (message-passing GNNs are bounded by the Weisfeiler–Lehman test,
which is why variants like GIN exist), but it is rarely the practical bottleneck.

---

## The one bridge worth carrying

A graph attention layer is self-attention with the adjacency matrix as its mask and an additive
score instead of a dot product. Its message passing is a *learned* belief propagation — the same
"pass information along the edges" you knew from HMMs, with the messages now learned. Its output is
an embedding you compare by dot product, exactly like attention, CLIP, and retrieval. And because it
learns a *function* over neighbourhoods (inductive), it predicts on nodes and graphs it never saw —
new users, new roads. Nodes, patches, tokens, timesteps: the same primitive, differing only in what
counts as a neighbour.

---

[← Back to contents](index.html) · [Transformer Architecture →](transformers.html)
