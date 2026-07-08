---
title: Graph Attention Networks (GAT)
---

# Graph Attention Networks (GAT)

A GNN layer where each node updates itself by **attending to its neighbours**. The one sentence
that connects it to everything else in these notes: **GAT is self-attention masked by the graph's
adjacency matrix** — instead of every token attending to every token, every node attends only to
the nodes it is connected to. Same primitive (attention), different mask (the graph).

Everything here runs. It uses a dense adjacency matrix for clarity (real libraries use sparse
edge lists, but the math is identical).

---

## 1. The idea

A graph has nodes with feature vectors and edges saying which nodes are connected. A GAT layer
recomputes each node's vector as a **weighted average of its neighbours' (transformed) vectors**,
where the weights are learned attention coefficients:

- transform every node with a shared linear map `W`,
- for node *i*, score each neighbour *j* (how much should *i* listen to *j*),
- softmax those scores over *i*'s neighbours,
- aggregate: node *i*'s new vector is the attention-weighted sum of its neighbours' vectors.

The difference from transformer attention is **who is allowed to attend to whom**. In a
transformer every position is a neighbour of every position (full attention). In a graph, the
neighbours are given by the edges — so attention is **restricted to the adjacency structure**.

---

## 2. The math (additive attention)

For an edge from *j* to *i*, GAT computes an unnormalised score

```
e_ij = LeakyReLU( aᵀ [ W h_i ‖ W h_j ] )
```

where `W` is the shared linear transform and `a` is a learned attention vector. The `‖` is
concatenation of the two transformed endpoints. This is **additive (Bahdanau-style) attention**,
not the dot-product attention of transformers — GAT scores a *pair* of nodes with a small MLP
rather than a `QᵀK` inner product.

A useful algebra trick: because `aᵀ[Wh_i ‖ Wh_j] = a_srcᵀ(W h_i) + a_dstᵀ(W h_j)`, you can split
`a` into a **source** half and a **destination** half and compute each node's contribution once,
then broadcast to form all pairs. That is exactly what the implementation does.

Then normalise over *i*'s neighbours and aggregate:

```
α_ij = softmax_j( e_ij )   over j ∈ N(i)
h_i' = σ( Σ_j α_ij · W h_j )
```

The adjacency matrix enters as a **mask**: non-edges get `-inf` before the softmax, so they
receive exactly zero weight — the same `masked_fill(-inf)` trick as the causal mask in the
[decoder note](https://xiaonanzang.github.io/deep-learning-notes/transformer-decoder.html).

---

## 3. From-scratch implementation

```python
import torch
import torch.nn as nn

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=1, concat=True, alpha=0.2):
        super().__init__()
        self.n_heads, self.out_dim, self.concat = n_heads, out_dim, concat
        # shared linear transform W, producing n_heads independent projections at once
        self.W = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        # the attention vector a, split into its source and destination halves (see §2)
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

## 4. The bridge: GAT vs transformer attention

| | Transformer self-attention | GAT |
|---|---|---|
| Who attends to whom | every position ↔ every position (full) | only along **graph edges** (adjacency = mask) |
| Score function | **dot-product** `QᵀK / √dₖ` | **additive** `LeakyReLU(aᵀ[Wh_i ‖ Wh_j])` |
| Normalisation | softmax over all keys | softmax over a node's **neighbours** |
| Multi-head | yes | yes (identical idea) |
| "Positions" | tokens / patches / timesteps | **nodes** |

So a GAT is a transformer attention layer where the attention mask **is** the adjacency matrix. A
fully-connected graph makes GAT ≈ standard self-attention; a sparse graph restricts it. This is the
same "attention is attention" theme as ViT (tokens = patches) and the decoder (mask = causality):
here, **tokens = nodes, mask = the graph.**

**Contrast with non-attention GNNs.** GCN and GraphSAGE aggregate neighbours too, but with **fixed**
weights instead of learned attention:
- **GCN**: normalised **mean** of neighbours (weight `1/√(dᵢdⱼ)`, degree-based, not learned).
- **GraphSAGE**: sample a fixed number of neighbours, then mean / max / LSTM-pool them.
- Both are implemented with **scatter/`index_add_`** over an edge list rather than a dense
  adjacency, which is how real graph libraries scale to millions of edges.

GAT's contribution over these is exactly the attention: **the neighbour weights are learned and
input-dependent**, so a node can decide which neighbours matter for the current features rather
than weighting them all equally.

---

## The one bridge worth carrying

A graph attention layer is self-attention with the adjacency matrix as its mask, and an additive
score instead of a dot product. If you can code multi-head attention, you can code GAT — swap
`QᵀK` for the additive score and swap the causal/full mask for the graph. Nodes, patches, tokens,
timesteps: all the same primitive, differing only in what counts as a neighbour.

---

[← Back to contents](index.html) · [Transformer Architecture →](transformers.html)
