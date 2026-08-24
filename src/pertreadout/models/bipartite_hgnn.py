"""A bidirectional bipartite heterogeneous GNN over perturbation and gene nodes.

This is the model the original BMI 212 project set out to build.  What survives
in the reference material is a prototype notebook with unimplemented edge
functions and an out-of-memory error, plus a separate CPU smoke test that patched
a gene->perturbation layer into GEARS' own ``model.py``.  Neither was the
experiment, so this is a fresh implementation of the stated intent:

    two node sets -- perturbations and genes -- with message passing in *both*
    directions, augmenting the Gene Ontology and co-expression priors GEARS uses
    with a learned perturbation-gene affinity.

Deliberately shared with the GEARS baseline
-------------------------------------------
The decoder (shared MLP -> per-gene head -> cross-gene mixing -> residual on the
control profile), the loss, the data, and the split are the same for both
models.  Only the graph encoder differs.  That is the point: if the two models
land within a few percent of each other on a like-for-like readout, the reader
should be able to see that it is because they are close, not because they were
trained differently.

Cost note
---------
The prototype's dense perturbation-gene edge set is ``num_genes x num_perts``
edges -- 1.3M for Norman -- which is what produced the reference notebook's OOM.
Here the bipartite structure is split into a sparse structural part (a
perturbation is linked to the gene it targets, which is defined for *unseen*
perturbations too) carried by ``SAGEConv`` in both directions, and a dense but
low-rank learned affinity that never materialises a per-edge message tensor.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, SGConv

from pertreadout.models.gears_baseline import PriorGraphs


class MLP(nn.Module):
    """Small MLP with optional batch norm on the hidden layers."""

    def __init__(self, sizes: Sequence[int], batch_norm: bool = True, last_act: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            last = i == len(sizes) - 2
            if not last or last_act:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(sizes[i + 1]))
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BipartiteHGNN(nn.Module):
    """Perturbation <-> gene bipartite GNN with a GEARS-shaped decoder.

    Parameters
    ----------
    num_genes, num_perts:
        Node counts for the two node types.
    pert2gene:
        ``{pert_index: gene_index}`` for perturbations whose target gene is among
        the measured genes.  These are the structural bipartite edges.  They are
        available for a perturbation the model has never seen, which is what lets
        the bipartite half of the graph do anything on held-out perturbations.
    priors:
        GO graph over perturbations and co-expression graph over genes.  Pass the
        identity variants (see ``gears_baseline.identity_graph``) for the
        no-prior ablation.
    """

    def __init__(
        self,
        num_genes: int,
        num_perts: int,
        pert2gene: Dict[int, int],
        priors: PriorGraphs,
        hidden_size: int = 64,
        decoder_hidden_size: int = 16,
        bipartite_rank: int = 32,
        pert_emb_lambda: float = 0.2,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_genes = num_genes
        self.num_perts = num_perts
        self.hidden_size = hidden_size
        self.pert_emb_lambda = pert_emb_lambda
        self.device = device

        # ---- node embeddings ------------------------------------------------
        self.gene_emb = nn.Embedding(num_genes, hidden_size, max_norm=True)
        self.gene_pos_emb = nn.Embedding(num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(num_perts, hidden_size, max_norm=True)

        # ---- within-type priors (same conditioning GEARS uses) --------------
        self.register_buffer("coexpress_index", priors.coexpress_edge_index)
        self.register_buffer("coexpress_weight", priors.coexpress_edge_weight)
        self.register_buffer("go_index", priors.go_edge_index)
        self.register_buffer("go_weight", priors.go_edge_weight)
        self.gene_prior_conv = SGConv(hidden_size, hidden_size, 1)
        self.pert_prior_conv = SGConv(hidden_size, hidden_size, 1)

        # ---- bipartite structure, both directions ---------------------------
        pairs = sorted(pert2gene.items())
        if pairs:
            p_idx = torch.tensor([p for p, _ in pairs], dtype=torch.long)
            g_idx = torch.tensor([g for _, g in pairs], dtype=torch.long)
        else:  # a dataset where no perturbed gene is measured; keep shapes valid
            p_idx = torch.zeros(0, dtype=torch.long)
            g_idx = torch.zeros(0, dtype=torch.long)
        self.register_buffer("edge_p2g", torch.stack([p_idx, g_idx], dim=0))
        self.register_buffer("edge_g2p", torch.stack([g_idx, p_idx], dim=0))
        self.conv_p2g = SAGEConv((hidden_size, hidden_size), hidden_size)
        self.conv_g2p = SAGEConv((hidden_size, hidden_size), hidden_size)

        # ---- learned perturbation-gene affinity, low rank -------------------
        self.affinity_gene = nn.Linear(hidden_size, bipartite_rank, bias=False)
        self.affinity_pert = nn.Linear(hidden_size, bipartite_rank, bias=False)
        self.affinity_scale = bipartite_rank ** -0.5

        # ---- fuse the two bipartite channels back into each node type -------
        self.gene_merge = MLP([hidden_size * 3, hidden_size, hidden_size], last_act=True)
        self.pert_merge = MLP([hidden_size * 3, hidden_size, hidden_size], last_act=True)
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_act=True)

        # ---- decoder: identical in shape to the GEARS baseline's -------------
        self.recovery = MLP([hidden_size, hidden_size * 2, hidden_size])
        self.indv_w1 = nn.Parameter(torch.rand(num_genes, hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(num_genes, 1))
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)
        self.cross_gene_state = MLP([num_genes, decoder_hidden_size, decoder_hidden_size])
        self.indv_w2 = nn.Parameter(torch.rand(1, num_genes, decoder_hidden_size + 1))
        self.indv_b2 = nn.Parameter(torch.rand(1, num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)

        self.bn_gene = nn.BatchNorm1d(hidden_size)
        self.bn_base = nn.BatchNorm1d(hidden_size)

    # ------------------------------------------------------------------ #
    def encode(self) -> tuple[torch.Tensor, torch.Tensor]:
        """One pass of the bipartite encoder; independent of the cell batch.

        Returns ``(gene_h, pert_h)`` of shapes ``(num_genes, H)`` and
        ``(num_perts, H)``.
        """
        idx_g = torch.arange(self.num_genes, device=self.gene_emb.weight.device)
        idx_p = torch.arange(self.num_perts, device=self.pert_emb.weight.device)

        gene_h = self.bn_gene(self.gene_emb(idx_g)).relu()
        pos = self.gene_prior_conv(self.gene_pos_emb(idx_g), self.coexpress_index, self.coexpress_weight)
        gene_h = gene_h + self.pert_emb_lambda * pos

        pert_h = self.pert_prior_conv(self.pert_emb(idx_p), self.go_index, self.go_weight)

        # sparse structural messages, both directions
        msg_g = self.conv_p2g((pert_h, gene_h), self.edge_p2g)
        msg_p = self.conv_g2p((gene_h, pert_h), self.edge_g2p)

        # dense learned affinity, never materialised per edge:
        # A = tanh(Wg gene_h @ (Wp pert_h)^T / sqrt(r))  -- (num_genes, num_perts)
        a = torch.tanh(
            self.affinity_gene(gene_h) @ self.affinity_pert(pert_h).T * self.affinity_scale
        )
        aff_g = a @ pert_h / max(self.num_perts, 1)
        aff_p = a.T @ gene_h / max(self.num_genes, 1)

        gene_h = self.gene_merge(torch.cat([gene_h, msg_g, aff_g], dim=-1))
        pert_h = self.pert_merge(torch.cat([pert_h, msg_p, aff_p], dim=-1))
        return gene_h, pert_h

    # ------------------------------------------------------------------ #
    def forward(self, data):
        """Predict post-perturbation expression for every cell in a PyG batch.

        The batch is GEARS' own cell-graph batch: ``data.x`` is the control
        profile the cell is decoded from, ``data.pert_idx`` the perturbation
        indices (``-1`` padding), and the model outputs a residual on ``x``.
        """
        x, pert_idx = data.x, data.pert_idx
        num_graphs = int(data.batch.max().item()) + 1

        gene_h, pert_h = self.encode()

        base = gene_h.unsqueeze(0).repeat(num_graphs, 1, 1)          # (B, G, H)

        # add each cell's perturbation embedding to every gene of that cell
        rows, perts = [], []
        for graph_i, idxs in enumerate(pert_idx):
            for j in idxs:
                j = int(j)
                if j != -1:
                    rows.append(graph_i)
                    perts.append(j)
        if rows:
            pooled = torch.zeros(num_graphs, self.hidden_size, device=base.device)
            pooled.index_add_(
                0, torch.tensor(rows, device=base.device), pert_h[torch.tensor(perts, device=base.device)]
            )
            active = torch.tensor(sorted(set(rows)), device=base.device)
            fused = self.pert_fuse(pooled[active])
            base[active] = base[active] + fused.unsqueeze(1)

        flat = self.bn_base(base.reshape(num_graphs * self.num_genes, -1))

        out = self.recovery(flat).reshape(num_graphs, self.num_genes, -1)
        out = (out.unsqueeze(-1) * self.indv_w1).sum(dim=2) + self.indv_b1   # (B, G, 1)

        cross = self.cross_gene_state(out.squeeze(-1))                        # (B, D)
        cross = cross.unsqueeze(1).expand(-1, self.num_genes, -1)
        out = torch.cat([out, cross], dim=2)
        out = (out * self.indv_w2).sum(dim=2) + self.indv_b2                  # (B, G)

        out = out.reshape(num_graphs * self.num_genes, 1) + x.reshape(-1, 1)
        return out.reshape(num_graphs, self.num_genes)


def build(
    pert_data,
    device: str = "cpu",
    hidden_size: int = 64,
    decoder_hidden_size: int = 16,
    bipartite_rank: int = 32,
    priors: Optional[PriorGraphs] = None,
    use_priors: bool = True,
):
    """Instantiate the bipartite model on prepared data.

    Mirrors ``gears_baseline.build``: returns ``(module, context)`` where the
    context carries the control profile and DE filter the shared loss needs.
    """
    from gears import GEARS

    from pertreadout.models.gears_baseline import build_prior_graphs

    if priors is None:
        priors = build_prior_graphs(pert_data, use_priors=use_priors)

    # GEARS' wrapper is the authority on node ordering, the control profile and
    # the DE filter used by the loss; reuse it rather than re-deriving them.
    wrapper = GEARS(pert_data, device=device)
    gene_list = wrapper.gene_list
    pert2gene = {i: gene_list.index(p) for i, p in enumerate(wrapper.pert_list) if p in gene_list}

    model = BipartiteHGNN(
        num_genes=wrapper.num_genes,
        num_perts=wrapper.num_perts,
        pert2gene=pert2gene,
        priors=priors,
        hidden_size=hidden_size,
        decoder_hidden_size=decoder_hidden_size,
        bipartite_rank=bipartite_rank,
        device=device,
    ).to(device)

    context = {
        "ctrl_expression": wrapper.ctrl_expression,
        "dict_filter": wrapper.dict_filter,
        "direction_lambda": 0.1,
        "num_genes": wrapper.num_genes,
        "num_perts": wrapper.num_perts,
        "priors": priors,
    }
    return model, context
