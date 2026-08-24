"""The GEARS baseline -- the published model, depended on rather than forked.

GEARS is MIT-licensed and installable (``pip install cell-gears==0.1.2``).  This
module does not copy any of it.  It borrows the pinned package's own graph
construction and its ``GEARS_Model``, so the baseline in this repo *is* GEARS,
not a reimplementation of it that could quietly differ.

Reference: Roohani, Huang & Leskovec, "Predicting transcriptional outcomes of
novel multigene perturbations with GEARS", Nature Biotechnology 42, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class PriorGraphs:
    """The two prior graphs GEARS conditions on.

    ``go`` links perturbations by Gene Ontology similarity; ``coexpress`` links
    genes by co-expression measured on the *training* conditions only.  The
    identity variant of each is what the ablation swaps in.
    """

    go_edge_index: torch.Tensor
    go_edge_weight: torch.Tensor
    coexpress_edge_index: torch.Tensor
    coexpress_edge_weight: torch.Tensor


def identity_graph(num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Self-loops only: a graph that carries no prior information at all."""
    idx = torch.arange(num_nodes, dtype=torch.long)
    return torch.stack([idx, idx], dim=0), torch.ones(num_nodes, dtype=torch.float32)


def build_prior_graphs(
    pert_data,
    use_priors: bool = True,
    coexpress_threshold: float = 0.4,
    k_neighbours: int = 20,
    cache: bool = True,
) -> PriorGraphs:
    """Build (or blank out) the GO and co-expression priors for a dataset.

    With ``use_priors=False`` both graphs become identity graphs.  That is the
    ablation the source project ran informally and reported as "GO priors added
    less than expected": here it is a supported configuration with seeds, so the
    claim can be checked rather than asserted.

    Building the GO graph means reading and filtering the full GO edge list,
    which costs minutes and is identical for every model and every variant on a
    given (dataset, split, seed).  It is cached next to the prepared data;
    ``cache=False`` forces a rebuild.
    """
    num_genes = len(pert_data.gene_names)
    num_perts = len(pert_data.pert_names)

    if not use_priors:
        gi, gw = identity_graph(num_perts)
        ci, cw = identity_graph(num_genes)
        return PriorGraphs(gi, gw, ci, cw)

    cache_file = (
        Path(pert_data.dataset_path)
        / f"priors_{pert_data.split}_{pert_data.seed}_{pert_data.train_gene_set_size}.pt"
    )
    if cache and cache_file.exists():
        return PriorGraphs(**torch.load(cache_file, weights_only=True))

    from gears.utils import GeneSimNetwork, get_similarity_network

    common = dict(
        adata=pert_data.adata,
        threshold=coexpress_threshold,
        k=k_neighbours,
        data_path=pert_data.data_path,
        data_name=pert_data.dataset_name,
        split=pert_data.split,
        seed=pert_data.seed,
        train_gene_set_size=pert_data.train_gene_set_size,
        set2conditions=pert_data.set2conditions,
    )

    co_edges = get_similarity_network(network_type="co-express", **common)
    co = GeneSimNetwork(co_edges, pert_data.gene_names.tolist(), node_map=pert_data.node_map)

    go_edges = get_similarity_network(
        network_type="go", pert_list=pert_data.pert_names.tolist(), **common
    )
    go = GeneSimNetwork(go_edges, pert_data.pert_names.tolist(), node_map=pert_data.node_map_pert)

    priors = PriorGraphs(go.edge_index, go.edge_weight, co.edge_index, co.edge_weight)
    if cache:
        torch.save(vars(priors), cache_file)
    return priors


def build(
    pert_data,
    device: str = "cpu",
    hidden_size: int = 64,
    priors: Optional[PriorGraphs] = None,
    use_priors: bool = True,
    num_go_gnn_layers: int = 1,
    num_gene_gnn_layers: int = 1,
    decoder_hidden_size: int = 16,
):
    """Instantiate the GEARS model on prepared data.

    Returns ``(module, context)``.  ``context`` carries the pieces the shared
    training loop needs -- the control expression vector and the DE filter that
    GEARS' direction-aware loss uses -- so the baseline and the bipartite model
    are trained by exactly the same code under exactly the same loss.
    """
    from gears import GEARS

    if priors is None:
        priors = build_prior_graphs(pert_data, use_priors=use_priors)

    wrapper = GEARS(pert_data, device=device)
    wrapper.model_initialize(
        hidden_size=hidden_size,
        num_go_gnn_layers=num_go_gnn_layers,
        num_gene_gnn_layers=num_gene_gnn_layers,
        decoder_hidden_size=decoder_hidden_size,
        G_go=priors.go_edge_index,
        G_go_weight=priors.go_edge_weight,
        G_coexpress=priors.coexpress_edge_index,
        G_coexpress_weight=priors.coexpress_edge_weight,
    )

    context = {
        "ctrl_expression": wrapper.ctrl_expression,
        "dict_filter": wrapper.dict_filter,
        "direction_lambda": wrapper.config["direction_lambda"],
        "num_genes": wrapper.num_genes,
        "num_perts": wrapper.num_perts,
        "priors": priors,
    }
    return wrapper.model.to(device), context
