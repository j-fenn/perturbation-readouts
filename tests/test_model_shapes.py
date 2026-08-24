"""Shape and wiring tests for the bipartite model.

No dataset and no GEARS wrapper required: the model is exercised on a synthetic
batch shaped exactly like the one GEARS' dataloader yields, so these run in CI
where the 3.5 GB of public data is not downloaded.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from pertreadout.models.bipartite_hgnn import BipartiteHGNN  # noqa: E402
from pertreadout.models.gears_baseline import PriorGraphs, identity_graph  # noqa: E402

N_GENES, N_PERTS, HIDDEN = 40, 6, 16


def make_model(**kwargs) -> BipartiteHGNN:
    go_i, go_w = identity_graph(N_PERTS)
    co_i, co_w = identity_graph(N_GENES)
    return BipartiteHGNN(
        num_genes=N_GENES,
        num_perts=N_PERTS,
        pert2gene={i: i * 3 for i in range(N_PERTS)},
        priors=PriorGraphs(go_i, go_w, co_i, co_w),
        hidden_size=HIDDEN,
        decoder_hidden_size=4,
        bipartite_rank=8,
        **kwargs,
    )


class FakeBatch:
    """The fields GEARS' cell-graph batch exposes, and nothing else."""

    def __init__(self, n_graphs: int, pert_idx):
        self.x = torch.randn(n_graphs * N_GENES, 1)
        self.y = torch.randn(n_graphs, N_GENES)
        self.batch = torch.arange(n_graphs).repeat_interleave(N_GENES)
        self.pert_idx = pert_idx
        self.pert = ["p"] * n_graphs


def test_identity_graph_is_self_loops_only():
    idx, w = identity_graph(5)
    assert idx.shape == (2, 5)
    assert torch.equal(idx[0], idx[1])
    assert torch.allclose(w, torch.ones(5))


def test_encoder_returns_one_embedding_per_node():
    model = make_model().eval()
    gene_h, pert_h = model.encode()
    assert gene_h.shape == (N_GENES, HIDDEN)
    assert pert_h.shape == (N_PERTS, HIDDEN)


def test_forward_returns_one_profile_per_cell():
    model = make_model().eval()
    batch = FakeBatch(4, [[0, -1], [1, 2], [-1, -1], [3, -1]])
    out = model(batch)
    assert out.shape == (4, N_GENES)
    assert torch.isfinite(out).all()


def test_forward_handles_an_all_control_batch():
    """Every cell unperturbed: there is no perturbation index to gather."""
    model = make_model().eval()
    out = model(FakeBatch(3, [[-1, -1]] * 3))
    assert out.shape == (3, N_GENES)
    assert torch.isfinite(out).all()


def test_output_is_a_residual_on_the_control_profile():
    """The model predicts a delta added to ``x``; shifting ``x`` shifts the output."""
    model = make_model().eval()
    batch = FakeBatch(2, [[0, -1], [1, -1]])
    with torch.no_grad():
        before = model(batch)
        batch.x = batch.x + 1.0
        after = model(batch)
    assert torch.allclose(after - before, torch.ones_like(before), atol=1e-5)


def test_different_perturbations_give_different_predictions():
    model = make_model().eval()
    a = FakeBatch(1, [[0, -1]])
    b = FakeBatch(1, [[4, -1]])
    b.x = a.x
    with torch.no_grad():
        assert not torch.allclose(model(a), model(b))


def test_gradients_reach_the_bipartite_parameters():
    """The learned perturbation-gene affinity must actually be trained."""
    model = make_model().train()
    batch = FakeBatch(4, [[0, -1], [1, 2], [3, -1], [4, -1]])
    loss = torch.nn.functional.mse_loss(model(batch), batch.y)
    loss.backward()
    for name in ("affinity_gene", "affinity_pert", "conv_p2g", "conv_g2p"):
        module = getattr(model, name)
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f"no gradient reached {name}"
        assert any(g.abs().sum() > 0 for g in grads), f"zero gradient at {name}"


def test_a_perturbation_with_no_measured_target_gene_is_allowed():
    """Some perturbed genes are not in the measured gene set; that must not crash."""
    go_i, go_w = identity_graph(N_PERTS)
    co_i, co_w = identity_graph(N_GENES)
    model = BipartiteHGNN(
        num_genes=N_GENES, num_perts=N_PERTS, pert2gene={},
        priors=PriorGraphs(go_i, go_w, co_i, co_w),
        hidden_size=HIDDEN, decoder_hidden_size=4, bipartite_rank=8,
    ).eval()
    out = model(FakeBatch(2, [[0, -1], [1, -1]]))
    assert out.shape == (2, N_GENES)
    assert torch.isfinite(out).all()
