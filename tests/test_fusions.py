"""Mezclas de salidas, zscore_score, ramas repetidas y cabezas heterogéneas."""
import pytest
import torch

from ham import analysis as A
from ham.config import AttentionConfig, ModelConfig
from ham.models import build_model
from ham.models.fusion import ZScoreFusion
from ham.models.heads import AttentionHead, MultiHeadAttention

N = 1 + (16 // 4) ** 2


def tiny_config(branches, fusion="concat_linear", fusion_kwargs=None, dropout=0.1, head_branches=None):
    return ModelConfig(
        image_size=16, patch_size=4, hidden_size=16, num_layers=2, num_attention_heads=2,
        num_classes=5, hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
        attention=AttentionConfig(branches=branches, fusion=fusion, fusion_kwargs=fusion_kwargs or {},
                                  head_branches=head_branches),
    )


MIX_VARIANTS = [
    (["dot", "additive"], "weighted_output", {"lambda_": 0.5, "learnable": True}),
    (["dot", "additive_free"], "weighted_output", {"lambda_": 0.7}),
    (["dot", "additive_pairwise"], "weighted_output", {"lambda_": 0.5, "learnable": True, "branch_dropout": 0.2}),
    (["dot", "additive"], "gated_output", {}),
    (["dot", "additive_free"], "gated_output", {"branch_dropout": 0.1}),
    (["dot", "additive_free"], "zscore_score", {"learnable": True}),
    (["dot", "dot"], "concat_linear", {}),
    (["dot", "additive_free"], "sum_output", {}),
    (["dot", "additive"], "sum_output", {}),
]


@pytest.mark.parametrize("branches,fusion,kwargs", MIX_VARIANTS)
def test_new_fusions_shapes_fused_and_backward(branches, fusion, kwargs):
    model = build_model(tiny_config(branches, fusion, kwargs))
    logits, attns = model.eval()(torch.randn(3, 3, 16, 16), output_attentions=True)
    assert logits.shape == (3, 5)
    names = ["dot", "dot_2"] if branches == ["dot", "dot"] else branches
    for layer in attns:
        assert list(layer)[:2] == names
        if fusion != "concat_linear":
            assert layer["fused"].shape == (3, 2, N, N)
            assert torch.allclose(layer["fused"].sum(-1), torch.ones(3, 2, N), atol=1e-5)
    model.train()(torch.randn(2, 3, 16, 16)).sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("branches,fusion,kwargs", [
    (["dot", "additive"], "weighted_output", {"lambda_": 0.3}),
    (["dot", "additive_free"], "weighted_output", {"lambda_": 0.6}),
    (["dot", "additive"], "gated_output", {}),
    (["dot", "additive_free"], "sum_output", {}),
])
def test_mix_output_equals_fused_attention_times_v(branches, fusion, kwargs):
    """Mezclar salidas es aplicar (Σ c_b A_b) a v: el ``fused`` reportado (÷ Σ c_b) es exacto."""
    head = AttentionHead(8, branches, fusion, kwargs).eval()
    if fusion == "gated_output":
        torch.nn.init.normal_(head.fusion.gate.weight)  # compuerta no trivial
    q, k, v = torch.randn(3, 2, N, 8).unbind(0)
    out, w = head(q, k, v)
    total = len(branches) if fusion == "sum_output" else 1.0
    assert torch.allclose(out, total * w["fused"] @ v, atol=1e-5)


def test_sum_output_is_twice_weighted_output_half():
    torch.manual_seed(0)
    s = AttentionHead(8, ["dot", "additive_free"], "sum_output").eval()
    w = AttentionHead(8, ["dot", "additive_free"], "weighted_output", {"lambda_": 0.5}).eval()
    q, k, v = torch.randn(3, 2, N, 8).unbind(0)
    (out_s, ws), (out_w, ww) = s(q, k, v), w(q, k, v)
    assert torch.allclose(out_s, 2 * out_w, atol=1e-6)
    assert torch.allclose(ws["fused"], ww["fused"], atol=1e-6)


def test_weighted_output_extremes_match_single_branch():
    x = torch.randn(2, 3, 16, 16)
    for lam, single in [(1.0, "dot"), (0.0, "additive_free")]:
        torch.manual_seed(0)
        ham = build_model(tiny_config(["dot", "additive_free"], "weighted_output", {"lambda_": lam}, dropout=0.0)).eval()
        ref = build_model(tiny_config([single], dropout=0.0)).eval()
        ref.load_state_dict(ham.state_dict(), strict=True)
        with torch.no_grad():
            assert torch.allclose(ham(x), ref(x), atol=1e-6)


def test_zscore_is_invariant_to_branch_scale_and_shift():
    f = ZScoreFusion(8, 2)
    s1, s2 = torch.randn(2, N, N), torch.randn(2, N, N)
    base = f.combine([s1, s2])
    assert torch.allclose(base, f.combine([5.0 * s1 + 3.0, 0.5 * s2 - 2.0]), atol=1e-4)


def test_branch_dropout_one_hot_in_train_only():
    head = AttentionHead(8, ["dot", "additive"], "weighted_output", {"lambda_": 0.5, "branch_dropout": 0.999})
    outs = [torch.randn(64, N, 8), torch.randn(64, N, 8)]
    torch.manual_seed(0)
    c = head.fusion.train().mixing(outs)
    assert all(torch.all((x == 0) | (x == 1)) for x in c)
    assert torch.all(c[0] + c[1] == 1)
    c = head.fusion.eval().mixing(outs)
    assert float(c[0]) == pytest.approx(0.5)


def test_repeated_branches_are_independent():
    head = AttentionHead(8, ["dot", "dot"], "concat_linear")
    assert head.names == ["dot", "dot_2"]
    model = build_model(tiny_config(["dot", "dot"]))
    single = build_model(tiny_config(["dot"]))
    concat = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in single.parameters())
    assert concat == 2 * 2 * (2 * 8 * 8 + 8)  # capas x cabezas x Linear(2d -> d)


def test_head_branches_shapes_and_backward():
    cfg = tiny_config(["dot"], head_branches=[["dot"], ["additive"]], dropout=0.0)
    model = build_model(cfg)
    _, attns = model.eval()(torch.randn(3, 3, 16, 16), output_attentions=True)
    for layer in attns:
        assert layer["dot"].shape == (3, 1, N, N)
        assert layer["additive"].shape == (3, 1, N)
        assert layer["fused"].shape == (3, 2, N, N)
        assert torch.allclose(layer["fused"][:, 0], layer["dot"][:, 0])
        assert torch.allclose(layer["fused"][:, 1], layer["additive"][:, 0, None].expand(-1, N, -1))
    model.train()(torch.randn(2, 3, 16, 16)).sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    stats = A.AttentionStats(patch_size=4)
    stats.update(A.layer_dicts(attns, cfg.attention.branches))
    r = stats.result()
    assert set(r["branches"]) == {"dot", "additive", "fused"}
    assert set(r["js_between"]) == {"dot|additive", "dot|fused", "additive|fused"}
    with pytest.raises(ValueError):
        MultiHeadAttention(16, 2, ["dot"], head_branches=[["dot"]])


def test_contributions_and_lambdas_for_mix_fusions():
    from ham.train import fusion_lambdas

    x = torch.randn(3, 3, 16, 16)
    model = build_model(tiny_config(["dot", "additive"], "weighted_output",
                                    {"lambda_": 0.5, "learnable": True}, dropout=0.0)).eval()
    c = A.branch_contributions(model, x)
    assert set(c) == {"dot", "additive"}
    assert all(c["dot"][i] + c["additive"][i] == pytest.approx(1.0, abs=1e-5) for i in range(2))
    assert all(abs(v - 0.5) < 1e-5 for layer in fusion_lambdas(model) for v in layer)
    gated = build_model(tiny_config(["dot", "additive"], "gated_output", dropout=0.0)).eval()
    assert A.branch_contributions(gated, x) is not None
    split = build_model(tiny_config(["dot"], head_branches=[["dot"], ["additive"]])).eval()
    assert A.branch_contributions(split, x) is None


@pytest.mark.parametrize("branches,fusion,kwargs", [
    (["dot", "additive"], "zscore_score", {}),  # pooling no tiene scores [N, N]
    (["dot", "additive", "additive_free"], "weighted_output", {}),
    (["dot", "additive"], "weighted_output", {"branch_dropout": 1.0}),
    (["dot", "additive_free"], "zscore_score", {"scale": 0.0}),
    (["dot", "additive_free"], "sum_output", {"branch_dropout": 0.1}),
])
def test_invalid_new_fusions(branches, fusion, kwargs):
    with pytest.raises(ValueError):
        build_model(tiny_config(branches, fusion, kwargs))
