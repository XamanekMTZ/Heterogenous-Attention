import pytest
import torch

from ham.config import AttentionConfig, ModelConfig
from ham.models import build_model

VARIANTS = [
    ["dot"],
    ["additive"],
    ["additive_pairwise"],
    ["dot", "additive"],
    ["dot", "additive_pairwise"],
    ["additive_free"],
    ["dot", "additive_free"],
    ["additive_free_scaled"],
]
# (ramas, fusión, fusion_kwargs) de las fusiones a nivel de scores
SCORE_VARIANTS = [
    (["dot", "additive_free"], "weighted_score", {"lambda_": 0.85}),
    (["dot", "additive_free"], "weighted_score", {"lambda_": 0.85, "learnable": True}),
    (["dot", "additive_pairwise"], "weighted_score", {"lambda_": 0.5}),
    (["dot", "additive_free"], "sum_score", {}),
    (["dot", "additive_free_scaled"], "weighted_score", {"lambda_": 0.85}),
    (["dot", "additive_free_scaled"], "sum_score", {}),
]


def tiny_config(branches, fusion="concat_linear", fusion_kwargs=None, dropout=0.1):
    return ModelConfig(
        image_size=16, patch_size=4, hidden_size=16, num_layers=2, num_attention_heads=2,
        num_classes=5, hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
        attention=AttentionConfig(branches=branches, fusion=fusion, fusion_kwargs=fusion_kwargs or {}),
    )


@pytest.mark.parametrize("branches", VARIANTS)
def test_forward_and_attention_shapes(branches):
    model = build_model(tiny_config(branches)).eval()
    x = torch.randn(3, 3, 16, 16)
    logits, attns = model(x, output_attentions=True)
    assert logits.shape == (3, 5)
    assert len(attns) == 2
    n = 1 + (16 // 4) ** 2
    expected = {"dot": (3, 2, n, n), "additive_pairwise": (3, 2, n, n), "additive_free": (3, 2, n, n),
                "additive_free_scaled": (3, 2, n, n), "additive": (3, 2, n)}
    for layer in attns:
        per_branch = layer if isinstance(layer, dict) else {branches[0]: layer}
        assert list(per_branch) == branches
        for name, w in per_branch.items():
            assert w.shape == expected[name]
            assert torch.allclose(w.sum(-1), torch.ones(w.shape[:-1]), atol=1e-5)


@pytest.mark.parametrize("branches", VARIANTS)
def test_backward(branches):
    model = build_model(tiny_config(branches))
    model(torch.randn(2, 3, 16, 16)).sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_unknown_branch_and_fusion():
    with pytest.raises(ValueError):
        build_model(tiny_config(["nope"]))
    cfg = tiny_config(["dot", "additive"])
    cfg.attention.fusion = "nope"
    with pytest.raises(ValueError):
        build_model(cfg)


@pytest.mark.parametrize("branches,fusion,kwargs", SCORE_VARIANTS)
def test_score_fusion_shapes_and_backward(branches, fusion, kwargs):
    model = build_model(tiny_config(branches, fusion, kwargs))
    logits, attns = model.eval()(torch.randn(3, 3, 16, 16), output_attentions=True)
    assert logits.shape == (3, 5)
    n = 1 + (16 // 4) ** 2
    for layer in attns:
        assert list(layer) == branches + ["fused"]
        for w in layer.values():
            assert w.shape == (3, 2, n, n)
            assert torch.allclose(w.sum(-1), torch.ones(w.shape[:-1]), atol=1e-5)
    model.train()(torch.randn(2, 3, 16, 16)).sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_weighted_lambda_extremes_match_single_branch():
    """λ=1 equivale a dot puro y λ=0 a additive_free puro (mismos pesos qkv)."""
    x = torch.randn(2, 3, 16, 16)
    for lam, single in [(1.0, "dot"), (0.0, "additive_free")]:
        torch.manual_seed(0)
        ham = build_model(tiny_config(["dot", "additive_free"], "weighted_score", {"lambda_": lam}, dropout=0.0)).eval()
        ref = build_model(tiny_config([single], dropout=0.0)).eval()
        # additive_free no tiene parámetros: los state_dict coinciden salvo por los nombres de rama
        ref.load_state_dict(ham.state_dict(), strict=True)
        with torch.no_grad():
            assert torch.allclose(ham(x), ref(x), atol=1e-6)


def test_additive_free_scaled_is_sqrt_d_scaling():
    from ham.models.attention import build_attention
    q, k = torch.randn(2, 7, 12), torch.randn(2, 7, 12)
    raw = build_attention("additive_free", 12).scores(q, k)
    scaled = build_attention("additive_free_scaled", 12).scores(q, k)
    assert torch.allclose(scaled, raw / 12 ** 0.5)


def test_learnable_lambda_is_parameter_and_logged():
    from ham.train import fusion_lambdas

    model = build_model(tiny_config(["dot", "additive_free"], "weighted_score", {"lambda_": 0.85, "learnable": True}))
    assert sum(n.endswith("lambda_logit") for n, _ in model.named_parameters()) == 2 * 2  # capas x cabezas
    lambdas = fusion_lambdas(model)
    assert len(lambdas) == 2 and all(abs(v - 0.85) < 1e-5 for layer in lambdas for v in layer)
    fixed = build_model(tiny_config(["dot", "additive_free"], "weighted_score", {"lambda_": 0.85}))
    assert fusion_lambdas(fixed) == []
    assert not any("lambda" in k for k in fixed.state_dict())  # λ fijo no va al checkpoint


@pytest.mark.parametrize("branches,fusion,kwargs", [
    (["dot", "additive"], "weighted_score", {}),  # additive (pooling) no tiene scores [N, N]
    (["dot", "additive_free", "additive_pairwise"], "weighted_score", {}),  # requiere 2 ramas
    (["dot", "additive_free"], "weighted_score", {"lambda_": 1.5}),
    (["dot", "additive_free"], "weighted_score", {"lambda_": 1.0, "learnable": True}),
])
def test_invalid_score_fusions(branches, fusion, kwargs):
    with pytest.raises(ValueError):
        build_model(tiny_config(branches, fusion, kwargs))
