import math

import pytest
import torch

from ham import analysis as A
from ham.config import AttentionConfig, ModelConfig
from ham.models import build_model


def tiny_config(branches, fusion="concat_linear", fusion_kwargs=None, dropout=0.1):
    return ModelConfig(
        image_size=16, patch_size=4, hidden_size=16, num_layers=2, num_attention_heads=2,
        num_classes=5, hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
        attention=AttentionConfig(branches=branches, fusion=fusion, fusion_kwargs=fusion_kwargs or {}),
    )


def test_normalize_attention_pooling_rows():
    w = torch.softmax(torch.randn(2, 3, 5), -1)
    full = A.normalize_attention(w)
    assert full.shape == (2, 3, 5, 5)
    assert torch.equal(full[:, :, 0], full[:, :, 4])
    assert torch.allclose(full.sum(-1), torch.ones(2, 3, 5))


def test_entropy_uniform_and_onehot():
    assert torch.allclose(A.entropy(torch.full((4, 7), 1 / 7)), torch.ones(4))
    assert torch.allclose(A.entropy(torch.eye(7)), torch.zeros(7), atol=1e-6)


def test_js_bounds():
    p = torch.softmax(torch.randn(3, 6), -1)
    assert torch.allclose(A.js_divergence(p, p), torch.zeros(3), atol=1e-6)
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    assert A.js_divergence(a, b).item() == pytest.approx(1.0, abs=1e-5)


def test_attention_distance():
    n = 1 + 16  # CLS + rejilla 4x4
    eye = torch.eye(n).expand(1, 2, n, n)
    assert torch.allclose(A.attention_distance(eye, patch_size=4), torch.zeros(1, 2))
    d = A.patch_distances(16, patch_size=4)
    assert d[0, 1].item() == pytest.approx(4.0)
    assert d[0, 15].item() == pytest.approx(math.hypot(12, 12))


def test_rollout_identity_and_rows():
    n = 5
    assert torch.allclose(A.rollout([torch.eye(n).expand(2, n, n)] * 3), torch.eye(n).expand(2, n, n))
    r = A.rollout([torch.softmax(torch.randn(2, n, n), -1) for _ in range(3)])
    assert torch.allclose(r.sum(-1), torch.ones(2, n), atol=1e-5)


def test_head_diversity_identical_heads():
    w = torch.softmax(torch.randn(2, 1, 5, 5), -1).expand(2, 4, 5, 5)
    assert torch.allclose(A.head_diversity(w), torch.zeros(2), atol=1e-6)


def test_classification_metrics_perfect():
    labels = torch.arange(10).repeat(3)
    logits = torch.full((30, 10), -1e4)
    logits[torch.arange(30), labels] = 1e4
    m = A.classification_metrics(logits, labels, 10)
    assert m["acc"] == pytest.approx(100.0)
    assert m["f1_macro"] == pytest.approx(100.0)
    assert m["ece"] == pytest.approx(0.0, abs=1e-4)
    assert sum(map(sum, m["confusion"])) == 30
    assert "top5" not in m


def test_classification_metrics_known_case():
    labels = torch.tensor([0, 0, 1, 1])
    logits = torch.tensor([[2.0, 0], [0, 2.0], [0, 2.0], [0, 2.0]])  # un error en la clase 0
    m = A.classification_metrics(logits, labels, 2)
    assert m["acc"] == pytest.approx(75.0)
    assert m["per_class_acc"] == pytest.approx([50.0, 100.0])
    assert m["precision_macro"] == pytest.approx(100 * (1.0 + 2 / 3) / 2)


def test_mcnemar_and_bootstrap():
    a = torch.tensor([True] * 10 + [False] * 2)
    assert A.mcnemar(a, a) == {"n10": 0, "n01": 0, "p_value": 1.0}
    b = torch.tensor([False] * 10 + [True] * 2)
    r = A.mcnemar(a, b)
    assert (r["n10"], r["n01"]) == (10, 2)
    assert r["p_value"] == pytest.approx(0.03857, abs=1e-4)
    bs = A.bootstrap_diff(a, b, n_boot=500)
    assert bs["ci_low"] <= bs["diff"] <= bs["ci_high"]


def test_delete_patches_and_random_mask():
    x = torch.ones(2, 3, 8, 8)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[:, 0] = True
    out = A.delete_patches(x, mask, patch_size=4)
    assert out[:, :, :4, :4].abs().sum() == 0 and out[:, :, 4:, :].min() == 1
    m = A.random_patch_mask(3, 16, 0.25, torch.Generator().manual_seed(0))
    assert m.sum(1).tolist() == [4, 4, 4]


def test_branch_contributions_concat_and_none():
    x = torch.randn(3, 3, 16, 16)
    model = build_model(tiny_config(["dot", "additive"], dropout=0.0)).eval()
    c = A.branch_contributions(model, x)
    assert set(c) == {"dot", "additive"}
    for i in range(2):
        assert c["dot"][i] + c["additive"][i] == pytest.approx(1.0, abs=1e-5)
    assert A.branch_contributions(build_model(tiny_config(["dot"])).eval(), x) is None
    score = build_model(tiny_config(["dot", "additive_free"], "sum_score")).eval()
    assert A.branch_contributions(score, x) is None


def test_attention_stats_on_model():
    model = build_model(tiny_config(["dot", "additive_free"], "weighted_score", {"lambda_": 1.0}, 0.0)).eval()
    with torch.no_grad():
        _, attns = model(torch.randn(2, 3, 16, 16), output_attentions=True)
    stats = A.AttentionStats(patch_size=4)
    stats.update(A.layer_dicts(attns, ["dot", "additive_free"]))
    r = stats.result()
    assert set(r["branches"]) == {"dot", "additive_free", "fused"}
    assert len(r["branches"]["dot"]["entropy_cls"]) == 2
    # con λ = 1 la atención fusionada es exactamente la de dot
    assert r["js_between"]["dot|fused"] == pytest.approx([0.0, 0.0], abs=1e-5)
