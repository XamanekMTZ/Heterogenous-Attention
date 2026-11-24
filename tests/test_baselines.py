"""Baselines publicados (LSA, GPSA, SPT, stem convolucional, LocalViT, ResNet) y mezcla de lotes."""
import pytest
import torch

from ham.config import AttentionConfig, ModelConfig
from ham.models import build_model
from ham.train import BatchMixer


def config(**kw):
    branches = kw.pop("branches", ["dot"])
    return ModelConfig(image_size=16, patch_size=4, hidden_size=16, num_layers=2, num_attention_heads=4,
                       num_classes=5, attention=AttentionConfig(branches=branches), **kw)


VARIANTS = [
    {"branches": ["lsa"]},
    {"branches": ["gpsa"]},
    {"patch_embed": "spt"},
    {"patch_embed": "conv_stem"},
    {"mlp_local_kernel": 3},
    {"patch_embed": "spt", "branches": ["lsa"]},
    {"arch": "resnet", "cnn_widths": [4, 8, 16]},
]


@pytest.mark.parametrize("kw", VARIANTS)
def test_forward_backward(kw):
    model = build_model(config(**dict(kw)))
    logits = model(torch.randn(3, 3, 16, 16))
    assert logits.shape == (3, 5)
    logits.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("branch", ["lsa", "gpsa"])
def test_attention_is_distribution(branch):
    model = build_model(config(branches=[branch])).eval()
    _, attns = model(torch.randn(2, 3, 16, 16), output_attentions=True)
    for w in attns:
        assert w.shape == (2, 4, 17, 17)
        assert torch.allclose(w.sum(-1), torch.ones(w.shape[:-1]), atol=1e-5)


def test_lsa_masks_diagonal():
    model = build_model(config(branches=["lsa"])).eval()
    _, attns = model(torch.randn(2, 3, 16, 16), output_attentions=True)
    assert torch.all(attns[0].diagonal(dim1=-2, dim2=-1) == 0)


def test_gpsa_heads_start_local_at_distinct_offsets():
    model = build_model(config(branches=["gpsa"]))
    heads = model.encoder.layers[0].attention.attn_heads
    pos = torch.stack([h.attn.positional_scores(17, "cpu").softmax(-1) for h in heads])
    # cada cabeza concentra la parte posicional en un vecino distinto del parche central
    peaks = pos[:, 1 + 4 * 1 + 1, 1:].argmax(-1)
    assert len(set(peaks.tolist())) == 4


@pytest.mark.parametrize("mode", ["cutmix", "mixup", "cutmix_mixup"])
def test_mixer(mode):
    mixer = BatchMixer(mode, alpha=1.0, prob=1.0, seed=0)
    x, y = torch.randn(8, 3, 16, 16), torch.arange(8)
    xm, ya, yb, lam = mixer(x, y)
    assert xm.shape == x.shape and 0.0 <= lam <= 1.0
    assert torch.equal(ya, y) and sorted(yb.tolist()) == list(range(8))


def test_mixer_none_is_identity():
    x, y = torch.randn(4, 3, 8, 8), torch.arange(4)
    xm, ya, yb, lam = BatchMixer("none", 1.0, 1.0, 0)(x, y)
    assert xm is x and lam == 1.0
