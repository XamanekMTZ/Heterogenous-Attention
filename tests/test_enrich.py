"""Enriquecimientos de la salida de las cabezas (ham.models.enrich)."""
import pytest
import torch

from ham import analysis as A
from ham.config import AttentionConfig, ModelConfig
from ham.models import build_model

ENRICH = [["local"], ["gate"], ["global"], ["local", "gate"], ["local", "global"], ["local", "global", "gate"]]


def tiny_config(enrich, branches=("dot",), fusion="concat_linear", head_branches=None, image_size=16,
                enrich_kwargs=None, dropout=0.1):
    return ModelConfig(
        image_size=image_size, patch_size=4, hidden_size=16, num_layers=2, num_attention_heads=2,
        num_classes=5, hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
        attention=AttentionConfig(branches=list(branches), fusion=fusion, head_branches=head_branches,
                                  enrich=enrich, enrich_kwargs=enrich_kwargs or {}),
    )


def randomize(model):
    """Pesos de los enriquecimientos lejos de la identidad, para que las pruebas los vean."""
    with torch.no_grad():
        for layer in model.encoder.layers:
            for p in layer.attention.enrich.parameters():
                p.normal_(0, 0.5)
    return model


@pytest.mark.parametrize("enrich", ENRICH)
@pytest.mark.parametrize("image_size", [16, 32])
def test_shapes_and_backward(enrich, image_size):
    model = randomize(build_model(tiny_config(enrich, image_size=image_size)))
    logits, attns = model(torch.randn(2, 3, image_size, image_size), output_attentions=True)
    assert logits.shape == (2, 5)
    n = 1 + (image_size // 4) ** 2
    assert attns[0].shape == (2, 2, n, n)  # los pesos de atención no cambian de forma
    logits.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("enrich", ENRICH)
def test_identity_at_init_and_same_base_weights(enrich):
    """Con la misma semilla, el modelo enriquecido arranca con los mismos pesos y logits que sin enriquecer."""
    torch.manual_seed(0)
    base = build_model(tiny_config([])).eval()
    torch.manual_seed(0)
    rich = build_model(tiny_config(enrich)).eval()
    rich_sd = rich.state_dict()
    for k, v in base.state_dict().items():
        assert torch.equal(v, rich_sd[k]), k
    x = torch.randn(3, 3, 16, 16)
    assert torch.allclose(base(x), rich(x), atol=1e-6)


@pytest.mark.parametrize("branches,fusion,head_branches", [
    (["dot", "additive"], "concat_linear", None),
    (["dot", "additive_free"], "weighted_output", None),
    (["dot", "additive_free"], "weighted_score", None),
    (["dot"], "concat_linear", [["dot"], ["additive"]]),
])
def test_works_with_ham(branches, fusion, head_branches):
    model = randomize(build_model(tiny_config(["local", "gate"], branches, fusion, head_branches)))
    model(torch.randn(2, 3, 16, 16)).sum().backward()


def test_local_only_mixes_neighbors_and_leaves_cls():
    from ham.models.enrich import LocalEnrichment
    e = LocalEnrichment(4, 2)
    with torch.no_grad():
        e.conv.weight.fill_(1.0)
    v = torch.zeros(1, 2, 1 + 16, 2)
    v[0, 0, 1 + 5] = 1.0  # parche (1, 1) de una rejilla 4x4
    d = e.delta(None, None, None, v)
    assert torch.all(d[:, :, 0] == 0)  # CLS
    touched = (d[0, 0, 1:, 0] != 0).nonzero().flatten().tolist()
    assert sorted(touched) == [0, 1, 2, 4, 5, 6, 8, 9, 10]  # vecindad 3x3 de (1, 1)
    assert torch.all(d[0, 1] == 0)  # otra cabeza: groups = H·d


def test_kernel_kwarg_and_invalid_options():
    model = build_model(tiny_config(["local"], enrich_kwargs={"local": {"kernel": 5}}))
    assert model.encoder.layers[0].attention.enrich[0].conv.kernel_size == (5, 5)
    with pytest.raises(ValueError):
        build_model(tiny_config(["local"], enrich_kwargs={"local": {"kernel": 4}}))
    with pytest.raises(ValueError):
        build_model(tiny_config(["nada"]))
    with pytest.raises(ValueError):
        build_model(tiny_config(["gate"], enrich_kwargs={"local": {"kernel": 3}}))


def test_enrich_contributions():
    model = randomize(build_model(tiny_config(["local", "global", "gate"]))).eval()
    out = A.enrich_contributions(model, torch.randn(4, 3, 16, 16))
    assert out["enrich"] == ["local", "global", "gate"]
    assert set(out["terms"]) == {"attention", "local", "global"}
    for layer in range(2):
        assert sum(v[layer] for v in out["terms"].values()) == pytest.approx(1.0, abs=1e-5)
    assert all(0 < m < 2 for m in out["gate"]["mean"])
    assert A.enrich_contributions(build_model(tiny_config([])), torch.randn(1, 3, 16, 16)) is None


def test_gate_at_init_is_one():
    model = build_model(tiny_config(["gate"])).eval()
    out = A.enrich_contributions(model, torch.randn(2, 3, 16, 16))
    assert out["gate"]["mean"] == pytest.approx([1.0, 1.0])
    assert out["gate"]["frac_below_0.5"] == [0.0, 0.0]
