import pytest
import torch
from PIL import Image

from ham.datasets.transforms import build_transforms


def _img(size):
    return Image.new("RGB", (size, size), color=(120, 60, 200))


@pytest.mark.parametrize("augment", ["standard", "legacy", "none"])
@pytest.mark.parametrize("src_size", [32, 256])
def test_output_size(augment, src_size):
    for train in (True, False):
        x = build_transforms(32, train=train, augment=augment)(_img(src_size))
        assert x.shape == (3, 32, 32)


def test_standard_keeps_test_scale():
    """Sin padding ni flip, el crop de 'standard' es la imagen de test; 'legacy' no."""
    img = Image.effect_noise((64, 64), 50).convert("RGB")
    test = build_transforms(32, train=False)(img)
    torch.manual_seed(0)
    std = build_transforms(32, train=True, augment="standard")
    hits = sum(torch.equal(std(img), test) for _ in range(2000))  # p = 1/162 por intento
    assert hits > 0  # crop centrado sin flip coincide exactamente con el test
    legacy = build_transforms(32, train=True, augment="legacy")
    assert not any(torch.equal(legacy(img), test) for _ in range(200))


def test_unknown_augment_fails():
    with pytest.raises(ValueError):
        build_transforms(32, train=False, augment="foo")
