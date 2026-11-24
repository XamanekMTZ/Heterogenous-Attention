from pathlib import Path

import pytest
import yaml

from ham.config import config_from_dict, load_config, save_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(p for p in CONFIGS.glob("*.yaml") if not p.name.startswith("_") and p.name != "base.yaml"))
def test_all_configs_load(path):
    cfg = load_config(path)
    assert cfg.name == path.stem
    assert cfg.model.attention.branches


def test_inheritance_and_overrides():
    cfg = load_config(CONFIGS / "food101_ham_concat.yaml", ["train.epochs=1", "model.attention.branches=[dot]"])
    assert cfg.data.name == "food101"
    assert cfg.model.image_size == 192 and cfg.train.batch_size == 128  # _food101.yaml
    assert cfg.train.weight_decay == 0.05  # base.yaml
    assert cfg.train.epochs == 1
    assert cfg.model.attention.branches == ["dot"]


def test_unknown_key_fails():
    with pytest.raises(KeyError):
        load_config(CONFIGS / "cifar10_dot.yaml", ["train.epoch=1"])


def test_roundtrip(tmp_path):
    cfg = load_config(CONFIGS / "cifar10_ham_concat.yaml", ["seed=3"])
    save_config(cfg, tmp_path / "c.yaml")
    again = config_from_dict(yaml.safe_load((tmp_path / "c.yaml").read_text()))
    assert again == cfg


def test_resume_action(tmp_path):
    from ham.train import resume_action
    from ham.utils.io import run_dir_for

    cfg = load_config(CONFIGS / "cifar10_dot.yaml", [f"train.output_dir={tmp_path}"])
    assert resume_action(cfg) == "new"
    run_dir = run_dir_for(cfg)
    run_dir.mkdir(parents=True)
    (run_dir / "train.log").write_text("cortado a medias")
    assert resume_action(cfg) == "restart"
    (run_dir / "test.json").write_text("{}")
    assert resume_action(cfg) == "skip"
