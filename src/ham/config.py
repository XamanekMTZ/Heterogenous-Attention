"""Configuración tipada de experimentos.

Un experimento se describe con un YAML que puede heredar de otro con la clave
``base:`` (ruta relativa al propio YAML). Desde la línea de comandos se pueden
sobrescribir valores con ``seccion.clave=valor``; el valor se interpreta como YAML
(``train.epochs=1``, ``model.attention.branches=[dot,additive]``).

Las claves desconocidas producen un error: así un typo en un YAML no pasa
desapercibido y la configuración guardada en cada run es exactamente la usada.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_origin, get_type_hints

import yaml


@dataclass
class AttentionConfig:
    # Una rama -> atención homogénea; dos o más -> HAM (se fusionan por cabeza).
    # Ramas disponibles: ver ham.models.attention.ATTENTION_REGISTRY
    branches: list[str] = field(default_factory=lambda: ["dot"])
    # Fusión de las salidas de las ramas: ver ham.models.fusion.FUSION_REGISTRY
    fusion: str = "concat_linear"
    fusion_kwargs: dict[str, Any] = field(default_factory=dict)
    # Opcional: ramas propias de cada cabeza (heterogeneidad *entre* cabezas), p. ej.
    # [[dot], [dot], [additive], [additive]]. Si se da, reemplaza a ``branches``.
    head_branches: list[list[str]] | None = None
    # Enriquecimientos de la salida de las cabezas, en orden (ver ham.models.enrich.ENRICH_REGISTRY),
    # p. ej. [local, gate], y sus argumentos: {local: {kernel: 5}}.
    enrich: list[str] = field(default_factory=list)
    enrich_kwargs: dict[str, Any] = field(default_factory=dict)

    def per_head(self, num_heads: int) -> list[list[str]]:
        return self.head_branches if self.head_branches is not None else [self.branches] * num_heads


@dataclass
class ModelConfig:
    # vit | resnet (CNN de referencia, ver ham.models.cnn; ignora las claves de atención)
    arch: str = "vit"
    image_size: int = 32
    patch_size: int = 4
    num_channels: int = 3
    hidden_size: int = 48
    num_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int | None = None  # None -> 4 * hidden_size
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    qkv_bias: bool = True
    num_classes: int | None = None  # None -> lo determina el dataset
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    # Tokenización: patch (convolución de paso P) | spt (Shifted Patch Tokenization, Lee et al. 2021)
    # | conv_stem (convoluciones 3×3 de paso 2, Xiao et al. 2021)
    patch_embed: str = "patch"
    # > 0: convolución depthwise k×k entre las dos capas del MLP sobre la rejilla de parches (LocalViT)
    mlp_local_kernel: int = 0
    # arch=resnet: canales por etapa y bloques residuales por etapa
    cnn_widths: list[int] = field(default_factory=lambda: [16, 32, 64])
    cnn_blocks: int = 1

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size


@dataclass
class DataConfig:
    name: str = "cifar10"  # ver ham.datasets.DATASET_REGISTRY
    root: str = "data"
    # Fracción del conjunto de entrenamiento oficial reservada para validación
    # (selección de modelo). El test oficial sólo se usa una vez al final. Si el
    # dataset no trae test etiquetado (Stanford Cars) se reserva otra fracción igual.
    val_fraction: float = 0.1
    # Semilla del split train/val, independiente de la semilla de entrenamiento,
    # para que todas las semillas de un experimento compartan el mismo split.
    split_seed: int = 0
    normalization: str = "dataset"  # dataset | legacy_cifar | imagenet | none
    augment: str = "standard"  # standard | crop (sin volteo) | randaugment | legacy | none
    # Fracción del train (después de separar validación) que se usa, con un subconjunto fijo
    # de ``split_seed``: el mismo para todas las semillas y variantes.
    train_fraction: float = 1.0
    num_workers: int = 4


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    scheduler: str = "cosine"  # cosine | none
    warmup_epochs: int = 10
    amp: bool = False
    deterministic: bool = True
    device: str = "auto"  # auto | cuda | cpu
    output_dir: str = "runs"
    overwrite: bool = False
    # Mezcla de imágenes por lote: none | cutmix | mixup | cutmix_mixup (una de las dos al azar),
    # con λ ~ Beta(mix_alpha, mix_alpha) y aplicada con probabilidad mix_prob.
    mix: str = "none"
    mix_alpha: float = 1.0
    mix_prob: float = 1.0
    label_smoothing: float = 0.0


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_with_base(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("base", None)
    if base is None:
        return raw
    return _deep_merge(_load_yaml_with_base((path.parent / base).resolve()), raw)


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _from_dict(cls, data: dict, path: str = ""):
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        where = path or "raíz"
        raise KeyError(f"Claves desconocidas en '{where}': {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        tp = hints[name]
        if dataclasses.is_dataclass(tp) and isinstance(value, dict):
            value = _from_dict(tp, value, f"{path}.{name}".lstrip("."))
        elif tp is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        elif get_origin(tp) is list and value is not None and not isinstance(value, list):
            value = [value]
        kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(data: dict) -> ExperimentConfig:
    return _from_dict(ExperimentConfig, data)


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Carga un YAML (resolviendo ``base:``) y aplica overrides ``a.b=valor``."""
    path = Path(path).resolve()
    data = _load_yaml_with_base(path)
    data.setdefault("name", path.stem)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override inválido '{item}', se esperaba clave=valor")
        key, value = item.split("=", 1)
        _set_dotted(data, key, yaml.safe_load(value))
    return config_from_dict(data)


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)


__all__ = [
    "AttentionConfig",
    "ModelConfig",
    "DataConfig",
    "TrainConfig",
    "ExperimentConfig",
    "load_config",
    "save_config",
    "config_from_dict",
]
