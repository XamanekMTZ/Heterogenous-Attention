"""Fuentes de datos. Cada constructor recibe ``(root, train_tf, eval_tf)`` y devuelve
un :class:`DatasetSource`.

``train_aug`` y ``train_eval`` son el *mismo* conjunto de entrenamiento oficial con
transformaciones distintas: de ahí se toman los índices de train (con augment) y de
validación (sin augment). ``test`` es el split oficial de evaluación, o ``None`` si
el dataset no lo trae etiquetado; en ese caso se reserva una fracción del train.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets as tvd

DATASET_REGISTRY: dict[str, callable] = {}
DATASET_STATS: dict[str, str] = {}  # nombre -> clave en transforms.NORMALIZATION


@dataclass
class DatasetSource:
    train_aug: Dataset
    train_eval: Dataset
    test: Dataset | None
    num_classes: int


def register_dataset(name: str, stats: str, aliases: tuple[str, ...] = ()):
    def deco(fn):
        for n in (name, *aliases):
            DATASET_REGISTRY[n] = fn
            DATASET_STATS[n] = stats
        return fn

    return deco


@register_dataset("cifar10", stats="cifar10")
def cifar10(root, train_tf, eval_tf):
    kw = dict(root=root, download=True)
    return DatasetSource(
        train_aug=tvd.CIFAR10(train=True, transform=train_tf, **kw),
        train_eval=tvd.CIFAR10(train=True, transform=eval_tf, **kw),
        test=tvd.CIFAR10(train=False, transform=eval_tf, **kw),
        num_classes=10,
    )


@register_dataset("cifar100", stats="cifar100", aliases=("cifar-100",))
def cifar100(root, train_tf, eval_tf):
    kw = dict(root=root, download=True)
    return DatasetSource(
        train_aug=tvd.CIFAR100(train=True, transform=train_tf, **kw),
        train_eval=tvd.CIFAR100(train=True, transform=eval_tf, **kw),
        test=tvd.CIFAR100(train=False, transform=eval_tf, **kw),
        num_classes=100,
    )


@register_dataset("svhn", stats="svhn")
def svhn(root, train_tf, eval_tf):
    """Split ``train`` (73,257 imágenes) y ``test`` (26,032); no usa ``extra``."""
    kw = dict(root=os.path.join(root, "svhn"), download=True)
    return DatasetSource(
        train_aug=tvd.SVHN(split="train", transform=train_tf, **kw),
        train_eval=tvd.SVHN(split="train", transform=eval_tf, **kw),
        test=tvd.SVHN(split="test", transform=eval_tf, **kw),
        num_classes=10,
    )


@register_dataset("plantvillage", stats="imagenet")
def plantvillage(root, train_tf, eval_tf):
    """Espera ``root/plantvillage/{train,val}/<clase>/*``. La carpeta ``val`` del
    archivo distribuido se usa como **test**; la validación sale de ``train``."""
    base = os.path.join(root, "plantvillage")
    train_aug = tvd.ImageFolder(os.path.join(base, "train"), transform=train_tf)
    return DatasetSource(
        train_aug=train_aug,
        train_eval=tvd.ImageFolder(os.path.join(base, "train"), transform=eval_tf),
        test=tvd.ImageFolder(os.path.join(base, "val"), transform=eval_tf),
        num_classes=len(train_aug.classes),
    )


class Food101(Dataset):
    """Espera ``root/food-101/{images,meta}`` (distribución original de ETH)."""

    def __init__(self, root, split, transform):
        with open(os.path.join(root, "meta", f"{split}.txt")) as f:
            self.paths = [line.strip() for line in f if line.strip()]
        classes = sorted({p.split("/")[0] for p in self.paths})
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rel = self.paths[idx]
        label = self.class_to_idx[rel.split("/")[0]]
        if not rel.lower().endswith(".jpg"):
            rel += ".jpg"
        img = Image.open(os.path.join(self.root, "images", rel)).convert("RGB")
        return (self.transform(img) if self.transform else img), label


@register_dataset("food101", stats="imagenet", aliases=("food-101",))
def food101(root, train_tf, eval_tf):
    base = os.path.join(root, "food-101")
    train_aug = Food101(base, "train", train_tf)
    return DatasetSource(
        train_aug=train_aug,
        train_eval=Food101(base, "train", eval_tf),
        test=Food101(base, "test", eval_tf),
        num_classes=len(train_aug.class_to_idx),
    )


class StanfordCars(Dataset):
    """Sólo ``cars_train_annos.mat`` trae etiquetas en el devkit disponible, así que
    train/val/test salen todos de ``cars_train`` (``test=None``)."""

    def __init__(self, root, transform):
        import scipy.io as sio

        devkit = os.path.join(root, "car_devkit", "devkit")
        self.annotations = sio.loadmat(os.path.join(devkit, "cars_train_annos.mat"))["annotations"][0]
        meta = sio.loadmat(os.path.join(devkit, "cars_meta.mat"))["class_names"][0]
        self.class_names = [c[0] for c in meta]
        self.images_dir = os.path.join(root, "cars_train", "cars_train")
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        img = Image.open(os.path.join(self.images_dir, anno["fname"][0])).convert("RGB")
        label = int(anno["class"][0][0]) - 1  # 1-based en el .mat
        return (self.transform(img) if self.transform else img), label


@register_dataset("stanfordcars", stats="imagenet", aliases=("stanford_cars", "cars"))
def stanford_cars(root, train_tf, eval_tf):
    base = os.path.join(root, "stanford_cars")
    train_aug = StanfordCars(base, train_tf)
    return DatasetSource(
        train_aug=train_aug,
        train_eval=StanfordCars(base, eval_tf),
        test=None,
        num_classes=len(train_aug.class_names),
    )
