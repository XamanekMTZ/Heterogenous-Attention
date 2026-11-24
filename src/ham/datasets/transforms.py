from torchvision import transforms as T

NORMALIZATION = {
    # Estadísticas por canal reales del train de CIFAR-10.
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # Valores que usaba 202506_Transformers para *todos* los datasets
    # (std de CIFAR-10 "per-pixel", ampliamente copiada). Sólo para comparar con runs viejos.
    "legacy_cifar": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}


def resolve_normalization(mode: str, dataset_stats: str):
    """``mode='dataset'`` usa las estadísticas propias del dataset."""
    if mode == "none":
        return None
    key = dataset_stats if mode == "dataset" else mode
    try:
        return NORMALIZATION[key]
    except KeyError:
        raise ValueError(f"Normalización '{mode}' no soportada: {sorted(NORMALIZATION)} | dataset | none") from None


AUGMENTS = ("standard", "crop", "randaugment", "legacy", "none")


def build_transforms(image_size: int, train: bool, augment: str = "standard", normalization=None):
    if augment not in AUGMENTS:
        raise ValueError(f"augment '{augment}' no soportado ({' | '.join(AUGMENTS)})")
    if train and augment in ("standard", "crop", "randaugment"):
        # Crop con padding y flip a la misma escala que el test (lo estándar en CIFAR). ``crop`` no
        # voltea (dígitos de SVHN); ``randaugment`` agrega RandAugment (2 operaciones, magnitud 9).
        ops = [
            T.Resize((image_size, image_size)),
            T.RandomCrop(image_size, padding=4),
        ]
        if augment != "crop":
            ops.append(T.RandomHorizontalFlip(p=0.5))
        if augment == "randaugment":
            ops.append(T.RandAugment(num_ops=2, magnitude=9))
    elif train and augment == "legacy":
        # Igual que utils_transform.get_transforms(augment=True) del código original.
        # El train queda a otra escala que el test, que no se redimensiona a +8.
        ops = [
            T.Resize((image_size + 8, image_size + 8)),
            T.RandomCrop(image_size, padding=4),
            T.RandomHorizontalFlip(p=0.5),
        ]
    else:
        ops = [T.Resize((image_size, image_size))]
    ops.append(T.ToTensor())
    if normalization is not None:
        ops.append(T.Normalize(*normalization))
    return T.Compose(ops)
