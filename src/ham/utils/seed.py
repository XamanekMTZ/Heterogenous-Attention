"""Control de aleatoriedad.

Con ``deterministic=True`` se fuerzan algoritmos deterministas de cuDNN/cuBLAS:
dos corridas con la misma semilla, hardware y versiones dan métricas idénticas
(a costa de algo de velocidad). Diferentes GPUs o versiones de CUDA pueden dar
resultados ligeramente distintos; por eso se registra el entorno en cada run.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Requerido por cuBLAS para operaciones deterministas; debe fijarse antes del primer uso de CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` para DataLoader: deriva la semilla de cada worker de la de torch."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
