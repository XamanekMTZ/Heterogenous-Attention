"""Mecanismos de atención por cabeza (una "rama" de HAM).

Todas las ramas reciben ``q, k, v`` de forma ``[B, N, d]`` (una sola cabeza) y
devuelven ``(out [B, N, d], weights)``. La forma de ``weights`` depende de la rama
y queda documentada en cada clase.

Para añadir un mecanismo: definir la clase con firma ``__init__(head_dim, dropout)``
y decorarla con ``@register_attention("nombre")``.

Las ramas *por pares* (``[B, N, N]``) exponen además ``scores(q, k)``, los puntajes
antes del softmax; las fusiones a nivel de scores (``ham.models.fusion``) los usan.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

ATTENTION_REGISTRY: dict[str, type[nn.Module]] = {}


def register_attention(name: str):
    def deco(cls):
        if name in ATTENTION_REGISTRY:
            raise KeyError(f"Atención '{name}' ya registrada")
        ATTENTION_REGISTRY[name] = cls
        return cls

    return deco


def build_attention(name: str, head_dim: int, dropout: float = 0.0) -> nn.Module:
    try:
        cls = ATTENTION_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Atención '{name}' no soportada. Disponibles: {sorted(ATTENTION_REGISTRY)}"
        ) from None
    return cls(head_dim, dropout)


@register_attention("dot")
class DotProductAttention(nn.Module):
    """Atención *scaled dot-product* (Vaswani et al., 2017).

    weights: ``[B, N, N]``.
    """

    def __init__(self, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def scores(self, q, k):
        return q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)

    def forward(self, q, k, v):
        weights = torch.softmax(self.scores(q, k), dim=-1)
        out = self.dropout(weights) @ v
        return out, weights


@register_attention("additive")
class AdditiveAttention(nn.Module):
    """Atención aditiva *tipo pooling* — port fiel de ``202506_Transformers``.

    Se calcula un puntaje por token, ``s_j = w^T tanh(W_q q_j + W_k k_j) + b``
    (el token j se combina sólo consigo mismo, no hay pares i-j), se normaliza con
    softmax sobre los N tokens y se obtiene **un único** vector de contexto
    ``c = sum_j a_j v_j``, que se replica a las N posiciones. Es decir, todos los
    tokens reciben la misma salida de esta rama.

    Es la variante con la que se obtuvieron los resultados de 2025. Para la versión
    por pares de Bahdanau ver :class:`PairwiseAdditiveAttention`.

    weights: ``[B, N]``.
    """

    def __init__(self, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(head_dim, 1)  # nombre heredado: es el vector w (+ b)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        scores = self.v_proj(torch.tanh(self.q_proj(q) + self.k_proj(k)))  # [B, N, 1]
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(self.dropout(weights) * v, dim=1, keepdim=True)  # [B, 1, d]
        out = context.expand(-1, q.size(1), -1)
        return out, weights.squeeze(-1)


@register_attention("additive_pairwise")
class PairwiseAdditiveAttention(nn.Module):
    """Atención aditiva por pares (Bahdanau et al., 2015).

    ``e_ij = w^T tanh(W_q q_i + W_k k_j)``, softmax sobre j. Memoria O(B·N²·d).

    weights: ``[B, N, N]``.
    """

    def __init__(self, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.w = nn.Linear(head_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def scores(self, q, k):
        hidden = torch.tanh(self.q_proj(q).unsqueeze(2) + self.k_proj(k).unsqueeze(1))  # [B,N,N,d]
        return self.w(hidden).squeeze(-1)  # [B, N, N]

    def forward(self, q, k, v):
        weights = torch.softmax(self.scores(q, k), dim=-1)
        out = self.dropout(weights) @ v
        return out, weights


@register_attention("additive_free")
class ParameterFreeAdditiveAttention(nn.Module):
    """Atención aditiva por pares **sin parámetros propios**.

    ``e_ij = sum_d tanh(q_i + k_j)_d``, softmax sobre j. Es la variante de Bahdanau
    de los notebooks de ``20250401_ArticuloJCR`` (009–011): no tiene ``W_q``, ``W_k``
    ni ``w``; usa directamente las proyecciones q, k de la cabeza. Memoria O(B·N²·d).

    weights: ``[B, N, N]``.
    """

    def __init__(self, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def scores(self, q, k):
        return torch.tanh(q.unsqueeze(2) + k.unsqueeze(1)).sum(-1)  # [B, N, N]

    def forward(self, q, k, v):
        weights = torch.softmax(self.scores(q, k), dim=-1)
        out = self.dropout(weights) @ v
        return out, weights


@register_attention("additive_free_scaled")
class ScaledParameterFreeAdditiveAttention(ParameterFreeAdditiveAttention):
    """Como ``additive_free``, pero ``e_ij = sum_d tanh(q_i + k_j)_d / sqrt(d)``.

    ``sum_d tanh`` llega hasta ±d, mientras que ``dot`` divide entre sqrt(d); sin
    escalar, la magnitud de esta rama domina las fusiones de scores (λ deja de ser
    la proporción de la mezcla). Con 1/sqrt(d) ambas ramas quedan en el mismo orden.

    weights: ``[B, N, N]``.
    """

    def scores(self, q, k):
        return super().scores(q, k) / (q.size(-1) ** 0.5)


@register_attention("lsa")
class LocalitySelfAttention(nn.Module):
    """*Locality Self-Attention* (Lee et al., 2021, *Vision Transformer for Small-Size Datasets*).

    Producto punto con temperatura aprendible por cabeza (``exp(log_scale)``, inicial 1/√d,
    sin weight decay) y la diagonal enmascarada: cada token atiende a los demás, no a sí mismo.

    weights: ``[B, N, N]``.
    """

    def __init__(self, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(-0.5 * math.log(head_dim)))
        self.dropout = nn.Dropout(dropout)

    def scores(self, q, k):
        s = q @ k.transpose(-2, -1) * self.log_scale.exp()
        eye = torch.eye(q.size(1), dtype=torch.bool, device=q.device)
        return s.masked_fill(eye, float("-inf"))

    def forward(self, q, k, v):
        weights = torch.softmax(self.scores(q, k), dim=-1)
        return self.dropout(weights) @ v, weights


@register_attention("gpsa")
class GatedPositionalSelfAttention(nn.Module):
    """*Gated Positional Self-Attention* (d'Ascoli et al., 2021, ConViT).

    ``A = (1 − σ(g))·softmax(q kᵀ/√d) + σ(g)·softmax(v_posᵀ r_ij)``, con ``r_ij = (δx, δy, δx² + δy²)``
    el desplazamiento entre parches de la rejilla. ``init_locality`` (llamado por
    ``MultiHeadAttention`` con el índice de la cabeza) centra cada cabeza en un desplazamiento
    distinto, como una convolución de √H × √H, y ``g = 1`` arranca con la parte posicional
    dominante. El CLS no tiene posición: su fila posicional es uniforme y ningún parche le
    asigna peso posicional.

    weights: ``[B, N, N]``.
    """

    def __init__(self, head_dim: int, dropout: float = 0.0, locality_strength: float = 1.0):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(3))
        self.gating = nn.Parameter(torch.ones(()))
        self.locality_strength = locality_strength
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def init_locality(self, head: int, num_heads: int):
        kernel = max(1, math.isqrt(num_heads))
        center = (kernel - 1) / 2
        h1, h2 = head % kernel, (head // kernel) % kernel
        self.pos.copy_(torch.tensor([2 * (h1 - center), 2 * (h2 - center), -1.0]) * self.locality_strength)

    def positional_scores(self, n: int, device) -> torch.Tensor:
        g = math.isqrt(n - 1)
        if g * g != n - 1:
            raise ValueError(f"gpsa requiere 1 CLS + rejilla cuadrada de parches (N={n})")
        ys, xs = torch.meshgrid(torch.arange(g, device=device), torch.arange(g, device=device), indexing="ij")
        xs, ys = xs.flatten().float(), ys.flatten().float()
        dx, dy = xs[None, :] - xs[:, None], ys[None, :] - ys[:, None]
        rel = torch.stack([dx, dy, dx**2 + dy**2], dim=-1)  # [P, P, 3]
        s = torch.zeros(n, n, device=device)
        s[1:, 1:] = rel @ self.pos
        s[1:, 0] = float("-inf")
        return s

    def forward(self, q, k, v):
        content = torch.softmax(q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5), dim=-1)
        positional = torch.softmax(self.positional_scores(q.size(1), q.device), dim=-1)
        gate = torch.sigmoid(self.gating)
        weights = (1 - gate) * content + gate * positional.unsqueeze(0)
        return self.dropout(weights) @ v, weights
