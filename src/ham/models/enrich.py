"""Enriquecimientos de la salida de las cabezas (antes de ``out_proj``).

Un enriquecimiento modifica ``out = A·v`` de todas las cabezas a la vez, después de
las ramas y la fusión, con información que la atención no tiene: vecindad espacial
(``local``), una no linealidad por token (``gate``) o un contexto global (``global``).
Son ortogonales a ``branches``/``fusion``/``head_branches`` y no cambian los pesos de
atención, así que mapas, entropías y JS de ``ham.analyze`` siguen siendo los de las ramas.

Todos arrancan en la identidad (pesos o escala en 0, compuerta en 1): el modelo
inicial es exactamente el mismo sin enriquecer, y lo que aporten lo aprende.

Interfaz: ``forward(x [B,N,D], q, k, v [B,H,N,d], out [B,H,N,d]) -> out``. Los aditivos
exponen ``delta(x, q, k, v)`` (el término sumado) y ``gate`` expone ``values(x)``
(la compuerta), que usa ``ham.analysis.enrich_contributions``.

Para añadir uno: clase con firma ``__init__(embed_dim, num_heads, dropout, **kwargs)``
decorada con ``@register_enrichment("nombre")``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ham.models.attention import AdditiveAttention

ENRICH_REGISTRY: dict[str, type[nn.Module]] = {}


def register_enrichment(name: str):
    def deco(cls):
        if name in ENRICH_REGISTRY:
            raise KeyError(f"Enriquecimiento '{name}' ya registrado")
        ENRICH_REGISTRY[name] = cls
        return cls

    return deco


def build_enrichment(name: str, embed_dim: int, num_heads: int, dropout: float = 0.0, **kwargs) -> nn.Module:
    try:
        cls = ENRICH_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Enriquecimiento '{name}' no soportado. Disponibles: {sorted(ENRICH_REGISTRY)}"
        ) from None
    return cls(embed_dim, num_heads, dropout, **kwargs)


class _AdditiveEnrichment(nn.Module):
    """``out + delta``; las subclases definen ``delta``."""

    kind = "additive"

    def delta(self, x, q, k, v):
        raise NotImplementedError

    def forward(self, x, q, k, v, out):
        return out + self.delta(x, q, k, v)


@register_enrichment("local")
class LocalEnrichment(_AdditiveEnrichment):
    """``out += DWConv_k×k(v)`` sobre la rejilla de parches (LePE, Chu/Dong et al. 2022).

    Convolución *depthwise* (un filtro por canal de cada cabeza, ``groups = H·d``) sobre
    los tokens de parche ordenados en su rejilla √(N−1) × √(N−1); el CLS recibe 0. Da a
    cada cabeza el sesgo de localidad que la atención por pares no tiene, con
    ``H·d·(k²+1)`` parámetros por capa. En la última capa la pérdida sólo usa el CLS, así
    que ese término no recibe gradiente y se queda en 0.

    enrich_kwargs: ``kernel`` (impar, default 3).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, kernel: int = 3):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError(f"local: kernel debe ser impar (es {kernel})")
        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel, padding=kernel // 2, groups=embed_dim)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def delta(self, x, q, k, v):
        B, H, N, d = v.shape
        g = math.isqrt(N - 1)
        if g * g != N - 1:
            raise ValueError(f"local requiere 1 CLS + rejilla cuadrada de parches (N={N})")
        grid = v[:, :, 1:].permute(0, 1, 3, 2).reshape(B, H * d, g, g)
        patches = self.conv(grid).reshape(B, H, d, N - 1).permute(0, 1, 3, 2)
        return torch.cat([torch.zeros_like(v[:, :, :1]), patches], dim=2)


@register_enrichment("gate")
class GateEnrichment(nn.Module):
    """``out ⊙ 2σ(W_g x)``: compuerta por token, cabeza y canal (Qiu et al. 2025, *Gated Attention*).

    Agrega una no linealidad entre ``W_v`` y ``out_proj`` (que sin ella se componen en
    una sola matriz de bajo rango) y deja que cada token apague cabezas o canales. Con
    ``W_g = 0`` y sesgo 0 la compuerta vale 1. ``D² + D`` parámetros por capa.
    """

    kind = "gate"

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def values(self, x):
        """Compuerta ``[B, H, N, d]`` en (0, 2)."""
        B, N, D = x.shape
        g = 2.0 * torch.sigmoid(self.proj(x))
        return g.reshape(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)

    def forward(self, x, q, k, v, out):
        return out * self.values(x)


@register_enrichment("global")
class GlobalEnrichment(_AdditiveEnrichment):
    """``out += α ⊙ c_h``, con ``c_h`` el contexto de la atención aditiva *pooling* de cada cabeza.

    Es la información que aporta la rama ``additive`` de ``ham_concat`` (un único
    vector por cabeza, replicado a todos los tokens), pero sumada con una escala por
    canal α que arranca en 0 en lugar de concatenada y proyectada.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        d = embed_dim // num_heads
        self.pool = nn.ModuleList(AdditiveAttention(d, dropout) for _ in range(num_heads))
        self.alpha = nn.Parameter(torch.zeros(num_heads, 1, d))

    def delta(self, x, q, k, v):
        ctx = torch.stack([p(q[:, h], k[:, h], v[:, h])[0] for h, p in enumerate(self.pool)], dim=1)
        return self.alpha * ctx
