"""Cabezas de atención y atención multi-cabeza.

Los nombres de atributos (``attn`` / ``attn1``, ``attn2``, ``fusion``, ``qkv``,
``attn_heads``, ``out_proj``) son los de ``202506_Transformers`` para que sus
checkpoints carguen sin conversión.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ham.models.attention import build_attention
from ham.models.enrich import build_enrichment
from ham.models.fusion import build_fusion


class AttentionHead(nn.Module):
    """Una cabeza con una rama (homogénea) o varias ramas fusionadas (HAM).

    ``forward`` devuelve ``(out [B, N, d], weights)``, donde ``weights`` es el
    tensor de la rama o, en HAM, un ``dict`` ``{nombre_rama: tensor}``. Con una
    fusión a nivel de scores o una mezcla de salidas (``weighted_output``,
    ``gated_output``) el ``dict`` incluye además ``"fused"``, la atención que
    realmente se aplica a ``v``.

    Una rama puede repetirse (p. ej. ``[dot, dot]``, control de capacidad); la
    segunda aparición se reporta como ``dot_2``.
    """

    def __init__(self, head_dim, branches, fusion="concat_linear", fusion_kwargs=None, dropout=0.0):
        super().__init__()
        if not branches:
            raise ValueError("Se requiere al menos una rama de atención")
        self.branches = list(branches)
        self.names = unique_names(self.branches)

        if len(self.branches) == 1:
            self.attn = build_attention(self.branches[0], head_dim, dropout)
            self.fusion = None
        else:
            for i, name in enumerate(self.branches, start=1):
                setattr(self, f"attn{i}", build_attention(name, head_dim, dropout))
            self.fusion = build_fusion(fusion, head_dim, len(self.branches), dropout, **(fusion_kwargs or {}))
            if self.fusion.level == "score":
                no_scores = [n for i, n in enumerate(self.branches, 1) if not hasattr(getattr(self, f"attn{i}"), "scores")]
                if no_scores:
                    raise ValueError(
                        f"La fusión '{fusion}' combina scores [N, N]; las ramas {no_scores} no los tienen"
                    )

    @property
    def is_heterogeneous(self) -> bool:
        return self.fusion is not None

    def forward(self, q, k, v):
        if not self.is_heterogeneous:
            return self.attn(q, k, v)
        if self.fusion.level == "score":
            scores = [getattr(self, f"attn{i}").scores(q, k) for i in range(1, len(self.branches) + 1)]
            out, fused = self.fusion(scores, v)
            weights = {name: torch.softmax(s, dim=-1) for name, s in zip(self.names, scores)}
            weights["fused"] = fused
            return out, weights
        outputs, weights = [], {}
        for i, name in enumerate(self.names, start=1):
            out, w = getattr(self, f"attn{i}")(q, k, v)
            outputs.append(out)
            weights[name] = w
        if not hasattr(self.fusion, "mixing"):
            return self.fusion(outputs), weights
        # Mezcla de salidas: Σ c_b A_b v = (Σ c_b A_b) v, así que la atención aplicada es exacta.
        # Se reporta normalizada (÷ Σ c_b) para que siga siendo una distribución (sum_output: Σ c_b = k).
        coeffs = self.fusion.mixing(outputs)
        n = q.size(1)
        fused = sum(c * as_matrix(w, n) for c, w in zip(coeffs, weights.values()))
        weights["fused"] = fused / sum(coeffs)
        return self.fusion(outputs, coeffs), weights


def unique_names(branches) -> list[str]:
    """``[dot, dot, additive]`` -> ``[dot, dot_2, additive]``."""
    seen: dict[str, int] = {}
    out = []
    for b in branches:
        seen[b] = seen.get(b, 0) + 1
        out.append(b if seen[b] == 1 else f"{b}_{seen[b]}")
    return out


def as_matrix(w: torch.Tensor, n: int) -> torch.Tensor:
    """Pesos de una rama -> ``[B, N, N]``. Pooling (``[B, N]``) da la misma fila para todo token."""
    return w.unsqueeze(-2).expand(*w.shape[:-1], n, w.size(-1)) if w.dim() == 2 else w


def effective_weights(w, n: int) -> torch.Tensor:
    """Atención que la cabeza aplica a ``v``, ``[B, N, N]``: la rama única, ``fused`` o,
    en ``concat_linear`` (que mezcla salidas con una matriz), la media de las ramas."""
    if not isinstance(w, dict):
        return as_matrix(w, n)
    if "fused" in w:
        return w["fused"]
    return torch.stack([as_matrix(x, n) for x in w.values()]).mean(0)


class MultiHeadAttention(nn.Module):
    """``head_branches`` (opcional) da a cada cabeza su propia lista de ramas, p. ej.
    ``[[dot], [dot], [additive], [additive]]``: heterogeneidad *entre* cabezas, que
    mezcla ``out_proj``. Si es ``None`` todas las cabezas usan ``branches``.

    ``enrich`` (opcional) es una lista de enriquecimientos (``ham.models.enrich``) que se
    aplican, en ese orden, a la salida de todas las cabezas antes de ``out_proj``;
    ``enrich_kwargs`` da los argumentos de cada uno: ``{local: {kernel: 5}}``."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        branches,
        fusion="concat_linear",
        fusion_kwargs=None,
        attn_dropout=0.0,
        proj_dropout=0.0,
        qkv_bias=True,
        head_branches=None,
        enrich=None,
        enrich_kwargs=None,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim debe ser divisible por num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if head_branches is None:
            head_branches = [branches] * num_heads
        elif len(head_branches) != num_heads:
            raise ValueError(f"head_branches tiene {len(head_branches)} cabezas y el modelo {num_heads}")

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_heads = nn.ModuleList(
            AttentionHead(self.head_dim, b, fusion, fusion_kwargs, attn_dropout) for b in head_branches
        )
        for i, head in enumerate(self.attn_heads):  # ramas con sesgo posicional por cabeza (gpsa)
            for m in head.modules():
                if hasattr(m, "init_locality"):
                    m.init_locality(i, num_heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(proj_dropout)
        enrich_kwargs = enrich_kwargs or {}
        unknown = set(enrich_kwargs) - set(enrich or [])
        if unknown:
            raise ValueError(f"enrich_kwargs para enriquecimientos no usados: {sorted(unknown)}")
        # Con el RNG aparte: el resto del modelo se inicializa igual que sin enriquecer (misma semilla).
        with torch.random.fork_rng(devices=[]):
            self.enrich = nn.ModuleList(
                build_enrichment(name, embed_dim, num_heads, attn_dropout, **(enrich_kwargs.get(name) or {}))
                for name in (enrich or [])
            )

    def forward(self, x, return_attn=False):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, d]

        outs, weights = [], []
        for i, head in enumerate(self.attn_heads):
            out, w = head(q[:, i], k[:, i], v[:, i])
            outs.append(out)
            weights.append(w)

        if self.enrich:
            out = torch.stack(outs, dim=1)  # [B, H, N, d]
            for e in self.enrich:
                out = e(x, q, k, v, out)
            outs = out.unbind(1)
        out = self.dropout(self.out_proj(torch.cat(outs, dim=-1)))
        if not return_attn:
            return out
        if len({tuple(h.names) for h in self.attn_heads}) > 1:  # cabezas heterogéneas: siempre dicts
            weights = [w if isinstance(w, dict) else {h.names[0]: w} for w, h in zip(weights, self.attn_heads)]
        return out, _stack_heads(weights, N)


def _stack_heads(weights, n: int):
    """Apila pesos por cabeza en la dimensión 1 -> ``[B, H, ...]`` (o dict de ellos).

    Con cabezas heterogéneas (``head_branches``, ``weights`` como dicts con ramas
    distintas) cada rama se apila sobre las cabezas que la tienen y ``"fused"`` es la
    atención efectiva de *todas* las cabezas, ``[B, H, N, N]``.
    """
    keys = [tuple(w) if isinstance(w, dict) else None for w in weights]
    if all(k == keys[0] for k in keys):
        if keys[0] is not None:
            return {name: torch.stack([w[name] for w in weights], dim=1) for name in weights[0]}
        return torch.stack(weights, dim=1)
    per_name: dict[str, list] = {}
    for w in weights:
        for k, x in w.items():
            if k != "fused":
                per_name.setdefault(k, []).append(x)
    out = {k: torch.stack(v, dim=1) for k, v in per_name.items()}
    out["fused"] = torch.stack([effective_weights(w, n) for w in weights], dim=1)
    return out
