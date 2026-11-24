"""Fusión de las ramas de atención dentro de una cabeza HAM.

Hay dos niveles de fusión (atributo de clase ``level``):

- ``"output"``: recibe ``outputs: list[Tensor [B, N, d]]`` (la salida de cada rama,
  ya pasada por su softmax) y devuelve ``Tensor [B, N, d]``.
- ``"score"``: recibe ``scores: list[Tensor [B, N, N]]`` (puntajes de cada rama
  *antes* del softmax, ver ``scores(q, k)`` en ``ham.models.attention``) y ``v``, y
  devuelve ``(out [B, N, d], weights [B, N, N])`` con la atención fusionada. Sólo
  admite ramas por pares (no ``additive`` tipo pooling).

Mezclar scores antes del softmax equivale a *multiplicar* las distribuciones
(``softmax(λ s_1 + (1-λ) s_2) ∝ p_1^λ p_2^(1-λ)``): la rama más marcada decide y
la otra puede quedar uniforme. Las mezclas de salidas (``_MixFusion``:
``weighted_output``, ``sum_output``, ``gated_output``) promedian distribuciones
(``λ A_1 + (1-λ) A_2``).

En ambos casos el orden de las listas es el de ``AttentionConfig.branches``.

Para añadir una fusión:

    @register_fusion("nombre")
    class MiFusion(nn.Module):
        level = "output"  # o "score"
        def __init__(self, head_dim: int, num_branches: int, dropout: float = 0.0, **kwargs):
            ...

y crear un YAML con ``model.attention.fusion: nombre`` (los hiperparámetros
propios van en ``model.attention.fusion_kwargs``).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

FUSION_REGISTRY: dict[str, type[nn.Module]] = {}


def register_fusion(name: str):
    def deco(cls):
        if name in FUSION_REGISTRY:
            raise KeyError(f"Fusión '{name}' ya registrada")
        FUSION_REGISTRY[name] = cls
        return cls

    return deco


def build_fusion(name: str, head_dim: int, num_branches: int, dropout: float = 0.0, **kwargs) -> nn.Module:
    try:
        cls = FUSION_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Fusión '{name}' no soportada. Disponibles: {sorted(FUSION_REGISTRY)}"
        ) from None
    return cls(head_dim, num_branches, dropout=dropout, **kwargs)


@register_fusion("concat_linear")
class ConcatLinearFusion(nn.Linear):
    """``Linear(concat(out_1, ..., out_k))`` — la fusión de ``202506_Transformers``.

    Hereda de ``nn.Linear`` para conservar las claves ``fusion.weight`` /
    ``fusion.bias`` de los checkpoints anteriores.
    """

    level = "output"

    def __init__(self, head_dim: int, num_branches: int, dropout: float = 0.0):
        super().__init__(head_dim * num_branches, head_dim)

    def forward(self, outputs):
        return super().forward(torch.cat(outputs, dim=-1))


def _init_lambda(module: nn.Module, lambda_: float, learnable: bool):
    """λ fijo (buffer, fuera del checkpoint) o aprendido como ``sigmoid(lambda_logit)``."""
    if learnable:
        if not 0.0 < lambda_ < 1.0:
            raise ValueError(f"Con learnable=True, lambda_ debe estar en (0, 1) (es {lambda_})")
        module.lambda_logit = nn.Parameter(torch.tensor(math.log(lambda_ / (1.0 - lambda_))))
    else:
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError(f"lambda_ debe estar en [0, 1] (es {lambda_})")
        module.register_buffer("lambda_fixed", torch.tensor(float(lambda_)), persistent=False)
    module.learnable = learnable


def _lambda_value(module: nn.Module) -> torch.Tensor:
    return torch.sigmoid(module.lambda_logit) if module.learnable else module.lambda_fixed


class _ScoreFusion(nn.Module):
    """Base de las fusiones a nivel de scores: ``softmax(combine(scores)) @ v``."""

    level = "score"

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def combine(self, scores):
        raise NotImplementedError

    def forward(self, scores, v):
        weights = torch.softmax(self.combine(scores), dim=-1)
        return self.dropout(weights) @ v, weights


@register_fusion("weighted_score")
class WeightedScoreFusion(_ScoreFusion):
    """Fusión ponderada de scores entre dos ramas (notebooks 009/011 de ``20250401_ArticuloJCR``).

    ``A = softmax(λ·s_1 + (1-λ)·s_2)``. Con ``branches: [dot, additive_free]``,
    λ=1 es atención dot-product pura y λ=0 es Bahdanau pura.

    fusion_kwargs:
        lambda_: valor de λ (fijo) o su valor inicial (si ``learnable``). Default 0.85.
        learnable: si es ``True`` λ se aprende por cabeza, parametrizado como
            ``sigmoid(lambda_logit)`` para mantenerlo en (0, 1).
    """

    def __init__(self, head_dim: int, num_branches: int, dropout: float = 0.0,
                 lambda_: float = 0.85, learnable: bool = False):
        super().__init__(dropout)
        if num_branches != 2:
            raise ValueError(f"weighted_score requiere exactamente 2 ramas (hay {num_branches})")
        _init_lambda(self, lambda_, learnable)

    lambda_value = property(lambda self: _lambda_value(self))

    def combine(self, scores):
        lam = self.lambda_value
        return lam * scores[0] + (1.0 - lam) * scores[1]


@register_fusion("sum_score")
class SumScoreFusion(_ScoreFusion):
    """Suma directa de scores: ``A = softmax(s_1 + ... + s_k)`` (notebook 009, "Suma")."""

    def __init__(self, head_dim: int, num_branches: int, dropout: float = 0.0):
        super().__init__(dropout)

    def combine(self, scores):
        return torch.stack(scores).sum(0)


@register_fusion("zscore_score")
class ZScoreFusion(_ScoreFusion):
    """Scores estandarizados por fila antes de mezclarlos.

    ``A = softmax(λ·a_1·z(s_1) + (1-λ)·a_2·z(s_2))`` con ``z(s) = (s - media_j) / desv_j``
    (por fila i, sobre las claves j) y ``a_b = exp(log_scale_b)`` una escala aprendida
    por rama y cabeza (inicial ``scale``). Así ninguna rama domina sólo por su rango
    (``dot`` sin techo contra ``Σ tanh`` acotado a ±d). Sigue siendo una mezcla en
    logits (producto de distribuciones); ver ``weighted_output`` para la mezcla de
    probabilidades.

    fusion_kwargs:
        lambda_, learnable: como en ``weighted_score`` (default 0.5, fijo).
        scale: escala inicial de cada rama estandarizada (default 1.0).
    """

    def __init__(self, head_dim: int, num_branches: int, dropout: float = 0.0,
                 lambda_: float = 0.5, learnable: bool = False, scale: float = 1.0):
        super().__init__(dropout)
        if num_branches != 2:
            raise ValueError(f"zscore_score requiere exactamente 2 ramas (hay {num_branches})")
        if scale <= 0:
            raise ValueError(f"scale debe ser > 0 (es {scale})")
        _init_lambda(self, lambda_, learnable)
        self.log_scale = nn.Parameter(torch.full((num_branches,), math.log(scale)))

    lambda_value = property(lambda self: _lambda_value(self))

    @staticmethod
    def standardize(s, eps: float = 1e-5):
        return (s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + eps)

    def combine(self, scores):
        a = self.log_scale.exp()
        lam = self.lambda_value
        return lam * a[0] * self.standardize(scores[0]) + (1.0 - lam) * a[1] * self.standardize(scores[1])


# --------------------------------------------------------------------------- mezclas de salidas

class _MixFusion(nn.Module):
    """Base de las mezclas de salidas: ``out = Σ_b c_b · out_b`` con ``Σ_b c_b = 1`` (convexas;
    ``sum_output`` usa ``c_b = 1``).

    Como cada ``out_b = A_b · v``, la mezcla equivale a aplicar ``Σ_b c_b · A_b`` a ``v``:
    una mezcla de *probabilidades* (después del softmax de cada rama), no de scores.
    Ninguna rama puede anular a otra por la escala de sus scores y ``c_b`` sí es la
    proporción de la mezcla. ``heads.AttentionHead`` reporta esa atención como ``fused``.

    ``branch_dropout`` (sólo en entrenamiento): con esa probabilidad, por muestra, se
    usa una sola rama elegida al azar (``c`` one-hot), para que cada rama tenga que ser
    útil por sí misma y no se coadapten.
    """

    level = "output"

    def __init__(self, num_branches: int, branch_dropout: float = 0.0):
        super().__init__()
        if not 0.0 <= branch_dropout < 1.0:
            raise ValueError(f"branch_dropout debe estar en [0, 1) (es {branch_dropout})")
        self.num_branches = num_branches
        self.branch_dropout = branch_dropout

    def coefficients(self, outputs) -> list[torch.Tensor]:
        """Peso de cada rama, ``[B, N, 1]`` o escalar; suman 1."""
        raise NotImplementedError

    def mixing(self, outputs) -> list[torch.Tensor]:
        coeffs = self.coefficients(outputs)
        if not (self.training and self.branch_dropout > 0):
            return coeffs
        B, dev = outputs[0].size(0), outputs[0].device
        drop = (torch.rand(B, device=dev) < self.branch_dropout).view(B, 1, 1)
        keep = torch.randint(self.num_branches, (B,), device=dev).view(B, 1, 1)
        return [torch.where(drop, (keep == b).to(outputs[0].dtype), c) for b, c in enumerate(coeffs)]

    def branch_terms(self, outputs, coeffs=None) -> list[torch.Tensor]:
        """Aporte de cada rama a la salida (``c_b · out_b``); su suma es la salida."""
        coeffs = self.mixing(outputs) if coeffs is None else coeffs
        return [c * o for c, o in zip(coeffs, outputs)]

    def forward(self, outputs, coeffs=None):
        return torch.stack(self.branch_terms(outputs, coeffs)).sum(0)


@register_fusion("weighted_output")
class WeightedOutputFusion(_MixFusion):
    """Suma ponderada de salidas: ``out = λ·A_1 v + (1-λ)·A_2 v = (λ·A_1 + (1-λ)·A_2) v``.

    Admite cualquier rama, también ``additive`` (pooling). λ aprendido por cabeza.

    fusion_kwargs:
        lambda_, learnable: como en ``weighted_score`` (default 0.5, fijo).
        branch_dropout: ver ``_MixFusion`` (default 0).
    """

    def __init__(self, head_dim: int, num_branches: int, dropout: float = 0.0,
                 lambda_: float = 0.5, learnable: bool = False, branch_dropout: float = 0.0):
        super().__init__(num_branches, branch_dropout)
        if num_branches != 2:
            raise ValueError(f"weighted_output requiere exactamente 2 ramas (hay {num_branches})")
        _init_lambda(self, lambda_, learnable)

    lambda_value = property(lambda self: _lambda_value(self))

    def coefficients(self, outputs):
        lam = self.lambda_value
        return [lam, 1.0 - lam]


@register_fusion("sum_output")
class SumOutputFusion(_MixFusion):
    """Suma de salidas: ``out = A_1 v + ... + A_k v`` (espejo de ``sum_score`` después del softmax).

    Es la única mezcla con ``Σ c_b = k`` en lugar de 1: con dos ramas equivale a
    ``2 × weighted_output`` con λ=0.5 (misma atención, salida al doble). ``fused`` se
    reporta normalizado (``Σ A_b / k``). Sin ``branch_dropout``: el one-hot cambiaría la
    escala de la salida entre entrenamiento y evaluación.
    """

    def __init__(self, head_dim: int, num_branches: int, dropout: float = 0.0, branch_dropout: float = 0.0):
        super().__init__(num_branches)
        if num_branches < 2:
            raise ValueError(f"sum_output requiere al menos 2 ramas (hay {num_branches})")
        if branch_dropout:
            raise ValueError("sum_output no admite branch_dropout (cambiaría la escala de la salida)")

    def coefficients(self, outputs):
        return [1.0] * self.num_branches


@register_fusion("gated_output")
class GatedOutputFusion(_MixFusion):
    """Mezcla de salidas con compuerta por token: ``out_i = Σ_b g_ib · out_ib``.

    ``g_i = softmax(W [out_i1 ‖ ... ‖ out_ik] + c)``: cada token decide cuánto usar de
    cada rama (con 2 ramas es un sigmoide). Se inicializa en ``W = 0``, ``c = 0``, es
    decir, mezcla pareja.

    fusion_kwargs:
        branch_dropout: ver ``_MixFusion`` (default 0).
    """

    def __init__(self, head_dim: int, num_branches: int, dropout: float = 0.0, branch_dropout: float = 0.0):
        super().__init__(num_branches, branch_dropout)
        self.gate = nn.Linear(head_dim * num_branches, num_branches)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def coefficients(self, outputs):
        g = torch.softmax(self.gate(torch.cat(outputs, dim=-1)), dim=-1)  # [B, N, k]
        return list(g.unsqueeze(-1).unbind(-2))  # k x [B, N, 1]
