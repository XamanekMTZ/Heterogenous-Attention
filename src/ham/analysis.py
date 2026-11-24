"""Métricas de análisis post-entrenamiento: clasificación, atención, fidelidad, robustez.

Funciones puras sobre tensores (las usa ``ham.analyze`` y se prueban en
``tests/test_analysis.py``). Convenciones:

- Pesos de atención de una rama: ``[B, H, N, N]`` (filas = consultas, suman 1) o
  ``[B, H, N]`` en la aditiva tipo pooling; :func:`normalize_attention` lleva el
  segundo caso al primero repitiendo la fila, que es lo que esa rama calcula (el
  mismo contexto para cada token).
- El token 0 es el CLS; los parches forman una rejilla cuadrada.
- La divergencia de Jensen-Shannon se reporta normalizada a [0, 1] (base 2).
"""
from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

EPS = 1e-12


# --------------------------------------------------------------------------- atención

def normalize_attention(w: torch.Tensor) -> torch.Tensor:
    """``[B,H,N]`` (pooling) -> ``[B,H,N,N]`` con filas idénticas; ``[B,H,N,N]`` sin cambios."""
    if w.dim() == 3:
        return w.unsqueeze(2).expand(-1, -1, w.size(-1), -1)
    if w.dim() != 4:
        raise ValueError(f"Pesos de atención con forma inesperada {tuple(w.shape)}")
    return w


def layer_dicts(attentions, branches) -> list[dict[str, torch.Tensor]]:
    """Salida de ``model(x, output_attentions=True)`` -> ``[{rama: [B,H,N,N]}]`` por capa.

    En modelos de una rama la clave es el nombre de esa rama (``branches[0]``).
    """
    out = []
    for layer in attentions:
        items = layer.items() if isinstance(layer, dict) else [(branches[0], layer)]
        out.append({name: normalize_attention(w) for name, w in items})
    return out


def effective_attention(layer: dict[str, torch.Tensor]) -> torch.Tensor:
    """Atención aplicada a ``v`` en una capa, promediada sobre cabezas: ``[B,N,N]``.

    ``fused`` si la fusión es de scores; si no, la media de las ramas. Para
    ``concat_linear`` es una aproximación: esa fusión mezcla salidas, no pesos.
    """
    if "fused" in layer:
        return layer["fused"].mean(1)
    return torch.stack([w.mean(1) for w in layer.values()]).mean(0)


def entropy(w: torch.Tensor) -> torch.Tensor:
    """Entropía de cada fila normalizada por ``log(N_claves)`` (1 = uniforme). ``[..., N]`` -> ``[...]``."""
    n = w.size(-1)
    return -(w * (w + EPS).log()).sum(-1) / math.log(n)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon por fila, en [0, 1]."""
    m = 0.5 * (p + q)
    kl = lambda a: (a * ((a + EPS) / (m + EPS)).log()).sum(-1)  # noqa: E731
    return (0.5 * kl(p) + 0.5 * kl(q)) / math.log(2)


def patch_distances(num_patches: int, patch_size: int) -> torch.Tensor:
    """Distancia euclidiana en píxeles entre centros de parches: ``[P, P]``."""
    g = int(round(num_patches ** 0.5))
    if g * g != num_patches:
        raise ValueError(f"{num_patches} parches no forman una rejilla cuadrada")
    ys, xs = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    coords = torch.stack([ys.flatten(), xs.flatten()], 1).float()
    return torch.cdist(coords, coords) * patch_size


def attention_distance(w: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Distancia media de atención (Dosovitskiy et al. 2021) en píxeles, por cabeza.

    Sólo parches -> parches (se excluye el CLS y se renormaliza cada fila).
    ``[B,H,N,N]`` -> ``[B,H]``.
    """
    a = w[..., 1:, 1:]
    a = a / (a.sum(-1, keepdim=True) + EPS)
    d = patch_distances(a.size(-1), patch_size).to(a)
    return (a * d).sum(-1).mean(-1)


def head_diversity(w: torch.Tensor) -> torch.Tensor:
    """JS media entre pares de cabezas (todas las filas). ``[B,H,N,N]`` -> ``[B]``."""
    h = w.size(1)
    if h < 2:
        return torch.zeros(w.size(0), device=w.device)
    pairs = [js_divergence(w[:, i], w[:, j]).mean(-1) for i, j in itertools.combinations(range(h), 2)]
    return torch.stack(pairs).mean(0)


def rollout(layers: list[torch.Tensor]) -> torch.Tensor:
    """Attention rollout (Abnar y Zuidema 2020) con conexión residual 0.5/0.5.

    ``layers``: matrices ``[B,N,N]`` por capa (ya promediadas sobre cabezas). Devuelve ``[B,N,N]``.
    """
    result = None
    for a in layers:
        eye = torch.eye(a.size(-1), device=a.device, dtype=a.dtype)
        a = 0.5 * a + 0.5 * eye
        a = a / a.sum(-1, keepdim=True)
        result = a if result is None else a @ result
    return result


class AttentionStats:
    """Acumula métricas de atención por capa y rama sobre varios lotes."""

    def __init__(self, patch_size: int):
        self.patch_size = patch_size
        self.n = 0
        self.sums: dict[tuple, list[float]] = {}

    def _add(self, key, layer_idx, num_layers, value: float, batch: int):
        acc = self.sums.setdefault(key, [0.0] * num_layers)
        acc[layer_idx] = acc[layer_idx] + value * batch

    def update(self, layers: list[dict[str, torch.Tensor]]):
        b, nl = next(iter(layers[0].values())).size(0), len(layers)
        for li, layer in enumerate(layers):
            for name, w in layer.items():
                self._add(("entropy_cls", name), li, nl, entropy(w[:, :, 0, 1:] / w[:, :, 0, 1:].sum(-1, keepdim=True)).mean().item(), b)
                self._add(("entropy_all", name), li, nl, entropy(w).mean().item(), b)
                dist = attention_distance(w, self.patch_size)  # [B, H]
                self._add(("distance_px", name), li, nl, dist.mean().item(), b)
                # por cabeza: la más local es la medida de Santiago Toledo y Martínez (2026)
                self._add(("distance_px_heads", name), li, nl, dist.mean(0).detach().cpu().numpy(), b)
                self._add(("head_js", name), li, nl, head_diversity(w).mean().item(), b)
            for a, c in itertools.combinations(layer, 2):
                p, q = layer[a], layer[c]
                if p.shape != q.shape:  # cabezas heterogéneas: cada rama vive en otras cabezas
                    p, q = p.mean(1, keepdim=True), q.mean(1, keepdim=True)
                self._add(("js_between", f"{a}|{c}"), li, nl, js_divergence(p, q).mean().item(), b)
        self.n += b

    def result(self) -> dict:
        out: dict = {"branches": {}, "js_between": {}}
        for (metric, name), values in self.sums.items():
            values = [(v / self.n).tolist() if hasattr(v, "tolist") else v / self.n for v in values]
            if metric == "js_between":
                out["js_between"][name] = values
            else:
                out["branches"].setdefault(name, {})[metric] = values
        return out


@torch.no_grad()
def branch_contributions(model, images: torch.Tensor) -> dict[str, list[float]] | None:
    """Peso relativo de cada rama en fusiones de salida, por capa.

    Para cada cabeza, ``c_b = ‖t_b‖`` (norma por token, media), con ``t_b`` el aporte de
    la rama ``b`` a la salida: ``W_b · out_b`` en ``concat_linear`` (``W_b`` es el bloque
    de la fusión que corresponde a la rama) y ``c_b · out_b`` en las mezclas
    (``weighted_output``, ``gated_output``). Se normaliza para sumar 1 y se promedia
    sobre cabezas. ``None`` si el modelo no tiene fusión de salida en todas las cabezas
    o éstas no tienen las mismas ramas.
    """
    layers = model.encoder.layers
    heads0 = layers[0].attention.attn_heads
    head0 = heads0[0]
    if any(h.fusion is None or h.fusion.level != "output" or h.names != head0.names for h in heads0):
        return None
    captured = {}
    hooks = [
        layer.attention.register_forward_pre_hook(lambda m, args, i=i: captured.__setitem__(i, args[0]))
        for i, layer in enumerate(layers)
    ]
    try:
        model(images)
    finally:
        for h in hooks:
            h.remove()

    out = {name: [] for name in head0.names}
    for i, layer in enumerate(layers):
        mha = layer.attention
        x = captured[i]
        B, N, _ = x.shape
        qkv = mha.qkv(x).reshape(B, N, 3, mha.num_heads, mha.head_dim).permute(2, 0, 3, 1, 4)
        per_head = []
        for h, head in enumerate(mha.attn_heads):
            q, k, v = qkv[0][:, h], qkv[1][:, h], qkv[2][:, h]
            d = mha.head_dim
            outs = [getattr(head, f"attn{b + 1}")(q, k, v)[0] for b in range(len(head.branches))]
            if hasattr(head.fusion, "branch_terms"):
                terms = head.fusion.branch_terms(outs)
            else:
                terms = [o @ head.fusion.weight[:, b * d:(b + 1) * d].T for b, o in enumerate(outs)]
            norms = torch.stack([t.norm(dim=-1).mean() for t in terms])
            per_head.append(norms / norms.sum())
        mean = torch.stack(per_head).mean(0)
        for name, val in zip(head0.names, mean.tolist()):
            out[name].append(val)
    return out


def _capture_attention_inputs(model, images) -> dict[int, torch.Tensor]:
    """Entrada (ya normalizada) de la atención de cada capa."""
    captured = {}
    hooks = [
        layer.attention.register_forward_pre_hook(lambda m, args, i=i: captured.__setitem__(i, args[0]))
        for i, layer in enumerate(model.encoder.layers)
    ]
    try:
        model(images)
    finally:
        for h in hooks:
            h.remove()
    return captured


@torch.no_grad()
def enrich_contributions(model, images: torch.Tensor) -> dict | None:
    """Peso de los enriquecimientos (``ham.models.enrich``) por capa.

    ``terms``: norma media por token de la salida de las cabezas (``attention``, ``A·v``
    tras la fusión) y de cada término aditivo (``local``, ``global``), normalizadas para
    sumar 1 y promediadas sobre cabezas. ``gate``: media, desviación y fracción < 0.5 de
    la compuerta (vale 1 al inicio). ``None`` si el modelo no tiene enriquecimientos.
    """
    layers = model.encoder.layers
    enrich_names = list(model.config.attention.enrich)
    if not enrich_names:
        return None
    captured = _capture_attention_inputs(model, images)
    terms = {"attention": [], **{n: [] for n, e in zip(enrich_names, layers[0].attention.enrich)
                                 if e.kind == "additive"}}
    gate = {"mean": [], "std": [], "frac_below_0.5": []}
    for i, layer in enumerate(layers):
        mha = layer.attention
        x = captured[i]
        B, N, _ = x.shape
        q, k, v = mha.qkv(x).reshape(B, N, 3, mha.num_heads, mha.head_dim).permute(2, 0, 3, 1, 4)
        base = torch.stack([head(q[:, h], k[:, h], v[:, h])[0] for h, head in enumerate(mha.attn_heads)], 1)
        norms = {"attention": base.norm(dim=-1).mean((0, 2))}  # [H]
        for name, e in zip(enrich_names, mha.enrich):
            if e.kind == "additive":
                norms[name] = e.delta(x, q, k, v).norm(dim=-1).mean((0, 2))
            else:
                g = e.values(x)
                gate["mean"].append(g.mean().item())
                gate["std"].append(g.std().item())
                gate["frac_below_0.5"].append((g < 0.5).float().mean().item())
        total = torch.stack(list(norms.values())).sum(0)
        for name, n in norms.items():
            terms[name].append((n / total).mean().item())
    out = {"enrich": enrich_names, "terms": terms}
    if gate["mean"]:
        out["gate"] = gate
    return out


# --------------------------------------------------------------------------- clasificación

def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int, bins: int = 15) -> dict:
    """Accuracy, top-5, NLL, precision/recall/F1 macro, accuracy por clase, confusión y ECE."""
    logits, labels = logits.float(), labels.long()
    probs = logits.softmax(-1)
    conf, pred = probs.max(-1)
    correct = (pred == labels).float()
    c = num_classes
    confusion = torch.bincount(labels * c + pred, minlength=c * c).view(c, c)
    tp = confusion.diag().float()
    support, predicted = confusion.sum(1).float(), confusion.sum(0).float()
    precision = torch.where(predicted > 0, tp / predicted.clamp(min=1), torch.zeros_like(tp))
    recall = torch.where(support > 0, tp / support.clamp(min=1), torch.zeros_like(tp))
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall + EPS), torch.zeros_like(tp))
    e, reliability = expected_calibration_error(conf, correct, bins)
    out = {
        "acc": 100.0 * correct.mean().item(),
        "nll": F.cross_entropy(logits, labels).item(),
        "precision_macro": 100.0 * precision.mean().item(),
        "recall_macro": 100.0 * recall.mean().item(),
        "f1_macro": 100.0 * f1.mean().item(),
        "ece": 100.0 * e,
        "per_class_acc": (100.0 * recall).tolist(),
        "confusion": confusion.tolist(),
        "reliability": reliability,
    }
    if c >= 20:
        top5 = logits.topk(5, -1).indices.eq(labels[:, None]).any(-1).float()
        out["top5"] = 100.0 * top5.mean().item()
    return out


def expected_calibration_error(conf: torch.Tensor, correct: torch.Tensor, bins: int = 15):
    """ECE con bins de igual ancho. Devuelve ``(ece, {conf, acc, count} por bin)``."""
    idx = (conf * bins).ceil().long().clamp(1, bins) - 1
    count = torch.bincount(idx, minlength=bins).float()
    conf_sum = torch.bincount(idx, weights=conf, minlength=bins)
    acc_sum = torch.bincount(idx, weights=correct, minlength=bins)
    nz = count > 0
    mean_conf = torch.where(nz, conf_sum / count.clamp(min=1), torch.zeros_like(conf_sum))
    mean_acc = torch.where(nz, acc_sum / count.clamp(min=1), torch.zeros_like(acc_sum))
    ece = ((mean_acc - mean_conf).abs() * count).sum().item() / max(1, conf.numel())
    return ece, {"conf": mean_conf.tolist(), "acc": mean_acc.tolist(), "count": count.long().tolist()}


def mcnemar(correct_a, correct_b) -> dict:
    """Prueba exacta de McNemar (binomial, dos colas) entre dos clasificadores pareados.

    ``n10``: aciertos de A que B falla; ``n01``: al revés.
    """
    from scipy.stats import binomtest

    a = torch.as_tensor(correct_a).bool()
    b = torch.as_tensor(correct_b).bool()
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = 1.0 if n10 + n01 == 0 else binomtest(n10, n10 + n01, 0.5).pvalue
    return {"n10": n10, "n01": n01, "p_value": float(p)}


def bootstrap_diff(correct_a, correct_b, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Diferencia de accuracy ``A − B`` (puntos %) con IC por bootstrap pareado sobre imágenes."""
    a = torch.as_tensor(correct_a).float()
    b = torch.as_tensor(correct_b).float()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, a.numel(), (n_boot, a.numel()), generator=g)
    diffs = 100.0 * (a[idx].mean(1) - b[idx].mean(1))
    lo, hi = torch.quantile(diffs, torch.tensor([alpha / 2, 1 - alpha / 2])).tolist()
    return {"diff": 100.0 * (a.mean() - b.mean()).item(), "ci_low": lo, "ci_high": hi}


# --------------------------------------------------------------------------- perturbaciones

def patch_mask_to_pixels(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """``[B, P]`` booleana (True = borrar) -> ``[B, 1, H, W]``."""
    g = int(round(mask.size(1) ** 0.5))
    m = mask.view(-1, 1, g, g).float()
    return F.interpolate(m, scale_factor=patch_size, mode="nearest")


def delete_patches(x: torch.Tensor, mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Pone a 0 los parches marcados. En el espacio normalizado 0 es la media del dataset."""
    return x * (1.0 - patch_mask_to_pixels(mask, patch_size).to(x))


def random_patch_mask(batch: int, num_patches: int, frac: float, generator: torch.Generator) -> torch.Tensor:
    k = int(round(frac * num_patches))
    order = torch.rand(batch, num_patches, generator=generator).argsort(1)
    mask = torch.zeros(batch, num_patches, dtype=torch.bool)
    mask.scatter_(1, order[:, :k], True)
    return mask


def gaussian_noise(x_pix: torch.Tensor, sigma: float, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(x_pix.shape, generator=generator).to(x_pix)
    return (x_pix + sigma * noise).clamp(0, 1)


def gaussian_blur(x_pix: torch.Tensor, sigma: float) -> torch.Tensor:
    k = 2 * math.ceil(3 * sigma) + 1
    return TF.gaussian_blur(x_pix, kernel_size=[k, k], sigma=[sigma, sigma])


ROBUSTNESS = {
    "noise": (0.05, 0.1, 0.2),     # σ del ruido gaussiano (imagen en [0, 1])
    "blur": (0.5, 1.0, 1.5),       # σ del desenfoque gaussiano, en píxeles
    "occlusion": (0.1, 0.25, 0.5),  # fracción de parches borrados al azar
}

DELETION_FRACTIONS = [i / 16 for i in range(17)]
