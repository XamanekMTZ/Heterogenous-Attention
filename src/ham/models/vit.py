"""Vision Transformer (pre-norm) — port de ``202506_Transformers``.

Mantiene los nombres de atributos originales (``patch_embed``, ``encoder.layers``,
``encoder.final_norm``, ``mlp_head``) para cargar checkpoints anteriores.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ham.config import ModelConfig
from ham.models.heads import MultiHeadAttention


class NewGELUActivation(nn.Module):
    """Aproximación tanh de GELU (la usada en el código original)."""

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, dropout=0.1):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"image_size ({img_size}) debe ser divisible por patch_size ({patch_size})")
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def tokens(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, N_patches, D]

    def forward(self, x):
        x = self.tokens(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        return self.dropout(x)


class ShiftedPatchEmbedding(PatchEmbedding):
    """*Shifted Patch Tokenization* (Lee et al., 2021): la imagen y sus cuatro copias desplazadas
    media celda en diagonal se concatenan por canal; cada parche de 5·C·P² valores pasa por
    LayerNorm y una proyección lineal."""

    def __init__(self, img_size, patch_size, in_channels, embed_dim, dropout=0.1):
        super().__init__(img_size, patch_size, in_channels, embed_dim, dropout)
        del self.proj
        self.patch_size = patch_size
        patch_dim = 5 * in_channels * patch_size**2
        self.norm = nn.LayerNorm(patch_dim)
        self.linear = nn.Linear(patch_dim, embed_dim)

    def tokens(self, x):
        s, h, w = self.patch_size // 2, x.size(-2), x.size(-1)
        padded = nn.functional.pad(x, (s, s, s, s))
        shifts = [padded[..., s + dy:s + dy + h, s + dx:s + dx + w] for dx, dy in ((-s, -s), (s, -s), (-s, s), (s, s))]
        x = torch.cat([x, *shifts], dim=1)
        patches = nn.functional.unfold(x, self.patch_size, stride=self.patch_size).transpose(1, 2)
        return self.linear(self.norm(patches))


class ConvStemEmbedding(PatchEmbedding):
    """*Stem* convolucional (Xiao et al., 2021, *Early convolutions help transformers see better*):
    log2(P) convoluciones 3×3 de paso 2 con BatchNorm y ReLU, que duplican canales hasta D,
    y una convolución 1×1 a D en lugar de la convolución de paso P."""

    def __init__(self, img_size, patch_size, in_channels, embed_dim, dropout=0.1):
        super().__init__(img_size, patch_size, in_channels, embed_dim, dropout)
        del self.proj
        steps = int(math.log2(patch_size))
        if 2**steps != patch_size:
            raise ValueError(f"conv_stem requiere patch_size potencia de 2 (es {patch_size})")
        layers, c_in = [], in_channels
        for i in range(steps):
            c_out = embed_dim // 2 ** (steps - 1 - i)
            layers += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU()]
            c_in = c_out
        layers.append(nn.Conv2d(c_in, embed_dim, 1))
        self.stem = nn.Sequential(*layers)

    def tokens(self, x):
        return self.stem(x).flatten(2).transpose(1, 2)


PATCH_EMBEDDINGS = {"patch": PatchEmbedding, "spt": ShiftedPatchEmbedding, "conv_stem": ConvStemEmbedding}


class MLP(nn.Module):
    """``local_kernel > 0``: convolución depthwise k×k (y otra activación) entre ``fc1`` y ``fc2``
    sobre la rejilla de parches, como en LocalViT (Li et al., 2021); el CLS no la recibe."""

    def __init__(self, hidden_size, intermediate_size, dropout=0.0, local_kernel=0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = NewGELUActivation()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        if local_kernel:
            self.local = nn.Conv2d(intermediate_size, intermediate_size, local_kernel,
                                   padding=local_kernel // 2, groups=intermediate_size)

    def forward(self, x):
        h = self.activation(self.fc1(x))
        if hasattr(self, "local"):
            B, N, C = h.shape
            g = math.isqrt(N - 1)
            grid = h[:, 1:].transpose(1, 2).reshape(B, C, g, g)
            patches = self.activation(self.local(grid)).flatten(2).transpose(1, 2)
            h = torch.cat([h[:, :1], patches], dim=1)
        return self.dropout2(self.fc2(self.dropout1(h)))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h = cfg.hidden_size
        self.norm1 = nn.LayerNorm(h, eps=1e-6)
        self.attention = MultiHeadAttention(
            embed_dim=h,
            num_heads=cfg.num_attention_heads,
            branches=cfg.attention.branches,
            fusion=cfg.attention.fusion,
            fusion_kwargs=cfg.attention.fusion_kwargs,
            attn_dropout=cfg.attention_probs_dropout_prob,
            proj_dropout=cfg.hidden_dropout_prob,
            qkv_bias=cfg.qkv_bias,
            head_branches=cfg.attention.head_branches,
            enrich=cfg.attention.enrich,
            enrich_kwargs=cfg.attention.enrich_kwargs,
        )
        # Igual que el original: dropout residual además del dropout de out_proj y del MLP.
        self.dropout1 = nn.Dropout(cfg.hidden_dropout_prob)
        self.norm2 = nn.LayerNorm(h, eps=1e-6)
        self.mlp = MLP(h, cfg.intermediate_size, cfg.hidden_dropout_prob, cfg.mlp_local_kernel)
        self.dropout2 = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, x, output_attentions=False):
        if output_attentions:
            attn_out, attn = self.attention(self.norm1(x), return_attn=True)
        else:
            attn_out = self.attention(self.norm1(x))
        x = x + self.dropout1(attn_out)
        x = x + self.dropout2(self.mlp(self.norm2(x)))
        return (x, attn) if output_attentions else x


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList(TransformerEncoderBlock(cfg) for _ in range(cfg.num_layers))
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=1e-6)

    def forward(self, x, output_attentions=False):
        attentions = []
        for layer in self.layers:
            if output_attentions:
                x, attn = layer(x, output_attentions=True)
                attentions.append(attn)
            else:
                x = layer(x)
        x = self.final_norm(x)
        return (x, attentions) if output_attentions else x


class VisionTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.num_classes is None:
            raise ValueError("model.num_classes no está definido (lo asigna el dataset)")
        self.config = cfg
        if cfg.patch_embed not in PATCH_EMBEDDINGS:
            raise ValueError(f"patch_embed '{cfg.patch_embed}' no soportado: {sorted(PATCH_EMBEDDINGS)}")
        self.patch_embed = PATCH_EMBEDDINGS[cfg.patch_embed](
            cfg.image_size, cfg.patch_size, cfg.num_channels, cfg.hidden_size, cfg.hidden_dropout_prob
        )
        self.encoder = TransformerEncoder(cfg)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size, eps=1e-6),
            nn.Linear(cfg.hidden_size, cfg.num_classes),
        )

    def forward(self, x, output_attentions=False):
        """Devuelve ``logits`` o ``(logits, attentions)``.

        ``attentions`` es una lista por capa; cada elemento es ``[B, H, ...]`` o,
        en HAM, un ``dict`` ``{rama: [B, H, ...]}``.
        """
        x = self.patch_embed(x)
        if output_attentions:
            x, attentions = self.encoder(x, output_attentions=True)
            return self.mlp_head(x[:, 0]), attentions
        return self.mlp_head(self.encoder(x)[:, 0])


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.arch == "resnet":
        from ham.models.cnn import ResNet

        return ResNet(cfg)
    if cfg.arch != "vit":
        raise ValueError(f"arch '{cfg.arch}' no soportada (vit | resnet)")
    return VisionTransformer(cfg)
