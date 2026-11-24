"""CNN de referencia: una ResNet pequeña para CIFAR (He et al., 2016) con un presupuesto de
parámetros comparable al del ViT. Mismo protocolo de entrenamiento que los ViT."""
from __future__ import annotations

import torch.nn as nn

from ham.config import ModelConfig


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class ResNet(nn.Module):
    """``cnn_widths`` canales por etapa, ``cnn_blocks`` bloques por etapa; paso 2 desde la segunda."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.num_classes is None:
            raise ValueError("model.num_classes no está definido (lo asigna el dataset)")
        self.config = cfg
        w = cfg.cnn_widths
        self.stem = nn.Sequential(nn.Conv2d(cfg.num_channels, w[0], 3, padding=1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU())
        blocks, c_in = [], w[0]
        for i, c in enumerate(w):
            for j in range(cfg.cnn_blocks):
                blocks.append(BasicBlock(c_in, c, 2 if i > 0 and j == 0 else 1))
                c_in = c
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(cfg.hidden_dropout_prob),
                                  nn.Linear(c_in, cfg.num_classes))

    def forward(self, x, output_attentions=False):
        if output_attentions:
            raise ValueError("ResNet no tiene atención")
        return self.head(self.blocks(self.stem(x)))
