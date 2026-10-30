"""Modelos M1 (MLP), M2 y M2-P (la misma CNN). Ver docs/03_modelos.md."""
import torch
from torch import nn

N_CLASES = 30


class MLP(nn.Module):
    """M1: 784 -> 512 -> 256 -> 30, con ReLU y dropout."""

    def __init__(self, dropout: float = 0.3, resolucion: int = 28):
        super().__init__()
        self.red = nn.Sequential(
            nn.Flatten(),
            nn.Linear(resolucion * resolucion, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, N_CLASES))

    def forward(self, x):
        return self.red(x)


def bloque(entrada: int, salida: int) -> nn.Sequential:
    """[Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2]."""
    return nn.Sequential(
        nn.Conv2d(entrada, salida, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(salida), nn.ReLU(), nn.MaxPool2d(2))


class CNN(nn.Module):
    """M2 y M2-P: tres bloques de 32, 64 y 128 filtros -> GAP -> Dense(30).

    Dimensiones: 1x28x28 -> 32x14x14 -> 64x7x7 -> 128x3x3 -> 128 -> 30.
    """

    def __init__(self, dropout: float = 0.0, filtros=(32, 64, 128)):
        super().__init__()
        canales = (1,) + tuple(filtros)
        self.extractor = nn.Sequential(*[bloque(canales[i], canales[i + 1])
                                         for i in range(len(filtros))])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.clasificador = nn.Linear(filtros[-1], N_CLASES)

    def vectores(self, x):
        """El vector de 128 valores que sale del global average pooling (para H4)."""
        return self.gap(self.extractor(x)).flatten(1)

    def forward(self, x):
        return self.clasificador(self.dropout(self.vectores(x)))


def crear_modelo(nombre: str, dropout: float) -> nn.Module:
    if nombre == "M1":
        return MLP(dropout)
    if nombre in ("M2", "M2-P"):
        return CNN(dropout)
    raise ValueError(f"Modelo desconocido: {nombre}")


def preparar_entrada(imagenes_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 (N, 28, 28) -> float (N, 1, 28, 28) en [0, 1]."""
    return imagenes_uint8.unsqueeze(1).float().div_(255.0)


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
