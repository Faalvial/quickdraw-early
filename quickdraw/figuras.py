"""Figuras del paper. Cada función guarda un PNG y devuelve la figura.
Ver docs/05_metricas.md y docs/07_analisis_h4.md."""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .config import CATEGORIAS, FRACCIONES, NOMBRES_ES, PARES
from .raster import caja, rasterizar

matplotlib.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200,
                            "axes.spines.top": False, "axes.spines.right": False})
COLORES = {"M1": "#8c8c8c", "M2": "#1f77b4", "M2-P": "#d62728"}


def _guardar(fig, ruta):
    if ruta:
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, bbox_inches="tight")
    return fig


def prefijos(dibujos, indices, cfg, ruta=None, max_trazos=8):
    """Fila superior de cada dibujo: normalización causal. Fila inferior:
    encuadre con la caja del dibujo completo (solo como referencia)."""
    fig, ejes = plt.subplots(2 * len(indices), max_trazos,
                             figsize=(1.1 * max_trazos, 2.3 * len(indices)))
    for r, i in enumerate(indices):
        trazos = dibujos[i]["trazos"]
        ref = caja(trazos)
        for k in range(max_trazos):
            for fila, caja_ref in ((2 * r, None), (2 * r + 1, ref)):
                ax = ejes[fila, k]
                ax.axis("off")
                if k < len(trazos):
                    ax.imshow(rasterizar(trazos[:k + 1], cfg, caja_ref), cmap="gray_r")
                    if fila == 2 * r:
                        ax.set_title(f"k={k + 1}", fontsize=8)
        ejes[2 * r, 0].text(-6, 14, "causal", rotation=90, va="center", fontsize=8)
        ejes[2 * r + 1, 0].text(-6, 14, "caja final", rotation=90, va="center", fontsize=8)
    fig.tight_layout()
    return _guardar(fig, ruta)


def curvas(resultados, ruta=None, titulo="Exactitud por fracción del dibujo"):
    """resultados: {modelo: matriz (semillas, 10) de exactitud por fracción}."""
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    f = np.array(FRACCIONES) * 100
    for nombre, acc in resultados.items():
        acc = np.atleast_2d(acc) * 100
        media, de = acc.mean(0), acc.std(0)
        ax.plot(f, media, marker="o", ms=3, label=nombre, color=COLORES.get(nombre))
        if len(acc) > 1:
            ax.fill_between(f, media - de, media + de, alpha=0.2, color=COLORES.get(nombre))
    ax.set(xlabel="Porcentaje de trazos (%)", ylabel="Exactitud (%)", title=titulo,
           xticks=f, ylim=(0, 100))
    ax.legend(frameon=False)
    fig.tight_layout()
    return _guardar(fig, ruta)


def por_k(resultados, ruta=None):
    """resultados: {modelo: lista de filas de metricas.exactitud_por_k}."""
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for nombre, filas in resultados.items():
        ax.plot([r["k"] for r in filas], [100 * r["exactitud"] for r in filas],
                marker="o", ms=3, label=nombre, color=COLORES.get(nombre))
    ax.set(xlabel="Número de trazos dibujados (k)", ylabel="Exactitud (%)",
           title="Exactitud por número de trazos", ylim=(0, 100))
    ax.legend(frameon=False)
    fig.tight_layout()
    return _guardar(fig, ruta)


def tsne(mapas, ruta=None):
    """mapas: {titulo: (xy, etiquetas)}. Colorea los 5 pares; el resto en gris."""
    fig, ejes = plt.subplots(1, len(mapas), figsize=(5.2 * len(mapas), 4.6))
    ejes = np.atleast_1d(ejes)
    paleta = plt.get_cmap("tab10")
    for ax, (titulo, (xy, y)) in zip(ejes, mapas.items()):
        gris = ~np.isin(y, [CATEGORIAS.index(c) for p in PARES for c in p])
        ax.scatter(xy[gris, 0], xy[gris, 1], s=3, c="#d0d0d0", label="otras 20")
        for j, cat in enumerate(c for p in PARES for c in p):
            m = y == CATEGORIAS.index(cat)
            ax.scatter(xy[m, 0], xy[m, 1], s=4, color=paleta(j), label=NOMBRES_ES[cat])
        ax.set(title=titulo, xticks=[], yticks=[])
    ejes[-1].legend(markerscale=3, fontsize=7, frameon=False, bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return _guardar(fig, ruta)


def h4(conf, mediana, rho, p, ruta=None):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(conf, mediana, s=18)
    for c, x, y in zip(CATEGORIAS, conf, mediana):
        ax.annotate(NOMBRES_ES[c], (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set(xlabel="Confundibilidad en el espacio de M2 (1 − pureza kNN)",
           ylabel="Mediana de k*/K de M2-P",
           title=f"H4: Spearman ρ = {rho:.2f} (p = {p:.3f})")
    fig.tight_layout()
    return _guardar(fig, ruta)


def h5(K, ks, ruta=None):
    """Distribución de K − k*: cuántos trazos antes del final acertó el modelo."""
    m = ks > 0
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    d = K[m] - ks[m]
    ax.hist(d, bins=np.arange(0, min(d.max(), 25) + 2) - 0.5, color=COLORES["M2-P"])
    ax.set(xlabel="K − k* (trazos de anticipación)", ylabel="Dibujos",
           title=f"H5: {100 * (d > 0).mean():.1f} % de los acertados, antes del último trazo")
    fig.tight_layout()
    return _guardar(fig, ruta)


def confusion(matriz, titulo, ruta=None):
    norm = matriz / np.maximum(matriz.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    nombres = [NOMBRES_ES[c] for c in CATEGORIAS]
    ax.set_xticks(range(30), nombres, rotation=90, fontsize=6)
    ax.set_yticks(range(30), nombres, fontsize=6)
    ax.set(xlabel="Predicha", ylabel="Verdadera", title=titulo)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return _guardar(fig, ruta)


def trazo_a_trazo(cache, preds, posiciones, ruta=None, max_trazos=10):
    """Para algunos dibujos, el prefijo de cada trazo y lo que predice cada modelo.

    preds: {modelo: pred}; posiciones: índices de dibujo dentro de la caché.
    """
    fig, ejes = plt.subplots(len(posiciones), max_trazos,
                             figsize=(1.25 * max_trazos, 1.9 * len(posiciones)))
    ejes = np.atleast_2d(ejes)
    for r, i in enumerate(posiciones):
        ini, K, y = cache["inicio"][i], int(cache["K"][i]), int(cache["etiqueta"][i])
        for k in range(max_trazos):
            ax = ejes[r, k]
            ax.axis("off")
            if k >= K:
                continue
            ax.imshow(cache["imagenes"][ini + k], cmap="gray_r")
            texto = []
            for nombre, pred in preds.items():
                c = int(pred[ini + k])
                marca = "✓" if c == y else "✗"
                texto.append(f"{nombre}: {NOMBRES_ES[CATEGORIAS[c]]} {marca}")
            ax.set_title("\n".join(texto), fontsize=5)
        ejes[r, 0].text(-8, 14, NOMBRES_ES[CATEGORIAS[y]], rotation=90, va="center", fontsize=8)
    fig.tight_layout()
    return _guardar(fig, ruta)
