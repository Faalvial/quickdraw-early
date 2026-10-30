"""Análisis por categoría para H4: vectores de la CNN, confundibilidad con
vecinos más cercanos y mapas t-SNE. Ver docs/07_analisis_h4.md."""
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from .config import CATEGORIAS
from .entrenar import predecir
from .metricas import fraccion_estrella
from .raster import indices_completos, indices_fraccion


def vectores_cnn(modelo, cache, f: float = 1.0) -> np.ndarray:
    """Vector de 128 valores (salida del GAP) del prefijo de fracción f de cada dibujo."""
    idx = indices_completos(cache) if f >= 1 else indices_fraccion(cache, f)
    return predecir(modelo, cache["imagenes"][idx], vectores=True)


def confundibilidad(vectores, etiquetas, k: int = 10, n_clases: int = 30) -> np.ndarray:
    """1 - pureza: por categoría, la proporción promedio de los k vecinos más
    cercanos de sus dibujos que son de OTRA categoría."""
    vecinos = NearestNeighbors(n_neighbors=k + 1).fit(vectores)
    _, idx = vecinos.kneighbors(vectores)
    idx = idx[:, 1:]                                    # se quita el propio dibujo
    pureza = (etiquetas[idx] == etiquetas[:, None]).mean(1)
    return np.array([1 - pureza[etiquetas == c].mean() for c in range(n_clases)])


def mediana_fraccion_estrella(cache, ks, n_clases: int = 30) -> np.ndarray:
    """Mediana de k*/K por categoría (los que nunca aciertan cuentan como 1)."""
    fr = fraccion_estrella(cache, ks)
    return np.array([np.median(fr[cache["etiqueta"] == c]) for c in range(n_clases)])


def mapa_tsne(vectores, etiquetas, por_categoria: int = 150, perplejidad: int = 30,
              semilla: int = 0):
    """Proyección t-SNE en 2D de una submuestra equilibrada (solo para mirar)."""
    rng = np.random.default_rng(semilla)
    elegidos = np.concatenate([
        rng.choice(np.nonzero(etiquetas == c)[0],
                   size=min(por_categoria, int((etiquetas == c).sum())), replace=False)
        for c in np.unique(etiquetas)])
    perp = min(perplejidad, max(5, (len(elegidos) - 1) // 3))
    xy = TSNE(n_components=2, perplexity=perp, init="pca", random_state=semilla,
              learning_rate="auto").fit_transform(vectores[elegidos])
    return xy, etiquetas[elegidos]


def separacion_de_pares(cache, pred, pares, fracciones):
    """Para cada par, qué porcentaje de sus dibujos se confunde con la otra
    categoría del par en cada fracción. Muestra en qué momento se separan."""
    filas = []
    for a, b in pares:
        ia, ib = CATEGORIAS.index(a), CATEGORIAS.index(b)
        m = np.isin(cache["etiqueta"], [ia, ib])
        y = cache["etiqueta"][m]
        otra = np.where(y == ia, ib, ia)
        for f in fracciones:
            p = pred[indices_fraccion(cache, f)][m]
            filas.append({"par": f"{a}/{b}", "fraccion": f,
                          "confusion_con_su_par": float((p == otra).mean())})
    return filas
