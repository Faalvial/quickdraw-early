"""Métricas progresivas: exactitud por fracción, top-3, AUC progresiva, k* y
matrices de confusión. Sección 3.7 del avance; ver docs/05_metricas.md.

Todas las funciones reciben una caché (ver raster.construir_cache) y las
predicciones del modelo para TODOS sus prefijos:
  pred  (N_prefijos,)    clase predicha
  top3  (N_prefijos, 3)  las tres clases más probables
"""
import numpy as np

from .config import CATEGORIAS, CATEGORIAS_PARES, FRACCIONES
from .raster import indices_completos, indices_fraccion

ETIQUETAS_PARES = [CATEGORIAS.index(c) for c in CATEGORIAS_PARES]


def mascara_progresiva(cache, min_trazos: int = 3) -> np.ndarray:
    """Dibujos que entran en las curvas progresivas y en H5 (K >= 3)."""
    return cache["K"] >= min_trazos


def mascara_grupo(cache, grupo: str) -> np.ndarray:
    """'pares' (las 10 categorías de los pares), 'azar' (las 20) o 'todas'."""
    es_par = np.isin(cache["etiqueta"], ETIQUETAS_PARES)
    return {"pares": es_par, "azar": ~es_par}.get(grupo, np.ones_like(es_par))


def correctos_por_fraccion(cache, pred, fracciones=FRACCIONES) -> np.ndarray:
    """Matriz booleana (N_dibujos, n_fracciones): ¿acertó con el prefijo de f?"""
    return np.stack([pred[indices_fraccion(cache, f)] == cache["etiqueta"]
                     for f in fracciones], axis=1)


def top3_por_fraccion(cache, top3, fracciones=FRACCIONES) -> np.ndarray:
    return np.stack([(top3[indices_fraccion(cache, f)] == cache["etiqueta"][:, None]).any(1)
                     for f in fracciones], axis=1)


def auc_progresiva(exactitudes, fracciones=FRACCIONES) -> float:
    """(1/0,9) * integral de 0,1 a 1 de acc(f) df, con la regla del trapecio."""
    exactitudes = np.asarray(exactitudes, dtype=float)
    f = np.asarray(fracciones, dtype=float)
    area = np.sum((exactitudes[1:] + exactitudes[:-1]) / 2 * np.diff(f))
    return float(area / (f[-1] - f[0]))


def auc_por_dibujo(correctos, fracciones=FRACCIONES) -> np.ndarray:
    """AUC progresiva de cada dibujo; su promedio es la AUC del conjunto.

    Sirve para el bootstrap: remuestrear dibujos y promediar.
    """
    f = np.asarray(fracciones, dtype=float)
    c = correctos.astype(float)
    return ((c[:, 1:] + c[:, :-1]) / 2 * np.diff(f)).sum(1) / (f[-1] - f[0])


def exactitud_por_k(cache, pred, k_max: int = 10):
    """Exactitud con exactamente k trazos, entre los dibujos que tienen K >= k."""
    filas = []
    for k in range(1, k_max + 1):
        m = cache["K"] >= k
        if not m.any():
            break
        idx = cache["inicio"][m] + k - 1
        filas.append({"k": k, "dibujos": int(m.sum()),
                      "exactitud": float((pred[idx] == cache["etiqueta"][m]).mean())})
    return filas


def k_estrella(cache, pred) -> np.ndarray:
    """k* de cada dibujo: primer trazo desde el cual la predicción es correcta y
    ya no cambia hasta el último trazo. Vale 0 si con el dibujo completo falla.
    """
    ks = np.zeros(len(cache["K"]), dtype=np.int32)
    for i, (ini, K, y) in enumerate(zip(cache["inicio"], cache["K"], cache["etiqueta"])):
        seq = pred[ini:ini + K]
        if seq[-1] != y:
            continue
        incorrectos = np.nonzero(seq != y)[0]
        ks[i] = (incorrectos[-1] + 2) if len(incorrectos) else 1
    return ks


def fraccion_estrella(cache, ks) -> np.ndarray:
    """k*/K de cada dibujo; los que nunca aciertan cuentan como 1 (el peor valor)."""
    return np.where(ks > 0, ks / cache["K"], 1.0)


def matriz_confusion(cache, pred, f: float, n_clases: int = 30) -> np.ndarray:
    """Filas: categoría verdadera; columnas: categoría predicha (conteos)."""
    p = pred[indices_fraccion(cache, f)]
    m = np.zeros((n_clases, n_clases), dtype=np.int64)
    np.add.at(m, (cache["etiqueta"], p), 1)
    return m


def resumen(cache, pred, top3, min_trazos: int = 3):
    """Tabla de métricas principales por grupo de categorías."""
    filas = []
    final = pred[indices_completos(cache)] == cache["etiqueta"]
    final_top3 = (top3[indices_completos(cache)] == cache["etiqueta"][:, None]).any(1)
    prog = mascara_progresiva(cache, min_trazos)
    correctos = correctos_por_fraccion(cache, pred)
    for grupo in ("todas", "azar", "pares"):
        g = mascara_grupo(cache, grupo)
        acc_f = correctos[g & prog].mean(0)
        filas.append({"grupo": grupo,
                      "exactitud_final": float(final[g].mean()),
                      "top3_final": float(final_top3[g].mean()),
                      "auc_progresiva": auc_progresiva(acc_f),
                      **{f"acc_f{int(round(f * 100))}": float(a)
                         for f, a in zip(FRACCIONES, acc_f)}})
    return filas
