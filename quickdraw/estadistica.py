"""Bootstrap, McNemar y Spearman. Sección 3.6 del avance; ver
docs/06_estadistica_e_hipotesis.md."""
import numpy as np
from scipy import stats


def bootstrap(valores_por_dibujo, n: int = 1000, semilla: int = 2026, nivel: float = 0.95):
    """Intervalo de confianza del promedio, remuestreando dibujos con reemplazo.

    valores_por_dibujo: arreglo de una o más columnas (por ejemplo, la diferencia
    de AUC entre dos modelos en cada dibujo). Devuelve (promedio, inferior, superior).
    """
    v = np.asarray(valores_por_dibujo, dtype=float)
    rng = np.random.default_rng(semilla)
    medias = np.array([v[rng.integers(0, len(v), len(v))].mean(0) for _ in range(n)])
    a = (1 - nivel) / 2
    return float(v.mean()), float(np.quantile(medias, a)), float(np.quantile(medias, 1 - a))


def mcnemar(correctos_a, correctos_b):
    """Prueba exacta de McNemar sobre los mismos dibujos.

    b: dibujos que A acierta y B falla; c: los que B acierta y A falla. Si los
    dos fueran igual de buenos, b y c serían parecidos. Con la prueba exacta
    (binomial), el ejemplo del avance (b=40, c=20) da p = 0,013.
    """
    a = np.asarray(correctos_a, dtype=bool)
    b_ = np.asarray(correctos_b, dtype=bool)
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    p = 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)
    return {"b": b, "c": c, "p": p}


def spearman(x, y):
    r = stats.spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue)}
