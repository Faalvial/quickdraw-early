"""Métricas progresivas (docs/05_metricas.md)."""
import numpy as np

from quickdraw.metricas import (auc_por_dibujo, auc_progresiva, correctos_por_fraccion,
                                fraccion_estrella, k_estrella)

PERRO, GATO = 1, 0


def cache_de(Ks, etiquetas):
    Ks = np.array(Ks)
    return {"K": Ks, "etiqueta": np.array(etiquetas),
            "inicio": np.concatenate([[0], np.cumsum(Ks)[:-1]])}


def test_k_estrella_ejemplo_del_avance():
    # perro, gato, perro, gato, gato con etiqueta gato: k* = 4
    cache = cache_de([5], [GATO])
    pred = np.array([PERRO, GATO, PERRO, GATO, GATO])
    assert k_estrella(cache, pred)[0] == 4


def test_k_estrella_casos_borde():
    cache = cache_de([3, 3, 3], [GATO, GATO, GATO])
    pred = np.array([GATO, GATO, GATO,       # acierta desde el primer trazo: k* = 1
                     GATO, GATO, PERRO,      # falla al final: k* = 0
                     PERRO, PERRO, GATO])    # solo el último: k* = K = 3
    ks = k_estrella(cache, pred)
    assert list(ks) == [1, 0, 3]
    assert list(fraccion_estrella(cache, ks)) == [1 / 3, 1.0, 1.0]


def test_auc_progresiva():
    f = (0.1, 0.5, 1.0)
    assert np.isclose(auc_progresiva([1, 1, 1], f), 1.0)       # siempre acierta
    assert np.isclose(auc_progresiva([0, 0, 0], f), 0.0)
    # trapecios: (0+1)/2*0,4 + (1+1)/2*0,5 = 0,7; dividido entre 0,9
    assert np.isclose(auc_progresiva([0, 1, 1], f), 0.7 / 0.9)


def test_auc_por_dibujo_promedia_igual():
    rng = np.random.default_rng(0)
    c = rng.random((50, 10)) > 0.4
    assert np.isclose(auc_por_dibujo(c).mean(), auc_progresiva(c.mean(0)))


def test_correctos_por_fraccion_7_trazos():
    # 7 trazos: f = 0,3 usa k = 3, f = 1 usa k = 7
    cache = cache_de([7], [GATO])
    pred = np.array([PERRO, PERRO, GATO, PERRO, PERRO, PERRO, GATO])
    c = correctos_por_fraccion(cache, pred, (0.3, 0.5, 1.0))
    assert c.tolist() == [[True, False, True]]                # f = 0,5 usa k = 4
