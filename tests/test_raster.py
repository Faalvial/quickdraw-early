"""Prefijos y rasterización (docs/02_prefijos_y_rasterizacion.md)."""
import numpy as np

from quickdraw.config import Config
from quickdraw.raster import (construir_cache, indices_completos, indices_fraccion,
                              k_de_fraccion, normalizar, rasterizar)

CFG = Config()
# Dibujo de 3 trazos: una línea horizontal arriba, otra vertical y una diagonal
# que se sale de la caja de los dos primeros.
TRAZOS = [[[0, 100], [0, 0]], [[50, 50], [0, 100]], [[100, 255], [100, 200]]]


def test_k_de_fraccion():
    assert k_de_fraccion(7, 0.3) == 3          # ceil(2,1) = 3, ejemplo del avance
    assert k_de_fraccion(10, 0.3) == 3         # 0,3*10 no debe dar 4 por redondeo
    assert k_de_fraccion(2, 0.1) == 1          # nunca menos de un trazo
    assert k_de_fraccion(9, 1.0) == 9


def test_causalidad():
    """La imagen del prefijo k no depende de los trazos k+1, ..., K."""
    a = rasterizar(TRAZOS[:2], CFG)
    cache = construir_cache([{"trazos": TRAZOS, "etiqueta": 0}], [0], CFG, procesos=1)
    assert np.array_equal(cache["imagenes"][1], a)
    otro = [{"trazos": TRAZOS[:2] + [[[0, 255], [255, 0]]], "etiqueta": 0}]
    cache2 = construir_cache(otro, [0], CFG, procesos=1)
    assert np.array_equal(cache2["imagenes"][1], a)
    assert not np.array_equal(cache2["imagenes"][2], cache["imagenes"][2])


def test_un_solo_punto_al_centro():
    img = rasterizar([[[10], [10]]], CFG)
    ys, xs = np.nonzero(img)
    assert img.max() > 0
    assert abs(ys.mean() - 13.5) < 1.5 and abs(xs.mean() - 13.5) < 1.5


def test_conserva_proporcion():
    """Una línea horizontal sigue siendo horizontal (un solo factor de escala)."""
    norm = normalizar([[[0, 200], [50, 50]]], CFG)
    xs, ys = norm[0]
    assert ys[0] == ys[1]
    assert xs[1] - xs[0] > 200


def test_cabe_con_margen():
    img = rasterizar(TRAZOS, CFG)
    assert img.shape == (28, 28) and img.dtype == np.uint8
    assert img[0].max() < 128 and img[-1].max() < 128      # el margen queda casi vacío


def test_indices_de_cache():
    dibujos = [{"trazos": TRAZOS, "etiqueta": 4}, {"trazos": TRAZOS[:1], "etiqueta": 7}]
    cache = construir_cache(dibujos, [0, 1], CFG, procesos=1)
    assert len(cache["imagenes"]) == 4
    assert list(cache["inicio"]) == [0, 3]
    assert list(indices_completos(cache)) == [2, 3]
    assert list(indices_fraccion(cache, 0.5)) == [1, 3]    # k = ceil(1,5) = 2; k = 1
