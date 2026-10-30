"""Particiones por dibujo (docs/01_datos.md)."""
import numpy as np

from quickdraw.config import Config
from quickdraw.datos import particionar


def test_sin_solape_y_estratificada():
    dibujos = [{"etiqueta": e, "key_id": f"{e}-{i}"} for e in range(3) for i in range(100)]
    partes = particionar(dibujos, Config())
    conjuntos = [set(v.tolist()) for v in partes.values()]
    assert sum(len(c) for c in conjuntos) == 300
    assert len(set.union(*conjuntos)) == 300                 # nadie en dos particiones
    etiquetas = np.array([d["etiqueta"] for d in dibujos])
    for nombre, n in (("entrenamiento", 70), ("validacion", 15), ("prueba", 15)):
        assert (np.bincount(etiquetas[partes[nombre]]) == n).all()


def test_reproducible():
    dibujos = [{"etiqueta": i % 2, "key_id": str(i)} for i in range(200)]
    a, b = particionar(dibujos, Config()), particionar(dibujos, Config())
    assert all(np.array_equal(a[k], b[k]) for k in a)
