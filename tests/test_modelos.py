"""Modelos (docs/03_modelos.md)."""
import torch

from quickdraw.modelos import CNN, MLP, contar_parametros, crear_modelo, preparar_entrada


def test_formas():
    x = preparar_entrada(torch.zeros(4, 28, 28, dtype=torch.uint8))
    assert x.shape == (4, 1, 28, 28)
    assert MLP()(x).shape == (4, 30)
    cnn = CNN()
    assert cnn(x).shape == (4, 30)
    assert cnn.vectores(x).shape == (4, 128)


def test_parametros_de_la_capa_final():
    # 128 entradas x 30 clases + 30 sesgos = 3870
    assert contar_parametros(CNN().clasificador) == 3870


def test_m2_y_m2p_son_la_misma_arquitectura():
    assert contar_parametros(crear_modelo("M2", 0.3)) == contar_parametros(crear_modelo("M2-P", 0.0))


def test_esquema_mixto():
    import numpy as np
    from quickdraw.entrenar import indices_de_epoca
    cache = {"K": np.full(2000, 5), "inicio": np.arange(2000) * 5}
    idx = indices_de_epoca(cache, "mixto", np.random.default_rng(0), prob_completo=0.5)
    completos = (idx - cache["inicio"] == 4).mean()
    assert 0.55 < completos < 0.65        # 0,5 completos + 0,5 * 1/5 que salen completos al azar
