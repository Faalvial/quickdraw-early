"""Pruebas estadísticas (docs/06_estadistica_e_hipotesis.md)."""
import numpy as np

from quickdraw.estadistica import bootstrap, mcnemar, spearman


def test_mcnemar_ejemplo_del_avance():
    # b = 40 dibujos que A acierta y B falla; c = 20 al revés
    a = np.array([1] * 40 + [0] * 20 + [1] * 100, bool)
    b = np.array([0] * 40 + [1] * 20 + [1] * 100, bool)
    r = mcnemar(a, b)
    assert (r["b"], r["c"]) == (40, 20)
    assert round(r["p"], 3) == 0.013


def test_mcnemar_sin_discordantes():
    assert mcnemar([1, 0, 1], [1, 0, 1])["p"] == 1.0


def test_bootstrap_contiene_la_media():
    v = np.random.default_rng(1).normal(0.3, 1, 2000)
    media, inf, sup = bootstrap(v, n=300)
    assert inf < media < sup
    assert inf > 0.2 and sup < 0.4


def test_spearman_monotono():
    r = spearman([1, 2, 3, 4, 5], [2, 4, 5, 9, 20])
    assert np.isclose(r["rho"], 1.0)


def test_tablas_latex(tmp_path):
    import pandas as pd
    from quickdraw import tablas
    m = pd.DataFrame([{"modelo": "M2", "semilla": s, "grupo": "todas", "exactitud_final": 0.8 + s / 100,
                       "top3_final": 0.9, "auc_progresiva": 0.5} for s in range(3)])
    texto = tablas.modelos(m, tmp_path / "t.tex")
    assert "81,0 $\\pm$ 1,0" in texto
    r = [{"hipotesis": "H1", "veredicto": "confirmada",
          "por_semilla": [{"caida_puntos": 12.34}, {"caida_puntos": 11.0}]}]
    assert "12,3; 11,0" in tablas.hipotesis(r, tmp_path / "h.tex")
