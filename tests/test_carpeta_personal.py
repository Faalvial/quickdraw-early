"""La carpeta personal copia modelos y resultados, pero comparte la caché."""
from pathlib import Path

from quickdraw.config import Config
from quickdraw.experimentos import preparar_carpeta_personal


def test_carpeta_personal(tmp_path):
    oficial = tmp_path / "trabajo"
    (oficial / "modelos" / "M2_s0").mkdir(parents=True)
    (oficial / "modelos" / "M2_s0" / "modelo.pt").write_text("pesos")
    (oficial / "modelos" / "M2_s0" / "checkpoint.pt").write_text("pesado")
    (oficial / "resultados" / "predicciones_validacion").mkdir(parents=True)
    (oficial / "resultados" / "tabla_busqueda.csv").write_text("a,b")

    personal = preparar_carpeta_personal(oficial, tmp_path / "trabajo_elias")
    assert (personal / "modelos" / "M2_s0" / "modelo.pt").read_text() == "pesos"
    assert not (personal / "modelos" / "M2_s0" / "checkpoint.pt").exists()
    assert not (personal / "resultados" / "predicciones_validacion").exists()
    assert (personal / "resultados" / "tabla_busqueda.csv").exists()

    # La segunda vez no copia nada: lo que el integrante cambió se conserva.
    (personal / "resultados" / "tabla_busqueda.csv").write_text("cambiado")
    preparar_carpeta_personal(oficial, personal)
    assert (personal / "resultados" / "tabla_busqueda.csv").read_text() == "cambiado"

    cfg = Config(ruta_trabajo=personal, ruta_cache_comun=oficial / "cache")
    assert cfg.ruta_cache == oficial / "cache"
    assert cfg.ruta_modelos == personal / "modelos"
    assert Config(ruta_trabajo=personal).ruta_cache == Path(personal) / "cache"
