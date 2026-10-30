"""Demo en Gradio para la exposición. Ver docs/08_demo.md.

Parte 1 · Dibujo en vivo: cada vez que cambia el lienzo se muestran las tres
clases más probables de M1, M2 y M2-P.
Parte 2 · Repetición: un dibujo real de validación, trazo por trazo, con la
predicción de cada modelo y la marca del último trazo (cuando acertó la IA
del juego, H5).

Uso en Colab:
    !pip install -q gradio
    from demo.demo import lanzar
    lanzar(cfg, D, modelos)       # con cfg, D y modelos del notebook principal
"""
import numpy as np
import torch
from PIL import Image

from quickdraw.config import CATEGORIAS, NOMBRES_ES, Config
from quickdraw.entrenar import DISPOSITIVO
from quickdraw.modelos import preparar_entrada

NOMBRES = [NOMBRES_ES[c] for c in CATEGORIAS]


def imagen_de_lienzo(valor, cfg: Config, recentrar: bool = True) -> np.ndarray:
    """Convierte lo que devuelve el Sketchpad en una imagen de 28x28 como las del
    entrenamiento: tinta en 255 y fondo en 0, encuadrada con la caja de lo
    dibujado (el equivalente en imagen de la normalización causal)."""
    if valor is None:
        return np.zeros((cfg.resolucion, cfg.resolucion), np.uint8)
    if isinstance(valor, dict):                        # Gradio 4 o posterior
        valor = valor.get("composite")
    img = Image.fromarray(np.asarray(valor)) if not isinstance(valor, Image.Image) else valor
    if img.mode == "RGBA":                             # fondo transparente → blanco
        fondo = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(fondo, img)
    tinta = 255 - np.asarray(img.convert("L"), dtype=np.uint8)
    tinta = np.where(tinta > 40, tinta, 0).astype(np.uint8)
    if not tinta.any():
        return np.zeros((cfg.resolucion, cfg.resolucion), np.uint8)
    if recentrar:
        ys, xs = np.nonzero(tinta)
        recorte = tinta[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        lado = max(recorte.shape)
        util = cfg.resolucion - 2 * cfg.margen
        lienzo = int(np.ceil(lado * cfg.resolucion / util))
        cuadro = np.zeros((lienzo, lienzo), np.uint8)
        y0 = (lienzo - recorte.shape[0]) // 2
        x0 = (lienzo - recorte.shape[1]) // 2
        cuadro[y0:y0 + recorte.shape[0], x0:x0 + recorte.shape[1]] = recorte
        tinta = cuadro
    return np.asarray(Image.fromarray(tinta).resize((cfg.resolucion,) * 2, Image.BOX), np.uint8)


@torch.no_grad()
def top3(modelo, imagen_uint8) -> dict:
    x = preparar_entrada(torch.from_numpy(imagen_uint8[None]).to(DISPOSITIVO))
    p = torch.softmax(modelo.eval()(x).float(), 1)[0].cpu().numpy()
    orden = np.argsort(p)[::-1][:3]
    return {NOMBRES[i]: float(p[i]) for i in orden}


def lanzar(cfg: Config, D, modelos, semilla: int = 0, share: bool = True, abrir: bool = True):
    import gradio as gr

    elegidos = {n: modelos[(n, semilla if n != "M1" else min(s for m, s in modelos if m == "M1"))]
                for n in ("M1", "M2", "M2-P")}
    val = D["caches"]["validacion"]
    ejemplos = [i for i in range(len(val["K"])) if 5 <= val["K"][i] <= 12][:300]
    opciones = [f"{j}: {NOMBRES[val['etiqueta'][i]]} ({val['K'][i]} trazos)"
                for j, i in enumerate(ejemplos)]

    def en_vivo(valor, recentrar):
        img = imagen_de_lienzo(valor, cfg, recentrar)
        grande = Image.fromarray(255 - img).resize((140, 140), Image.NEAREST)
        return [grande] + [top3(m, img) for m in elegidos.values()]

    def repeticion(opcion, k):
        i = ejemplos[int(opcion.split(":")[0])]
        K = int(val["K"][i])
        k = int(min(k, K))
        img = val["imagenes"][val["inicio"][i] + k - 1]
        grande = Image.fromarray(255 - img).resize((168, 168), Image.NEAREST)
        nota = (f"Trazo {k} de {K}. Verdadera: {NOMBRES[val['etiqueta'][i]]}. "
                + ("Este es el último trazo: aquí acertó la IA del juego." if k == K else ""))
        return [grande, nota] + [top3(m, img) for m in elegidos.values()]

    with gr.Blocks(title="¿Cuántos trazos bastan?") as app:
        gr.Markdown("# ¿Cuántos trazos bastan?\nM1 (MLP), M2 (CNN con dibujos completos) "
                    "y M2-P (CNN con prefijos).")
        with gr.Tab("Dibujo en vivo"):
            with gr.Row():
                lienzo = gr.Sketchpad(label="Dibujen aquí", type="numpy")
                with gr.Column():
                    vista = gr.Image(label="Lo que ve el modelo (28×28)")
                    recentrar = gr.Checkbox(True, label="Recentrar el dibujo "
                                            "(desactivar para el gato en la esquina)")
            salidas = [gr.Label(num_top_classes=3, label=n) for n in elegidos]
            lienzo.change(en_vivo, [lienzo, recentrar], [vista] + salidas)
            recentrar.change(en_vivo, [lienzo, recentrar], [vista] + salidas)
        with gr.Tab("Repetición de dibujos reales"):
            opcion = gr.Dropdown(opciones, value=opciones[0], label="Dibujo de validación")
            k = gr.Slider(1, 12, value=1, step=1, label="Trazos")
            imagen = gr.Image(label="Prefijo")
            nota = gr.Markdown()
            salidas2 = [gr.Label(num_top_classes=3, label=n) for n in elegidos]
            for control in (opcion, k):
                control.change(repeticion, [opcion, k], [imagen, nota] + salidas2)
    if abrir:
        app.launch(share=share)
    return app
