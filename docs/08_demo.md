# 08 · Demo para la exposición

**Archivos:** `demo/demo.py`, `notebooks/demo.ipynb`
**Responsable en el plan:** Jorly
**Condición del plan:** se hace en la semana 13 y solo si H1 y H2 ya están; si no, se usa `fig_trazo_a_trazo.png`.

## Cómo abrirla

Opción 1, el notebook de la demo (recomendado para la exposición): abran `notebooks/demo.ipynb` en Colab y ejecuten todo. Carga las cachés y los modelos ya entrenados de Drive; **no entrena nada** y se detiene con un mensaje si todavía no hay modelos.

Opción 2, al final del notebook principal, en una celda nueva:

```python
!pip install -q gradio
from demo.demo import lanzar
lanzar(cfg, D, modelos)
```

Con `share=True` (por defecto) Gradio da un enlace público temporal, útil para proyectar desde otra computadora o dejar que el profesor dibuje desde su celular.

## Qué muestra

1. **Dibujo en vivo.** Cada vez que cambia el lienzo se muestran las tres clases más probables de M1, M2 y M2-P, y la imagen de 28×28 que ve el modelo. Lo esperable: con pocos trazos M2-P ya propone la categoría correcta y M2 duda.
2. **Repetición de dibujos reales.** Un dibujo de validación (de 5 a 12 trazos), trazo por trazo con un control deslizante. En el último trazo aparece la nota "aquí acertó la IA del juego", que conecta con H5.

## Decisiones y por qué

- **La demo trabaja con la imagen, no con los trazos.** El lienzo de Gradio entrega una imagen. `imagen_de_lienzo` recorta la tinta con su caja, la centra con el mismo margen y la reduce con `BOX`: es el equivalente en imagen de la normalización causal (la caja de lo dibujado hasta ahora).
- **Casilla "Recentrar".** Si se desactiva, el dibujo se reduce tal cual está en el lienzo. Dibujar un gato pequeño en una esquina con la casilla apagada muestra por qué hace falta normalizar: el modelo nunca vio dibujos así.
- **Semilla 0.** Usa la semilla 0 de M2 y M2-P, y la única de M1.

## Dónde modificar

- Otro grosor de pincel: el trazo de Gradio debería parecerse al de entrenamiento (~1,75 px a 28×28). Si las predicciones en vivo son malas, prueben `gr.Sketchpad(brush=gr.Brush(default_size=...))` en `demo.py`.
- Otra semilla: `lanzar(cfg, D, modelos, semilla=1)`.
- Más o menos dibujos en la repetición: el filtro `5 <= val["K"][i] <= 12` y el `[:300]` en `lanzar`.
- Para probar sin abrir el servidor: `lanzar(..., abrir=False)` devuelve la aplicación sin lanzarla.
