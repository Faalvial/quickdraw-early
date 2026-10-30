# 00 · Cómo ejecutar el proyecto

Este documento explica cómo correr todo, qué se guarda, cómo retomar cuando Colab corta la sesión y **qué borrar cuando modifican algo**.

## 1. Preparar Drive (una sola vez)

1. Descompriman `quickdraw-early.zip` y suban la carpeta completa a **Mi unidad**, de modo que quede `MyDrive/quickdraw-early/`.
2. Descompriman `muestra_30.zip` (unos 62 MB) dentro de `MyDrive/quickdraw-early/datos/`, de modo que quede `MyDrive/quickdraw-early/datos/muestra_30/cat.ndjson.gz` y los demás.
   - Si no suben la muestra, el notebook la descarga sola con `datos/download_sample.py`, pero recorre los archivos completos de las 30 categorías y puede tardar alrededor de una hora.
3. Si otra persona del grupo va a usar la misma carpeta: compártanla y, desde su cuenta, *clic derecho → Organizar → Agregar acceso directo a Mi unidad*. La ruta tiene que quedar igual: `MyDrive/quickdraw-early`.

Si prefieren otra ruta, cambien `RAIZ` en la primera celda del notebook.

## 2. Primera prueba: modo rápido

1. Abran `notebooks/proyecto_quickdraw.ipynb` con Colab.
2. *Entorno de ejecución → Cambiar tipo de entorno → T4 GPU.*
3. En la primera celda pongan `MODO_RAPIDO = True`.
4. *Entorno de ejecución → Ejecutar todas.*

El modo rápido (`config_rapida` en `quickdraw/config.py`) usa 120 dibujos por categoría y 2 épocas. Tarda alrededor de un minuto y sirve **solo** para comprobar que todo corre de punta a punta. Sus números no significan nada. Guarda todo en `trabajo_rapido/`, separado de la corrida real.

## 3. La corrida real

Vuelvan a poner `MODO_RAPIDO = False` y ejecuten todo. El notebook avanza por etapas:

| Etapa | Qué hace | Tiempo aproximado en Colab con T4 |
|---|---|---|
| Datos | Lee 300 000 dibujos y rasteriza sus ~2,2 millones de prefijos | 10 a 20 min la primera vez; luego se lee la caché |
| Búsqueda | 12 combinaciones de M2 y 3 de M1, con 5 épocas y el 30 % de los datos | ~0,4 h |
| Entrenamiento | M1 (1 semilla), M2 y M2-P (3 semillas cada uno), hasta 20 épocas | ~2,2 h |
| Evaluación | Métricas, hipótesis y figuras en validación | Unos minutos |

Los tiempos de GPU vienen de la estimación del plan v5 (~2,6 h en total) y del tiempo medido aquí en CPU; midan los suyos en la semana 9 y ajusten. La caché ocupa unos 1,5 GB en Drive (entrenamiento + validación) y cabe en la memoria de Colab.

**Repartir el trabajo entre cuentas.** Cada modelo y cada semilla se guardan en su propia carpeta, así que dos personas pueden entrenar a la vez sobre la misma carpeta compartida sin pisarse. Por ejemplo, en una cuenta:

```python
modelos = E.entrenar_modelos(cfg, D, hp, nombres=("M2",))
```

y en la otra `nombres=("M2-P",)`. Al final, una sola persona ejecuta todo el notebook: los modelos ya entrenados solo se cargan.

## 4. Retomar cuando Colab corta la sesión

No hay que hacer nada especial: vuelvan a ejecutar todas las celdas. Cada etapa revisa qué ya existe en `trabajo/`:

- la caché de prefijos se lee del disco;
- la búsqueda salta las combinaciones ya probadas;
- un modelo terminado (`modelo.pt`) solo se carga;
- un modelo a medio entrenar sigue desde su `checkpoint.pt`, en la época donde quedó;
- las predicciones guardadas se reutilizan, salvo que el modelo se haya reentrenado después.

## 5. Qué se guarda y dónde

```
trabajo/
  cache/                    prefijos rasterizados (entrenamiento_n10000_r28_..., validacion_..., prueba_...)
  modelos/M2-P_s0/          checkpoint.pt, modelo.pt, historial.json, hiperparametros.json
  resultados/
    tabla_hechos.csv        dibujos, mediana de trazos y país más frecuente por categoría
    particiones/            key_id de cada dibujo en entrenamiento, validación y prueba
    busqueda_M1.json, busqueda_M2.json, tabla_busqueda.csv, hiperparametros_elegidos.json
    fig_prefijos.png
    predicciones_validacion/   predicción de cada modelo para cada prefijo
    validacion/             metricas.csv, metricas_media_desviacion.csv, hipotesis.json,
                            separacion_pares.csv y las figuras fig_*.png
    prueba/                 lo mismo, solo después de la semana 13
```

## 6. Si modifican algo: qué se recalcula solo y qué hay que borrar

El código protege contra la mezcla de resultados viejos y nuevos:

| Si cambian… | Qué pasa |
|---|---|
| Rasterización (`resolucion`, `grosor`, `margen`, `lienzo`), `dibujos_por_categoria` o `semilla_particion` | Se crea una caché nueva (el nombre del archivo lleva esos valores). Los modelos guardados ya no coinciden y el notebook **se detiene con un mensaje**: borren `trabajo/modelos/` y `trabajo/resultados/predicciones_*`. |
| La rejilla de búsqueda | Solo se prueban las combinaciones nuevas; las que salieron de la rejilla se ignoran. |
| Hiperparámetros, `lote`, `epocas_max`, `paciencia` | El notebook se detiene con un mensaje que muestra la configuración guardada y la actual. Borren la carpeta del modelo que quieren reentrenar. |
| La arquitectura (`quickdraw/modelos.py`) o la forma de entrenar (`quickdraw/entrenar.py`) | **Esto no se detecta solo.** Borren `trabajo/modelos/` y `trabajo/resultados/predicciones_*`. |
| Métricas, hipótesis o figuras | Basta con volver a ejecutar la sección 5 del notebook: no hace falta reentrenar. |

Si dudan, borrar `trabajo/` entero siempre es seguro: solo cuesta volver a calcular.

## 7. El conjunto de prueba (semana 13)

Mientras `USAR_PRUEBA = False`, ninguna función toca los dibujos de prueba: ni se rasterizan, y `predecir_todo(..., "prueba")` lanza un error. En la semana 13, con los modelos congelados:

1. pongan `USAR_PRUEBA = True` en la primera celda;
2. ejecuten todo **una sola vez**;
3. no cambien nada después de ver los resultados de prueba. Si algo sale mal, se reporta tal cual en el paper.

## 8. Fuera de Colab

```bash
pip install -r requirements.txt
cd notebooks && jupyter notebook proyecto_quickdraw.ipynb
```

Fuera de Colab, el notebook usa como raíz la carpeta de arriba de `notebooks/` (o la variable de entorno `QD_RAIZ`). Las pruebas automáticas se ejecutan desde la raíz del repositorio con `pytest tests`.

## 9. Resultados de la verificación (no son del proyecto)

Para comprobar que el código aprende, se hizo una corrida mediana en CPU: 1 000 dibujos por categoría, 4 épocas y 1 semilla. Exactitud final: M1 0,52, M2 0,69 y M2-P 0,63. AUC progresiva: 0,39, 0,50 y 0,53. Con tan poco entrenamiento estos números **no sirven para el paper**; solo muestran que el orden esperado aparece. En esa corrida H2 salió refutada porque M2-P perdió 5,8 puntos de exactitud final; en `04_entrenamiento_y_busqueda.md` hay una modificación posible si eso se repite en la corrida real.
