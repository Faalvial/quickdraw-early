# 06 · Estadística y evaluación de las hipótesis

**Archivos:** `quickdraw/estadistica.py`, `quickdraw/hipotesis.py`
**Responsable en el plan:** Omar (H1, H2 y H5) y Farid (H4)
**Teoría:** secciones 3.6.2 (bootstrap), 3.6.3 (McNemar) y 3.8 (hipótesis) del avance; sección 1 del plan v5

## Las pruebas (`estadistica.py`)

- **`bootstrap(valores, n=1000)`**: remuestrea **dibujos** con reemplazo, recalcula el promedio y toma los percentiles 2,5 y 97,5. Devuelve (promedio, inferior, superior). Se remuestrean dibujos y no prefijos porque los prefijos de un mismo dibujo no son independientes.
- **`mcnemar(correctos_a, correctos_b)`**: prueba exacta (binomial) sobre los mismos dibujos. b = dibujos que A acierta y B falla; c = al revés. Si fueran igual de buenos, b y c se parecerían. Con el ejemplo del avance (b = 40, c = 20) da p = 0,013, y `tests/test_estadistica.py` lo comprueba.
- **`spearman(x, y)`**: correlación de rangos; no supone una relación lineal.

## Las hipótesis (`hipotesis.py`)

Cada función evalúa una hipótesis **en cada semilla** y la marca "confirmada" solo si el criterio se cumple en las tres. Los criterios están fijados de antemano; no se ajustan después de ver los resultados.

| Hip. | Modelo | Criterio para confirmar | Archivo con el detalle |
|---|---|---|---|
| H1 | M2 | Exactitud a f = 1 menos exactitud a f = 0,5 > 10 puntos **y** McNemar p < 0,05 (dibujos con K ≥ 3) | `hipotesis.json` |
| H2 | M2 frente a M2-P | IC 95 % bootstrap de la diferencia de AUC por dibujo (M2-P − M2) con límite inferior > 0 **y** M2-P no pierde más de 1 punto de exactitud final | `hipotesis.json` |
| H4 | M2 y M2-P | Spearman ρ > 0 con p < 0,05 entre confundibilidad y mediana de k*/K (30 categorías) | `hipotesis.json`, `fig_h4.png` |
| H5 | M2-P | Proporción de dibujos con 0 < k* < K, con IC 95 % bootstrap de límite inferior > 0,5 (K ≥ 3) | `hipotesis.json`, `fig_h5.png` |

H3 (calibración y regla de parada) salió con la reducción del alcance del plan v5; se mantiene la numeración del avance para no confundir.

La celda "Detalle de cada hipótesis" del notebook muestra, semilla por semilla, los valores que deciden cada veredicto (`caida_puntos`, `mcnemar_p`, `ic_inf`, `rho`, etc.).

## Decisiones y por qué

- **Tres semillas, todas deben cumplir.** Una semilla puede tener suerte. Exigir las tres es más estricto que promediar y evita que una semilla buena tape dos malas.
- **H1 con McNemar y no con dos intervalos.** Las dos exactitudes se miden sobre los mismos dibujos; McNemar usa esa pareja (solo cuentan los dibujos donde los dos casos difieren) y es más potente.
- **H2 con dos condiciones.** Acertar antes no sirve si se pierde el final. La tolerancia de 1 punto es la del plan.
- **H5 sobre dibujos reconocidos.** Toda la muestra es de dibujos que el juego reconoció, así que el último trazo es, en la práctica, el momento en que la IA del juego acertó. Si el modelo acierta antes en más de la mitad, el modelo se anticipa al juego.

## Dónde modificar

- Más remuestreos o otro nivel de confianza: `Config.bootstrap_remuestreos`, y el argumento `nivel` de `bootstrap`.
- Cambiar un criterio: la línea `"cumple": ...` de cada función. **Solo antes de mirar los resultados de prueba**, y declarándolo en el paper.
- Evaluar con menos semillas (por ejemplo si no alcanzó el tiempo): `Config.semillas=(0,)`. El veredicto pasa a depender de una sola; hay que decirlo como limitación.
