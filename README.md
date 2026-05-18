# Predictor de Precios de Casas

Proyecto de Sistemas Inteligentes que compara tres modelos de regresión (KNN, SVR, Árbol de Decisión) para predecir el precio de una vivienda a partir de sus características.

---

## Cómo ejecutar

### Interfaz de terminal (TUI) — recomendada

```bash
python app.py
```

Usa [Textual](https://textual.textualize.io/) para renderizar una interfaz interactiva directamente en la terminal. Seleccioná el modelo con el radiobutton, ingresá los datos de la casa y presioná **Predecir**.

### Interfaz gráfica (GUI)

```bash
python AltTkinter.py
```

Versión equivalente con ventana Tkinter. No requiere dependencias adicionales más allá de las del proyecto.

> **Nota:** si los modelos entrenados no están presentes en `models/`, ambas interfaces los entrenan automáticamente al iniciar (toma unos segundos).

---

## Dependencias

| Paquete | Versión | Uso |
|---------|---------|-----|
| `scikit-learn` | 1.8.0 | KNN, SVR, Árbol de Decisión, GridSearchCV |
| `pandas` | 3.0.2 | Carga y preprocesamiento del dataset |
| `numpy` | 2.4.3 | Cálculo de métricas |
| `joblib` | 1.5.3 | Serialización de modelos entrenados |
| `textual` | 8.2.6 | Interfaz TUI (`app.py`) |
| `tkinter` | stdlib | Interfaz GUI (`AltTkinter.py`) |

Instalación:

```bash
pip install scikit-learn pandas numpy joblib textual
```

---

## Dataset

- **Fuente:** `dataset/house_price_all_numeric.csv`
- **Filas:** 4 546 registros de viviendas
- **Características usadas:** Area, Bedrooms, Bathrooms, Floors, Location, Condition, Garage, View, Age (derivada de YearBuilt)
- **Variable objetivo:** Price

| Estadístico | Precio (USD) |
|-------------|-------------|
| Mínimo      | 7 800       |
| Mediana     | 465 000     |
| Promedio    | 557 354     |
| Máximo      | 26 590 000  |

---

## Resultados de los modelos (Python — validación 80/20)

| Modelo | R²     | MAE           |
|--------|--------|---------------|
| KNN    | —      | 146 264       |
| **SVR**| —      | **138 173**   |
| Árbol  | —      | 163 335       |

SVR con kernel lineal (C=100 000) obtiene el menor error absoluto medio.

---

## Archivos RapidMiner

En la carpeta `rapidminer/` se incluyen los procesos `.rmp` listos para importar en **RapidMiner Studio 12.x**. No requieren ejecución adicional para el proyecto, pero están disponibles para reproducir los experimentos de validación cruzada y predicción.

| Archivo | Descripción |
|---------|-------------|
| `01_knn.rmp` | Validación cruzada 10-fold del modelo KNN |
| `02_svr.rmp` | Validación cruzada 10-fold del modelo SVR |
| `03_decision_tree.rmp` | Validación cruzada 10-fold del Árbol de Decisión |
| `04_comparacion.rmp` | Comparación paralela de los 3 modelos |
| `05_prediccion_knn.rmp` | Predicción sobre casas de prueba con KNN |
| `06_prediccion_svr.rmp` | Predicción sobre casas de prueba con SVR |
| `07_prediccion_arbol.rmp` | Predicción sobre casas de prueba con Árbol |

Las capturas de pantalla de los resultados en RapidMiner se agregarán próximamente en `img/`.

---

## Estructura del proyecto

```
.
├── app.py                  # Interfaz TUI (Textual)
├── AltTkinter.py           # Interfaz GUI (Tkinter)
├── engine.py               # Entrenamiento, predicción y métricas
├── dataset/
│   ├── house_price_all_numeric.csv   # Dataset limpio
│   ├── house_price_kc_raw.csv        # Dataset original
│   └── test_casas.csv                # 4 casas de ejemplo para predicción
├── models/
│   ├── knn.joblib          # Modelo KNN serializado
│   ├── svr.joblib          # Modelo SVR serializado
│   ├── tree.joblib         # Modelo Árbol serializado
│   └── best_params.json    # Hiperparámetros óptimos encontrados por GridSearch
└── rapidminer/             # Procesos RapidMiner Studio (.rmp)
```
