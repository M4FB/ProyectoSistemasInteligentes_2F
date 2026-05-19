import json
import os
import tkinter as tk
from tkinter import ttk

import pandas as pd
from joblib import dump as jdump, load as jload
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# ── Constantes ────────────────────────────────────────────────────────────────

FEATURE_COLS = ['Area', 'Bedrooms', 'Bathrooms', 'Floors',
                'Location', 'Condition', 'Garage', 'View', 'Age']
MODELS_DIR   = "models"
PARAMS_FILE  = f"{MODELS_DIR}/best_params.json"
JOBLIB_FILES = {"KNN": "knn.joblib", "SVR": "svr.joblib", "Árbol": "tree.joblib"}

# ── Datos ─────────────────────────────────────────────────────────────────────

def load_data():
    ds = pd.read_csv("dataset/house_price_all_numeric.csv")
    ds['Age'] = 2024 - ds['YearBuilt']
    X = ds.drop(columns=['Price', 'Id', 'YearBuilt'])
    y = ds['Price']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ── Entrenamiento / persistencia ──────────────────────────────────────────────

def models_exist():
    return (os.path.exists(PARAMS_FILE) and
            all(os.path.exists(f"{MODELS_DIR}/{f}") for f in JOBLIB_FILES.values()))

def train_all(X_train, y_train):
    print("Entrenando KNN...")
    knn = GridSearchCV(
        Pipeline([("sc", StandardScaler()), ("knn", KNeighborsRegressor())]),
        {'knn__n_neighbors': range(1, 51),
         'knn__weights': ['uniform', 'distance'],
         'knn__metric':  ['euclidean', 'manhattan']},
        cv=5, n_jobs=-1).fit(X_train, y_train)

    print("Entrenando SVR...")
    svr = GridSearchCV(
        Pipeline([("sc", StandardScaler()), ("svr", SVR())]),
        {'svr__kernel':  ['rbf', 'linear'],
         'svr__C':       [1e3, 1e4, 1e5],
         'svr__epsilon': [1e3, 1e4, 5e4],
         'svr__gamma':   ['scale', 'auto']},
        cv=5, n_jobs=-1).fit(X_train, y_train)

    print("Entrenando Árbol...")
    tree = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        {'max_depth':         [3, 5, 8, 10, 15],
         'min_samples_split': [2, 5, 10, 20],
         'min_samples_leaf':  [1, 2, 4, 8]},
        cv=5, n_jobs=-1).fit(X_train, y_train)

    return {"KNN": knn.best_estimator_, "SVR": svr.best_estimator_, "Árbol": tree.best_estimator_}

def save_models(modelos, maes):
    os.makedirs(MODELS_DIR, exist_ok=True)
    for nombre, modelo in modelos.items():
        jdump(modelo, f"{MODELS_DIR}/{JOBLIB_FILES[nombre]}")
    def params(m):
        est = list(m.named_steps.values())[-1] if hasattr(m, 'named_steps') else m
        return {k: v for k, v in est.get_params().items() if not callable(v)}
    with open(PARAMS_FILE, "w") as f:
        json.dump({n: {"params": params(m), "MAE": round(maes[n], 2)}
                   for n, m in modelos.items()}, f, indent=2)

def load_models():
    modelos = {n: jload(f"{MODELS_DIR}/{f}") for n, f in JOBLIB_FILES.items()}
    with open(PARAMS_FILE) as f:
        data = json.load(f)
    return modelos, {n: data[n]["MAE"] for n in data}

def cargar_o_entrenar():
    if models_exist():
        print("Cargando modelos desde disco...")
        return load_models()
    X_train, X_test, y_train, y_test = load_data()
    modelos = train_all(X_train, y_train)
    maes = {n: mean_absolute_error(y_test, m.predict(X_test)) for n, m in modelos.items()}
    save_models(modelos, maes)
    return modelos, maes

def predict(modelo, features: dict) -> float:
    age = 2024 - features["yearbuilt"]
    X = pd.DataFrame([[
        features["area"], features["bedrooms"], features["bathrooms"],
        features["floors"], features["location"], features["condition"],
        features["garage"], features["view"], age,
    ]], columns=FEATURE_COLS)
    return float(modelo.predict(X)[0])

# ── Interfaz ──────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self, modelos, maes):
        super().__init__()
        self.modelos = modelos
        self.maes    = maes
        self.modelo_var = tk.StringVar(value="SVR")
        self.title("Predictor de Precio de Casas")
        self.resizable(False, False)
        self._build()

    def _build(self):
        P = {"padx": 10, "pady": 3}

        # Título
        ttk.Label(self, text="   Predictor de Precio de Casas",
                  font=("", 13, "bold")).grid(row=0, column=0, columnspan=4, pady=(12, 2))
        ttk.Label(self, text="★  SVR es el modelo más preciso de los tres",
                  foreground="green").grid(row=1, column=0, columnspan=4)
        ttk.Separator(self).grid(row=2, column=0, columnspan=4, sticky="ew", padx=10, pady=6)

        # Selector de modelo
        frm = ttk.LabelFrame(self, text="Modelo")
        frm.grid(row=3, column=0, columnspan=4, sticky="ew", padx=10, pady=4)
        for col, nombre in enumerate(["KNN", "SVR", "Árbol"]):
            lbl = f"{nombre}  ★ mejor" if nombre == "SVR" else nombre
            ttk.Radiobutton(frm, text=lbl, variable=self.modelo_var,
                            value=nombre).grid(row=0, column=col, padx=20, pady=4)

        ttk.Separator(self).grid(row=4, column=0, columnspan=4, sticky="ew", padx=10, pady=6)

        # Entradas: (fila, col_lbl, col_w, label, key, default/opciones)
        # Las opciones son lista de (etiqueta, valor) para combos, o str para entries
        CAMPOS = [
            (5, 0, 1, "Area (sqft):",    "area",      "1800"),
            (5, 2, 3, "Año construido:", "yearbuilt", "1990"),
            (6, 0, 1, "Habitaciones:",   "bedrooms",  "3"),
            (6, 2, 3, "Pisos:",          "floors",    "1"),
            (7, 0, 1, "Baños:",          "bathrooms", "2"),
            (7, 2, 3, "Garage:",         "garage",
             [("No", 0), ("Sí", 1)]),
            (8, 0, 1, "Ubicación:",      "location",
             [("0 - Rural", 0), ("1 - Suburban", 1), ("2 - Urban", 2), ("3 - Downtown", 3)]),
            (8, 2, 3, "Condición:",      "condition",
             [("0 - Poor", 0), ("1 - Fair", 1), ("2 - Good", 2),
              ("3 - Very Good", 3), ("4 - Excellent", 4)]),
            (9, 0, 1, "Vista (0-4):",    "view",
             [(str(i), i) for i in range(5)]),
        ]

        # widgets[key] = {"type": "entry"/"combo", "var": StringVar, "values": [...]}
        self.inputs = {}

        for fila, cl, cw, label, key, opt in CAMPOS:
            ttk.Label(self, text=label, anchor="e").grid(row=fila, column=cl, sticky="e", **P)
            if isinstance(opt, str):
                var = tk.StringVar(value=opt)
                ttk.Entry(self, textvariable=var, width=14).grid(
                    row=fila, column=cw, sticky="w", **P)
                self.inputs[key] = {"type": "entry", "var": var}
            else:
                etiquetas = [e for e, _ in opt]
                valores   = [v for _, v in opt]
                default   = etiquetas[2] if len(etiquetas) > 2 else etiquetas[0]
                var = tk.StringVar(value=default)
                cb  = ttk.Combobox(self, textvariable=var, values=etiquetas,
                                   state="readonly", width=18)
                cb.grid(row=fila, column=cw, sticky="w", **P)
                self.inputs[key] = {"type": "combo", "var": var,
                                    "labels": etiquetas, "values": valores}

        ttk.Separator(self).grid(row=10, column=0, columnspan=4, sticky="ew", padx=10, pady=6)

        ttk.Button(self, text="  Calcular Precio  ",
                   command=self._calcular).grid(row=11, column=0, columnspan=4, pady=4)

        self.frm_res = ttk.LabelFrame(self, text="Resultado")
        self.frm_res.grid(row=12, column=0, columnspan=4, sticky="ew", padx=10, pady=(4, 14))
        self.lbl_res = ttk.Label(self.frm_res,
                                 text="Ingresa los datos y presiona Calcular.",
                                 font=("", 10), justify="left")
        self.lbl_res.grid(row=0, column=0, padx=12, pady=8, sticky="w")

    _LIMITS: dict = {
        "area":      (50,   20_000, "Area (sqft)"),
        "yearbuilt": (1800, 2024,   "Año construido"),
        "bedrooms":  (1,    20,     "Habitaciones"),
        "bathrooms": (1,    10,     "Baños"),
        "floors":    (1,    5,      "Pisos"),
    }

    def _calcular(self):
        errores = []
        features = {}

        for key, info in self.inputs.items():
            if info["type"] == "entry":
                lo, hi, nombre = self._LIMITS[key]
                raw = info["var"].get().strip()
                try:
                    n = int(raw)
                except ValueError:
                    errores.append(f"• {nombre}: debe ser un entero")
                    continue
                if not lo <= n <= hi:
                    errores.append(f"• {nombre}: debe estar entre {lo} y {hi}")
                    continue
                features[key] = n
            else:
                idx = info["labels"].index(info["var"].get())
                features[key] = info["values"][idx]

        if errores:
            self.lbl_res.config(
                foreground="red",
                text="⚠  Corrige los siguientes campos:\n" + "\n".join(errores))
            return

        modelo = self.modelos[self.modelo_var.get()]
        precio = predict(modelo, features)
        mae    = self.maes[self.modelo_var.get()]

        self.lbl_res.config(
            foreground="black",
            text=(f"Modelo:            {self.modelo_var.get()}\n"
                  f"Precio estimado:   ${precio:,.0f}\n"
                  f"Rango aproximado:  ${max(0, precio - mae):,.0f}  -  ${precio + mae:,.0f}\n"
                  f"Margen de error:   ± ${mae:,.0f}  (MAE del modelo)"))


if __name__ == "__main__":
    modelos, maes = cargar_o_entrenar()
    App(modelos, maes).mainloop()
