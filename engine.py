import json
import os
from joblib import dump as jdump, load as jload
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd

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


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def train_all(X_train, y_train) -> dict:
    print("  Entrenando KNN...")
    knn = GridSearchCV(
        Pipeline([("sc", StandardScaler()), ("knn", KNeighborsRegressor())]),
        {'knn__n_neighbors': range(1, 51),
         'knn__weights':     ['uniform', 'distance'],
         'knn__metric':      ['euclidean', 'manhattan']},
        cv=5, n_jobs=-1).fit(X_train, y_train)

    print("  Entrenando SVR...")
    svr = GridSearchCV(
        Pipeline([("sc", StandardScaler()), ("svr", SVR())]),
        {'svr__kernel':  ['rbf', 'linear'],
         'svr__C':       [1e3, 1e4, 1e5],
         'svr__epsilon': [1e3, 1e4, 5e4],
         'svr__gamma':   ['scale', 'auto']},
        cv=5, n_jobs=-1).fit(X_train, y_train)

    print("  Entrenando Árbol...")
    tree = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        {'max_depth':         [3, 5, 8, 10, 15],
         'min_samples_split': [2, 5, 10, 20],
         'min_samples_leaf':  [1, 2, 4, 8]},
        cv=5, n_jobs=-1).fit(X_train, y_train)

    return {"KNN": knn.best_estimator_, "SVR": svr.best_estimator_, "Árbol": tree.best_estimator_}


# ── Persistencia ──────────────────────────────────────────────────────────────

def models_exist() -> bool:
    return (os.path.exists(PARAMS_FILE) and
            all(os.path.exists(f"{MODELS_DIR}/{f}") for f in JOBLIB_FILES.values()))


def save_models(modelos: dict, maes: dict) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    for nombre, modelo in modelos.items():
        jdump(modelo, f"{MODELS_DIR}/{JOBLIB_FILES[nombre]}")
    def _params(m):
        est = list(m.named_steps.values())[-1] if hasattr(m, 'named_steps') else m
        return {k: v for k, v in est.get_params().items() if not callable(v)}
    with open(PARAMS_FILE, "w") as f:
        json.dump({n: {"params": _params(m), "MAE": round(maes[n], 2)}
                   for n, m in modelos.items()}, f, indent=2)


def load_models() -> tuple[dict, dict]:
    modelos = {n: jload(f"{MODELS_DIR}/{f}") for n, f in JOBLIB_FILES.items()}
    with open(PARAMS_FILE) as f:
        data = json.load(f)
    return modelos, {n: data[n]["MAE"] for n in data}


def setup() -> tuple[dict, dict]:
    """Punto de entrada: carga modelos o los entrena si es la primera vez."""
    if models_exist():
        print("Modelos ya entrenados — cargando desde disco.")
        return load_models()
    print("Primera ejecución — entrenando modelos...")
    X_train, X_test, y_train, y_test = load_data()
    modelos = train_all(X_train, y_train)
    maes = {n: mean_absolute_error(y_test, m.predict(X_test)) for n, m in modelos.items()}
    save_models(modelos, maes)
    return modelos, maes


# ── Predicción ────────────────────────────────────────────────────────────────

def predict(modelo, features: dict) -> float:
    age = 2024 - features["yearbuilt"]
    X = pd.DataFrame([[
        features["area"], features["bedrooms"], features["bathrooms"],
        features["floors"], features["location"], features["condition"],
        features["garage"], features["view"], age,
    ]], columns=FEATURE_COLS)
    return float(modelo.predict(X)[0])


# ── Evaluación (para correr desde consola) ────────────────────────────────────

if __name__ == "__main__":
    modelos, maes = setup()
    _, X_test, _, y_test = load_data()

    print()
    print(f"{'Modelo':<8} {'R²':>7} {'RMSE':>12} {'MAE':>12}")
    print("-" * 42)
    for nombre, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        print(f"{nombre:<8} {r2:>7.4f} {rmse:>12.2f} {mae:>12.2f}")
