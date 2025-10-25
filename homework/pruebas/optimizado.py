
import os
import json
import gzip
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    make_scorer,
    confusion_matrix,
)
import optuna


# ======================================================
# Paso 0: Cargar datos
# ======================================================
train_data = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
test_data = pd.read_csv("files/input/test_data.csv.zip", compression="zip")


# ======================================================
# Paso 1: Limpiar data
# ======================================================
def clean_data(df):
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default"})
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    df = df.dropna()
    return df


train_data = clean_data(train_data)
test_data = clean_data(test_data)


# ======================================================
# Paso 2: Split X / y
# ======================================================
x_train = train_data.drop(columns=["default"])
y_train = train_data["default"]
x_test = test_data.drop(columns=["default"])
y_test = test_data["default"]


# ======================================================
# Paso 3: Crear pipeline base
# ======================================================
def make_pipeline(estimator):
    cols_cat = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    cols_num = [c for c in x_train.columns if c not in cols_cat]

    transformer = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(dtype="int", handle_unknown="ignore"), cols_cat),
            ("num", MinMaxScaler(feature_range=(0, 1)), cols_num),
        ],
        remainder="passthrough",
    )

    selectkbest = SelectKBest(score_func=f_classif, k="all")

    pipeline = Pipeline(
        steps=[
            ("transformer", transformer),
            ("selectkbest", selectkbest),
            ("estimator", estimator),
        ]
    )

    return pipeline


# ======================================================
# Paso 4: Optimización con Optuna (versión mejorada 💡🔧)
# ======================================================

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
prec_scorer = make_scorer(precision_score, zero_division=0)

def optimizacion(trial):
    k = trial.suggest_int("selectkbest__k", 8, 30)
    C = trial.suggest_float("estimator__C", 0.008, 0.1)
    solver = trial.suggest_categorical("estimator__solver", ["liblinear", "lbfgs"])
    if solver == "liblinear":
        penalty = trial.suggest_categorical("estimator__penalty", ["l1", "l2"])
    else:
        penalty = "l2"
    class_weight = trial.suggest_categorical("estimator__class_weight", [None, "balanced"])

    # Modelo base
    estimator = LogisticRegression(
        random_state=42,
        max_iter=4000,
        solver=solver,
        penalty=penalty,
        C=C,
        class_weight=class_weight,
    )

    pipe = make_pipeline(estimator)
    pipe.set_params(selectkbest__k=k)

    # 🔧 CAMBIO CLAVE: Calculamos métricas sin filtrar
    accuracy = cross_val_score(pipe, x_train, y_train, scoring="accuracy", cv=cv, n_jobs=-1).mean()
    precision = cross_val_score(pipe, x_train, y_train, scoring=prec_scorer, cv=cv, n_jobs=-1).mean()

    # Guardamos atributos para análisis posterior
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("precision", precision)

    # 💡 CAMBIO: en vez de descartar (return -1), ponderamos ambas métricas
    score_combined = 0.7 * accuracy + 0.3 * precision  # 70% peso en accuracy, 30% en precision
    return score_combined


study = optuna.create_study(direction="maximize")
study.optimize(optimizacion, n_trials=70, show_progress_bar=False)

# Filtrar sólo los trials que lograron la precisión mínima
valid_trials = [
    t
    for t in study.trials
    if t.user_attrs.get("precision", 0) >= 0.693
    and t.user_attrs.get("threshold", None) is not None
]

if len(valid_trials) == 0:
    # fallback si ninguno cumple precisión >= 0.693
    best_trial = study.best_trial
else:
    # entre los válidos, elegimos el de mejor balanced accuracy
    best_trial = max(valid_trials, key=lambda t: t.user_attrs["ba"])

best_params = best_trial.params
best_threshold = best_trial.user_attrs.get("threshold", 0.5)

# extraemos hiperparámetros
best_k = best_params["selectkbest__k"]
best_C = best_params["estimator__C"]
best_solver = best_params["estimator__solver"]
best_classw = best_params["estimator__class_weight"]

if best_solver == "liblinear":
    best_penalty = best_params["estimator__penalty"]
else:
    best_penalty = "l2"


# ======================================================
# Paso 5: Entrenar el modelo final con los mejores hiperparámetros
# ======================================================

lg_final = LogisticRegression(
    random_state=42,
    max_iter=4000,
    solver=best_solver,
    penalty=best_penalty,
    C=best_C,
    class_weight=best_classw,
)

pipe_final = make_pipeline(lg_final)
pipe_final.set_params(selectkbest__k=best_k)

pipe_final.fit(x_train, y_train)

# Guardar el modelo final comprimido
def save_estimator(estimator):
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as file:
        pickle.dump(estimator, file)

save_estimator(pipe_final)


# ======================================================
# Paso 6: Métricas usando el threshold óptimo
# ======================================================

with gzip.open("files/models/model.pkl.gz", "rb") as f:
    loaded_model = pickle.load(f)

y_train_proba = loaded_model.predict_proba(x_train)[:, 1]
y_test_proba = loaded_model.predict_proba(x_test)[:, 1]

t = best_threshold  # <- el threshold elegido por Optuna bajo prec mínima

y_train_pred = (y_train_proba >= t).astype(int)
y_test_pred = (y_test_proba >= t).astype(int)

metrics = [
    {
        "type": "metrics",
        "dataset": "train",
        "precision": precision_score(y_train, y_train_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
        "recall": recall_score(y_train, y_train_pred),
        "f1_score": f1_score(y_train, y_train_pred),
    },
    {
        "type": "metrics",
        "dataset": "test",
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1_score": f1_score(y_test, y_test_pred),
    },
]

os.makedirs("files/output", exist_ok=True)

with open("files/output/metrics.json", "w", encoding="utf-8") as f:
    for row in metrics:
        f.write(json.dumps(row) + "\n")


# ======================================================
# Paso 7: Matrices de confusión
# ======================================================

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

cm_output = [
    {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {
            "predicted_0": int(cm_train[0][0]),
            "predicted_1": int(cm_train[0][1]),
        },
        "true_1": {
            "predicted_0": int(cm_train[1][0]),
            "predicted_1": int(cm_train[1][1]),
        },
    },
    {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {
            "predicted_0": int(cm_test[0][0]),
            "predicted_1": int(cm_test[0][1]),
        },
        "true_1": {
            "predicted_0": int(cm_test[1][0]),
            "predicted_1": int(cm_test[1][1]),
        },
    },
]

with open("files/output/metrics.json", "a", encoding="utf-8") as f:
    for row in cm_output:
        f.write(json.dumps(row) + "\n")


# Probabilidades del modelo
y_test_proba = loaded_model.predict_proba(x_test)[:, 1]

# Rangos de threshold
thresholds = np.linspace(0.1, 0.9, 200)

results = []
for threshold in thresholds:
    y_pred = (y_test_proba >= threshold).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    results.append((threshold, precision, recall, f1, bal_acc))

# Convertir a DataFrame
results_df = pd.DataFrame(results, columns=["threshold", "precision", "recall", "f1", "balanced_accuracy"])

# Mostrar el top 5 por F1
print(results_df.sort_values("f1", ascending=False).head())

