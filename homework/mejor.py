import pandas as pd
import gzip
import pickle
import os

test_data = pd.read_csv(
        f"files/input/test_data.csv.zip",
        index_col=False,
        compression="zip",
    )
train_data = pd.read_csv(
        f"files/input/train_data.csv.zip",
        index_col=False,
        compression="zip",
    )
## Paso 1 Limpiar la data

def clean_data(df):
    # 1. Eliminar la columna 'ID' si existe
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    # 2. Renombrar la columna a 'default' si existe
    if "default payment next month" in df.columns:
        df = df.rename(columns={'default payment next month': 'default'})
    # 3. Reemplazar valores >4 
    if 'EDUCATION' in df.columns:
        df['EDUCATION']=df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    # 4. Eliminar filas con valores NaN
    df = df.dropna()
    
    return df
train_data=clean_data(train_data)
test_data=clean_data(test_data)

## Paso 2 Dividir la data
x_train = train_data.drop(columns=['default'])
y_train = train_data['default']

x_test = test_data.drop(columns=['default'])
y_test = test_data['default']

## Paso 3 Creación del pipepline y ajuste de modelo
def make_pipeline(estimator):

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
    from sklearn.feature_selection import SelectKBest, f_classif

    cols=['SEX','EDUCATION','MARRIAGE','PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    cols_num=[c for c in x_train.columns if c not in cols]
    transformer = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(dtype="int",handle_unknown="ignore"),cols),
            ("num", MinMaxScaler(feature_range=(0,1)), cols_num),
        ],
        remainder="passthrough",
    )
    selectkbest=SelectKBest(score_func=f_classif, k="all")
    
    pipeline = Pipeline(
        steps=[
            ("tranformer", transformer),
            ("selectkbest", selectkbest),
            ("estimator", estimator),
        ],
        verbose=False,
    )

    return pipeline

# Modelo estimador
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(random_state=42, max_iter=4000)
lg_estimator=make_pipeline(lg)

## Paso 4 Optimizar hiperparámetros 10 splits
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, make_scorer
import numpy as np

# Grid mínimo para cumplir la especificación
param_grid = {
    "selectkbest__k": [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,"all"],
    "estimator__penalty":["l2"],
    "estimator__solver": ["liblinear"],
    "estimator__C": np.logspace(-4, 3, 15),
    "estimator__class_weight":[None, "balanced"],
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scorers = {
    "acc": "accuracy",
    "bal": make_scorer(balanced_accuracy_score),
}

grid_search = GridSearchCV(estimator=lg_estimator,                       
    param_grid=param_grid,
    scoring=scorers, 
    refit="acc",
    cv=cv,
    n_jobs=-1,
    verbose=0)



## Paso 5 Guardar el modelo comprimido
grid_search.fit(x_train,y_train)

def save_estimator(estimator):
    # Crear carpeta destino si no existe
    os.makedirs("files/models", exist_ok=True)
    # Guardar modelo comprimido en formato .pkl.gz
    with gzip.open("files/models/model.pkl.gz", "wb") as file:
        pickle.dump(estimator, file)

save_estimator(grid_search)
best_est = grid_search.best_estimator_
print(best_est)

## Paso 6 Métricas de precisión
import json
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

with gzip.open("files/models/model.pkl.gz", "rb") as f:
    loaded_model = pickle.load(f)

y_train_pred = grid_search.predict(x_train)
y_test_pred = grid_search.predict(x_test)

metrics = [
    {
        "type": "metrics",
        "dataset": "train",
        "precision": precision_score(y_train, y_train_pred),
        "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
        "recall": recall_score(y_train, y_train_pred),
        "f1_score": f1_score(y_train, y_train_pred)
    },
    {
        "type": "metrics",
        "dataset": "test",
        "precision": precision_score(y_test, y_test_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1_score": f1_score(y_test, y_test_pred)
    }
]

os.makedirs("files/output", exist_ok=True)

with open("files/output/metrics.json", "w") as f:
    for row in metrics:
        f.write(json.dumps(row) + "\n")

## Paso 7 Matrices de confusión
from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

os.makedirs("files/output", exist_ok=True)

metrics = [
    {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": int(cm_train[0][0]), "predicted_1": int(cm_train[0][1])},
        "true_1": {"predicted_0": int(cm_train[1][0]), "predicted_1": int(cm_train[1][1])}
    },
    {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": int(cm_test[0][0]), "predicted_1": int(cm_test[0][1])},
        "true_1": {"predicted_0": int(cm_test[1][0]), "predicted_1": int(cm_test[1][1])}
    }
]

with open("files/output/metrics.json", "a") as f:
    for row in metrics:
        f.write(json.dumps(row) + "\n")


y_train_proba = grid_search.predict_proba(x_train)[:, 1]
y_test_proba = grid_search.predict_proba(x_test)[:, 1]

print(y_train_proba[:5])
print(y_test_proba[:5])

print(y_train.value_counts(normalize=True))

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

# Probabilidades del modelo
y_test_proba = grid_search.predict_proba(x_test)[:, 1]

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

