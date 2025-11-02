# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
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

    cols=['SEX','EDUCATION','MARRIAGE']
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
    "selectkbest__k": [1, (len(x_train.columns)+1)],
    "estimator__penalty":["l1"],
    "estimator__solver": ["liblinear"],
    "estimator__C": [0.095,0.099,1],
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

