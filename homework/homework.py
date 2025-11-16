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

import os
import json
import gzip
import pickle
from glob import glob
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix

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
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#

def load_data():
    DATA_PATH = "files/input/"
    df_train = pd.read_csv(DATA_PATH +  "train_data.csv.zip")
    df_test = pd.read_csv(DATA_PATH + "test_data.csv.zip")
    return df_train, df_test

def clean_data(df):
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])

    df = df[df['EDUCATION'] != 0]
    df = df[df['MARRIAGE'] != 0]
    
    df = df.dropna()

    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    
    return df

df_train, df_test = load_data()
df_train = clean_data(df_train)
df_test = clean_data(df_test)

#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#

# --- 1. División del Dataset de Entrenamiento ---
x_train = df_train.drop(columns=['default'])
y_train = df_train['default']

# --- 2. División del Dataset de Prueba ---
x_test = df_test.drop(columns=['default'])
y_test = df_test['default']

#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#

# 1. Definir las columnas
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
numerical_features = list(set(x_train.columns).difference(categorical_features))

# 2. Crear el transformador de preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        # Aplicar OneHotEncoder a las variables categóricas
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('scaler',StandardScaler(with_mean=True, with_std=True),numerical_features),
    ],
    # Si alguna columna no está en las listas, la eliminamos
    remainder='drop' 
)

# 3. Crear el Pipeline completo
# El Pipeline secuencia el preprocesamiento y el modelo.
def create_pipeline():
    pipeline = Pipeline([
        ("pre", preprocessor),
        ("sel", SelectKBest(score_func=f_classif, k=20)),
        ("pca", PCA(n_components=None)),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(50, 30, 40, 60),
            alpha=0.26,
            learning_rate_init=0.001,
            max_iter=15000,
            random_state=21
        ))
    ])
    return pipeline

model_pipeline = create_pipeline()

#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#

def optimize_hyperparameters(pipeline, x_train, y_train):
    param_grid = {}

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10, 
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=2,
    )

    grid_search.fit(x_train, y_train)

    # Mostrar los mejores resultados
    print("\n--- 🏆 Resultados de la Optimización ---")
    print(f"Mejor score (Precisión Balanceada): {grid_search.best_score_}")
    print(f"Mejores Hiperparámetros: {grid_search.best_params_}")
    return grid_search

# Guardar el pipeline optimizado
best_model_pipeline = optimize_hyperparameters(model_pipeline, x_train, y_train)

#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#

# Nombre del archivo de destino
MODEL_PATH = "files/models/model.pkl.gz"


model_dir = os.path.dirname(MODEL_PATH)
if model_dir and not os.path.exists(model_dir):
    os.makedirs(model_dir)

    print(f"Directorio creado: {model_dir}")
with gzip.open(MODEL_PATH, 'wb') as f: # 'wb' = write binary
    pickle.dump(best_model_pipeline, f) 

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
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#

def calculate_metrics(y_true, y_pred, dataset_name):
    """Calcula y formatea las métricas solicitadas."""
    return {
        'type': 'metrics',
        'dataset': dataset_name,        
        'precision': precision_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }

# --- 1. Hacer Predicciones ---
y_pred_train = best_model_pipeline.predict(x_train)
y_pred_test = best_model_pipeline.predict(x_test)

# --- 2. Calcular Métricas ---
metrics_train = calculate_metrics(y_train, y_pred_train, 'train')
metrics_test = calculate_metrics(y_test, y_pred_test, 'test')

results_list = [metrics_train, metrics_test]


# --- 3. Guardar las Métricas en formato JSON Lines ---
OUTPUT_PATH = "files/output/metrics.json" 

# Crear la carpeta de destino si no existe
output_dir = os.path.dirname(OUTPUT_PATH)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Usar un bucle para escribir cada objeto JSON en una línea separada.
with open(OUTPUT_PATH, 'w') as f:
    for item in results_list:
        f.write(json.dumps(item) + '\n')

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
# Nombre del archivo de destino
OUTPUT_PATH = "files/output/metrics.json"

# --- 1. Calcular las Matrices de Confusión ---
cm_train = confusion_matrix(y_train, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred_test)

# Matriz de Confusión tiene el formato:
# [[TN, FP],
#  [FN, TP]]

def format_confusion_matrix(cm, dataset_name):
    """Formatea la matriz de confusión de NumPy a la estructura de diccionario solicitada."""
    TN, FP, FN, TP = cm.ravel()
    
    return {
        'type': 'cm_matrix',
        'dataset': dataset_name,
        'true_0': {
            "predicted_0": int(TN),
            "predicted_1": int(FP)
        },
        'true_1': {
            "predicted_0": int(FN),
            "predicted_1": int(TP)
        }
    }

# --- 2. Formatear las matrices de confusión ---
cm_dict_train = format_confusion_matrix(cm_train, 'train')
cm_dict_test = format_confusion_matrix(cm_test, 'test')

# --- 3. Cargar métricas existentes y agregar las matrices ---
results_list = []
try:
    # Leer línea por línea para decodificar el formato JSON Lines (NDJSON)
    with open(OUTPUT_PATH, 'r') as f:
        for line in f:
            if line.strip(): # Ignorar líneas vacías
                results_list.append(json.loads(line))
    print(f"\nSe cargaron {len(results_list)} métricas existentes.")

except FileNotFoundError:
    print(f"\nAdvertencia: Archivo {OUTPUT_PATH} no encontrado. Se creará uno nuevo.")
except json.JSONDecodeError as e:
    print(f"\nError al decodificar una línea del JSON existente: {e}. Asegúrate de que el archivo esté en formato JSON Lines.")


# Agregar las nuevas matrices de confusión a la lista
results_list.append(cm_dict_train)
results_list.append(cm_dict_test)


# --- 4. Guardar el contenido actualizado en formato JSON Lines ---
# Crear la carpeta de destino si no existe (por si acaso)
output_dir = os.path.dirname(OUTPUT_PATH)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Escribir toda la lista actualizada en formato JSON Lines
with open(OUTPUT_PATH, 'w') as f:
    for item in results_list:
        # json.dumps convierte el diccionario a una cadena JSON
        # '\n' asegura que cada objeto esté en una línea diferente
        f.write(json.dumps(item) + '\n')

print(f"\n✅ Paso 7 completado. Matrices de confusión añadidas a {OUTPUT_PATH} (Formato JSON Lines).")