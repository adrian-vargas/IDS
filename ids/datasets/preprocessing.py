# ids/datasets/preprocessing.py

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(file_path, target_column):
    """
    Carga y preprocesa el conjunto de datos desde el archivo especificado.
    """
    # Cargar el dataset con el separador adecuado
    df = pd.read_csv(file_path, sep=';')

    # Crear la columna 'Success' basada en 'G3'
    df['Success'] = (df['G3'] > 10).astype(int)

    # Eliminar las columnas 'G1', 'G2' y 'G3' si existen
    df.drop(columns=['G1', 'G2', 'G3'], errors='ignore', inplace=True)

    # Separar características y variable objetivo
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Identificar columnas categóricas y numéricas
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Discretizar variables numéricas
    if numerical_cols:
        discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
        X[numerical_cols] = discretizer.fit_transform(X[numerical_cols])

    # Aplicar One-Hot Encoding a variables categóricas
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols)

    # Convertir todos los datos a tipo 'float' para evitar problemas
    X = X.astype(float)

    return X, y

def balance_data(X, y):
    """
    Balancea las clases en los datos utilizando SMOTE.
    """
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
