# ids/datasets/preprocessing.py

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(file_path, target_column):
    """
    Carga y preprocesa el conjunto de datos desde el archivo especificado.
    """
    df = pd.read_csv(file_path)
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Identificar columnas categóricas y numéricas
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Discretizar variables numéricas
    discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    X[numerical_cols] = discretizer.fit_transform(X[numerical_cols])

    # Aplicar One-Hot Encoding a variables categóricas
    X_encoded = pd.get_dummies(X, columns=categorical_cols)

    # Convertir todos los datos a tipo 'float' para evitar problemas
    X_encoded = X_encoded.astype(float)

    return X_encoded, y

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
