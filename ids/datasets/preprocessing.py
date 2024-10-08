import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(file_path, target_column, sep=';', target_threshold=None):
    """
    Carga y preprocesa el conjunto de datos desde el archivo especificado.
    
    Args:
    - file_path: Ruta del archivo de datos.
    - target_column: Nombre de la columna objetivo.
    - sep: Separador utilizado en el archivo CSV (por defecto, ';').
    - target_threshold: Umbral opcional para convertir variables numéricas en clases binarias. 
      (Si se proporciona, se aplicará a la columna objetivo si es numérica).
    
    Returns:
    - X: Características preprocesadas.
    - y: Variable objetivo preprocesada.
    """
    # Cargar el dataset con el separador adecuado
    df = pd.read_csv(file_path, sep=sep)

    # Si se proporciona un umbral para la columna objetivo, convertirla en clases binarias
    if target_threshold is not None and pd.api.types.is_numeric_dtype(df[target_column]):
        df[target_column] = (df[target_column] > target_threshold).astype(int)

    # Separar características y variable objetivo
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Identificar columnas categóricas y numéricas
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Discretizar variables numéricas (si existen)
    if numerical_cols:
        discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
        X[numerical_cols] = discretizer.fit_transform(X[numerical_cols])

    # Aplicar One-Hot Encoding a variables categóricas (si existen)
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols)

    # Convertir todos los datos a tipo 'float' para evitar problemas
    X = X.astype(float)

    return X, y

def balance_data(X, y):
    """
    Balancea las clases en los datos utilizando SMOTE.
    
    Args:
    - X: Características.
    - y: Variable objetivo.
    
    Returns:
    - X_balanced: Características balanceadas.
    - y_balanced: Variable objetivo balanceada.
    """
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Args:
    - X: Características.
    - y: Variable objetivo.
    - test_size: Tamaño del conjunto de prueba (por defecto, 0.2).
    - random_state: Semilla aleatoria para reproducibilidad (por defecto, 42).
    
    Returns:
    - X_train, X_test, y_train, y_test: Conjuntos de entrenamiento y prueba divididos.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
