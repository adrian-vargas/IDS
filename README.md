
# IDS - Interpretable Decision Sets

IDS es una librería en Python diseñada para entrenar modelos de Interpretable Decision Sets (IDS), un tipo de modelo de inteligencia artificial que genera conjuntos de reglas interpretables. Estos modelos son útiles en aplicaciones donde la interpretabilidad es esencial, ya que permiten comprender las decisiones tomadas por el modelo a través de reglas simples y concisas.

Esta librería está basada en el repositorio [pyIDS](https://github.com/jirifilip/pyIDS) y en el estudio sobre IDS realizado por Lakkaraju et al. (2016), quienes propusieron este marco para construir modelos predictivos que optimizan tanto la precisión como la interpretabilidad. Adaptamos y extendimos ideas clave del repositorio original y del trabajo académico para desarrollar una implementación ajustada a los requisitos de nuestro proyecto y experimentación.

## Características

- **Generación de reglas interpretables** a partir de datos.
- **Optimización de reglas** basada en métricas como soporte, confianza y cobertura.
- **Balanceo de clases** en los datos mediante SMOTE.
- **Preprocesamiento de datos** con discretización y codificación One-Hot.
- **Integración con scikit-learn** para facilitar el uso en pipelines de aprendizaje automático.

## Instalación

Para instalar la librería, clona el repositorio y utiliza `pip` para instalar las dependencias:

```bash
git clone https://github.com/adrian-vargas/IDS.git
cd IDS
pip install -r requirements.txt
```

Asegúrate de tener Python 3.12.5 o superior.

### Dependencias

Las principales dependencias de la librería incluyen:

- `numpy`
- `pandas`
- `apyori`
- `scikit-learn`
- `imbalanced-learn` (para SMOTE)
- `pulp` (para resolver problemas de optimización)

## Estructura del Proyecto

La estructura del proyecto es la siguiente:

```
IDS/
│
├── ids/                       # Código principal de la librería
│   ├── datasets/              # Carga y preprocesamiento de datos
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── models/                # Definición de modelos y clases relacionadas
│   │   ├── __init__.py
│   │   └── rule.py
│   ├── __init__.py
│   ├── ids.py                 # Implementación del modelo IDS
│   └── utils.py               # Funciones auxiliares
├── venv/                      # Entorno virtual (opcional)
├── .gitignore
├── LICENSE
├── README.md                  # Documentación del proyecto
├── requirements.txt           # Dependencias del proyecto
└── setup.py                   # Archivo de configuración para la instalación
```

## Uso

### 1. Importación y Configuración del Modelo IDS

```python
from ids import IDSModel
from ids.datasets import load_and_preprocess_data, balance_data, split_data
```

### 2. Cargar y Preprocesar los Datos

```python
# Cargar y preprocesar el dataset
X, y = load_and_preprocess_data("ruta/dataset.csv", target_column="G3", target_threshold=10)
X_balanced, y_balanced = balance_data(X, y)
X_train, X_test, y_train, y_test = split_data(X_balanced, y_balanced, test_size=0.3)
```

### 3. Entrenar el Modelo IDS

```python
# Inicializar y entrenar el modelo IDS
model = IDSModel(lambda1=0.1, lambda2=0.1, lambda3=1.0, lambda4=1.0)
model.fit(X_train, y_train)
```

### 4. Realizar Predicciones

```python
# Realizar predicciones en el conjunto de prueba
predictions = model.predict(X_test)
```

### 5. Imprimir las Reglas Seleccionadas

```python
# Imprimir las reglas interpretables generadas por el modelo
model.print_rules(X_train, y_train, label_mapping={0: "Reprobado", 1: "Aprobado"})
```

## Parámetros

Al inicializar el modelo `IDSModel`, puedes configurar los siguientes parámetros:

- `lambda1`, `lambda2`, `lambda3`, `lambda4`: Pesos de regularización en la función objetivo del modelo IDS.
- `min_support`: Soporte mínimo requerido para una regla.
- `min_confidence`: Confianza mínima requerida para una regla.
- `max_rule_length`: Longitud máxima permitida para las reglas generadas.

## Funciones

### IDSModel

- `fit(X, y)`: Entrena el modelo IDS utilizando los datos de entrenamiento.
- `predict(X)`: Realiza predicciones para los datos de entrada utilizando las reglas seleccionadas.
- `print_rules(X_train=None, y_train=None, label_mapping=None)`: Imprime las reglas interpretables seleccionadas por el modelo. Si se proporcionan datos de entrenamiento, muestra también la precisión y el número de muestras cubiertas por cada regla.

### Preprocesamiento de Datos

- `load_and_preprocess_data(file_path, target_column, sep=';', target_threshold=None)`: Carga y preprocesa los datos, convirtiendo variables numéricas en clases binarias si se proporciona un umbral.
- `balance_data(X, y)`: Balancea las clases en los datos utilizando SMOTE.
- `split_data(X, y, test_size=0.2, random_state=42)`: Divide los datos en conjuntos de entrenamiento y prueba.

## Licencia

Este proyecto está licenciado bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.
