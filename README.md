
# IDS - Interpretable Decision Sets

IDS es una librería en Python diseñada para entrenar modelos de Interpretable Decision Sets (IDS), un marco de modelos de inteligencia artificial transparente que genera conjuntos de reglas fácilmente interpretables. Estos modelos son ideales para aplicaciones donde la transparencia y la interpretabilidad.

Este proyecto es parte del Trabajo de Fin de Máster (TFM) *A Tool for Human Evaluation of Interpretability* realizado por Adrián Vargas en la Universidad Politécnica de Madrid. La librería IDS implementa y extiende ideas clave del estudio de [Lakkaraju et al. (2016)](https://cs.stanford.edu/people/jure/pubs/interpretable-kdd16.pdf) y el repositorio original [pyIDS](https://github.com/jirifilip/pyIDS), optimizando tanto la precisión como la interpretabilidad.

## Características Técnicas

- **Generación de reglas interpretables**: Utiliza Apriori para crear reglas basadas en soporte y confianza.
- **Optimización mediante programación lineal**: Garantiza un equilibrio entre precisión y simplicidad de las reglas.
- **Análisis de interpretabilidad**: Calcula métricas como sparsity, coverage, gini y parsimony.
- **Visualización avanzada**: Incluye gráficos globales y explicaciones locales mediante Graphviz y Matplotlib.
- **Balanceo de datos**: Compatibilidad con SMOTE para mejorar la distribución de clases.

## Estructura del Proyecto

```
IDS/
│
├── ids/                       # Código principal de la librería
│   ├── models/                # Definición de modelos y clases relacionadas
│   │   ├── __init__.py
│   │   └── rule.py            # Implementación de la clase Rule
│   ├── __init__.py
│   ├── ids.py                 # Implementación del modelo IDS
│   ├── metrics.py             # Cálculo de métricas de interpretabilidad
│   └── utils.py               # Funciones auxiliares para generación y visualización de reglas
├── venv/                      # Entorno virtual (opcional)
├── .gitignore
├── LICENSE
├── README.md                  # Documentación del proyecto
├── requirements.txt           # Dependencias del proyecto
└── setup.py                   # Archivo de configuración para la instalación
```

## Instalación

Para instalar la librería, clona el repositorio y utiliza `pip` para instalar las dependencias:

```bash
git clone https://github.com/adrian-vargas/IDS.git
cd IDS
pip install -r requirements.txt
```

Asegúrate de tener Python 3.12.5 o superior.

## Ejemplo de Uso

### 1. Importación y Configuración del Modelo IDS

```python
from ids import IDSModel
from ids.utils import generate_candidate_rules, print_and_save_rules
```

### 2. Entrenamiento del Modelo IDS

```python
# Inicializar y entrenar el modelo IDS
ids_model = IDSModel(lambda1=0.1, lambda2=0.1, lambda3=1.0, lambda4=1.0, min_support=0.05, min_confidence=0.6, max_rule_length=3)
ids_model.fit(X_train, y_train)
```

### 3. Visualización de Reglas

```python
# Imprimir y guardar las reglas seleccionadas
rules_df = print_and_save_rules(ids_model, X_train, y_train, output_file="ids_rules.csv")

# Visualizar las reglas globalmente
from ids.utils import visualize_ids_rules
visualize_ids_rules(rules_df)
```

## Funcionalidades Clave

- **`IDSModel`**: Entrenamiento y predicción con reglas interpretables.
- **Métricas de Interpretabilidad**:
  - `calculate_ids_interpretability_metrics`: Analiza precisión, sparsity, parsimony y cobertura.
  - `calculate_correct_incorrect_cover`: Evalúa qué tan bien cubren las reglas los datos.
- **Visualización**:
  - `visualize_ids_rules`: Genera diagramas globales y locales para las reglas seleccionadas.

## Contribuciones

La librería **IDS** es una alternativa adecuada para proyectos de inteligencia artificial explicable (XAI):

1. **Entornos sensibles a la interpretabilidad**  
   Diseñada para casos como educación o finanzas, donde se requiere que los usuarios comprendan y confíen en las decisiones del modelo. Ejemplo: clasificar el rendimiento académico.

2. **Proyectos de investigación en XAI**  
   IDS es una alternativa menos compleja y más eficiente que **Smooth Local Search (SLS)**, gracias a:
   - **Programación lineal optimizada**: Selección de reglas mediante el solver PuLP para evitar redundancias y simplificar la búsqueda.
   - **Menor consumo computacional**: Ideal para sistemas con recursos limitados o datasets moderados.
   - **Flexibilidad**: Configuraciones avanzadas para soporte, confianza y longitud máxima de reglas.

3. **Visualización integrada y explicaciones claras**  
   Con herramientas como Graphviz y Matplotlib, IDS permite generar explicaciones locales y globales de manera intuitiva, facilitando el análisis de reglas.

4. **Integración sencilla con `scikit-learn`**  
   Su arquitectura modular la hace fácilmente integrable en pipelines de aprendizaje automático existentes, permitiendo personalizaciones según el contexto del proyecto.

## Licencia

Este proyecto está licenciado bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.
