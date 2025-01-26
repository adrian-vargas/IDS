# ids/utils.py

import numpy as np
from apyori import apriori
import pandas as pd
from graphviz import Digraph
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from .models.rule import Rule

def generate_candidate_rules(df, min_support=0.05, min_confidence=0.6, max_length=3, boolean_cols=None):
    """
    Genera reglas candidatas utilizando el algoritmo Apriori.
    """
    if boolean_cols is None:
        boolean_cols = []

    # Preparar las transacciones
    transactions = []
    for i in range(df.shape[0]):
        transaction = []
        for feature in df.columns:
            value = df.iloc[i][feature]
            # Usar "=" para booleanos y desigualdades para otros
            if feature in boolean_cols:
                transaction.append(f"{feature}={int(value)}")
            else:
                transaction.append(f"{feature}={value}")
        transactions.append(transaction)

    # Ejecutar Apriori
    results = list(apriori(transactions, min_support=min_support, min_confidence=min_confidence, max_length=max_length))
    candidate_rules = []
    for result in results:
        for ordered_stat in result.ordered_statistics:
            if len(ordered_stat.items_base) > 0 and 'target=' in list(ordered_stat.items_add)[0]:
                conditions = []
                for item in ordered_stat.items_base:
                    feature, value = item.split('=')
                    value = int(value) if feature in boolean_cols else float(value)
                    conditions.append((feature, value))
                class_label = int(list(ordered_stat.items_add)[0].split('=')[1])
                rule = Rule(conditions, class_label, boolean_cols=boolean_cols)
                candidate_rules.append(rule)
    return candidate_rules

def calculate_rule_metrics(rules, df):
    """
    Calcula métricas como cobertura y precisión para cada regla.
    """
    rule_covers = {}
    rule_correct_covers = {}
    rule_lengths = {}
    rule_errors = {}
    num_samples = df.shape[0]

    for idx, rule in enumerate(rules):
        # Calcular la cobertura de la regla sin necesidad de 'boolean_cols' como parámetro
        cover = df.apply(lambda x: rule.covers(x), axis=1)
        correct = df['target'].astype(int) == rule.class_label
        correct_cover = cover & correct
        rule_covers[idx] = cover.astype(int)
        rule_correct_covers[idx] = correct_cover.astype(int)
        rule_lengths[idx] = len(rule)
        rule_errors[idx] = (cover & (~correct)).astype(int)

    return rule_covers, rule_correct_covers, rule_lengths, rule_errors

def print_and_save_rules(ids_model, X_train, y_train, label_mapping=None, output_file='ids_rules.csv'):
    """
    Extrae y guarda las reglas del modelo IDS en un archivo CSV, junto con métricas de las reglas e índices de las muestras cubiertas.

    Parámetros:
    - ids_model: El modelo IDS del cual se extraen las reglas.
    - X_train: Conjunto de características de entrenamiento.
    - y_train: Etiquetas de entrenamiento.
    - label_mapping: Mapeo de las etiquetas (ej: {0: 'Reprobado', 1: 'Aprobado'}).
    - output_file: Nombre del archivo CSV donde se guardarán las reglas.
    """
    if label_mapping is None:
        label_mapping = {0: 'Reprobado', 1: 'Aprobado'}

    rules_data = []
    for rule in ids_model.selected_rules:
        outcome = rule.class_label
        covered_samples = [i for i, row in X_train.iterrows() if rule.covers(row)]
        num_samples = len(covered_samples)
        correct_samples = sum([1 for i in covered_samples if y_train.iloc[i] == outcome])
        precision = correct_samples / num_samples if num_samples > 0 else 0
        sparsity = len(rule.conditions)
        parsimony = 1 / (sparsity + 1)
        coverage = num_samples / len(X_train)
        gini = 1 - (precision ** 2 + (1 - precision) ** 2)

        # Utilizar el método __repr__ de Rule para obtener la regla formateada
        formatted_rule = repr(rule)
        formatted_prediction = label_mapping.get(outcome, outcome)

        # Agregar los datos de la regla
        rules_data.append({
            'rule': formatted_rule,
            'prediction': formatted_prediction,
            'precision': precision,
            'parsimony': parsimony,
            'coverage': coverage,
            'gini': gini,
            'sparsity': sparsity,
            'samples': num_samples,
            'covered_indices': covered_samples  # Guardar los índices de las muestras cubiertas
        })

    # Guardar las reglas en un archivo CSV
    rules_df = pd.DataFrame(rules_data)
    rules_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Reglas guardadas en '{output_file}'")
    return rules_df

def visualize_ids_rules(rules_df, rule_col='rule', prediction_col='prediction', labels_map=None):
    """
    Visualiza las reglas del modelo IDS usando graphviz y matplotlib, con las palabras "si" y "entonces"
    incluidas en la tabla de reglas, así como la etiqueta de la clase correspondiente.

    Parámetros:
    - rules_df: DataFrame que contiene las reglas y sus respectivas predicciones.
    - rule_col: Nombre de la columna que contiene las reglas.
    - prediction_col: Nombre de la columna que contiene las predicciones.
    - labels_map: Mapeo opcional de etiquetas para usar colores predefinidos.
    """
    if rule_col not in rules_df.columns or prediction_col not in rules_df.columns:
        raise KeyError(f"Las columnas '{rule_col}' o '{prediction_col}' no están presentes en el DataFrame.")

    # Crear un nuevo grafo dirigido
    dot = Digraph(comment='Interpretable Decision Sets (IDS)', graph_attr={'size': '10,10'})

    # Extraer las reglas y predicciones del dataframe
    rules = rules_df[rule_col].tolist()
    predictions = rules_df[prediction_col].tolist()

    # Agregar nodos y reglas del conjunto IDS con colores
    for idx, rule in enumerate(rules, start=1):
        dot.node(str(idx), str(idx), style='filled', fillcolor='lightblue', color='black')

    # Definir colores predefinidos para la clasificación binaria
    if labels_map is None:
        labels_map = {
            'Aprobado': {'id': 'A', 'color': 'lightgreen'},
            'Reprobado': {'id': 'B', 'color': 'lightcoral'}
        }

    # Nodos de predicciones con colores específicos
    for label, info in labels_map.items():
        dot.node(info['id'], info['id'], shape='box', style='filled', fillcolor=info['color'], color='black')

    # Conectar nodos según las reglas
    for idx, prediction in enumerate(predictions, start=1):
        dot.edge(str(idx), labels_map[prediction]['id'])

    # Renderizar el gráfico en memoria
    dot.format = 'png'
    dot_data = dot.pipe()

    # Mostrar el grafo utilizando matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].axis('off')
    axs[1].axis('off')

    # Leer la imagen directamente desde la memoria
    graph_img = Image.open(BytesIO(dot_data))

    # Ajustar el formato de las reglas
    definitions = [
        [
            str(idx + 1),
            f"{rule.split('entonces')[0].strip()} entonces {prediction}"
        ]
        for idx, (rule, prediction) in enumerate(zip(rules, predictions))
    ]

    # Añadir las clases "A" y "B" con su significado
    definitions += [[info['id'], label] for label, info in labels_map.items()]

    # Crear la tabla de referencia de variables
    table_ax = axs[1]
    table_ax.axis('off')
    table = table_ax.table(cellText=definitions, colLabels=['ID', 'Definición'], loc='center', cellLoc='left', colWidths=[0.1, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1])

    # Mostrar la imagen del grafo y la tabla
    axs[0].imshow(graph_img)

    plt.tight_layout()
    plt.show()

def generate_ids_global_graph(ids_model):
    """
    Genera y muestra un grafo global para el modelo IDS utilizando las reglas extraídas.

    Parámetros:
    - ids_model: El modelo IDS entrenado.

    Retorna:
    - None. Muestra el grafo directamente en consola.
    """
    # Extraer las reglas del modelo IDS
    rules_df = print_and_save_rules(ids_model, X_train_smote, y_train_smote)

    # Crear un nuevo grafo dirigido con Graphviz
    dot = Digraph(comment='Interpretable Decision Sets (IDS)', format='png')

    # Crear nodos para cada regla con color azul
    for index, row in rules_df.iterrows():
        rule_id = f"Regla {index + 1}"
        rule_text = row['Regla']
        dot.node(rule_id, rule_text, shape='circle', style='filled', fillcolor='lightblue')

    # Nodos para los resultados (A: Aprobado, B: Reprobado) con colores verde y rojo
    dot.node('A', 'Aprobado', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('B', 'Reprobado', shape='box', style='filled', fillcolor='lightcoral')

    # Conectar cada regla con su predicción de clase
    for index, row in rules_df.iterrows():
        rule_id = f"Regla {index + 1}"
        prediction = row['Predicción']
        if prediction == 'Aprobado':
            dot.edge(rule_id, 'A')
        else:
            dot.edge(rule_id, 'B')

    # Renderizar y mostrar el grafo en consola
    display(Image(dot.pipe(format='png')))

def explain_local_ids(model, rules_df, test_features, rule_col='rule', prediction_col='prediction', labels_map=None, default_class='Reprobado', highlight_predicted_in_table=False):
    """
    Genera una explicación local para una observación específica en el modelo IDS,
    resaltando las reglas que cubren esta observación.
    """

    # Convertir las características de prueba a un DataFrame de una sola fila
    specific_observation = pd.DataFrame([test_features])

    # Obtener la clase predicha para la observación específica usando el modelo IDS
    predicted_class = model.predict(specific_observation)[0]

    # Convertir la clase predicha a string
    predicted_class_str = 'Aprobado' if predicted_class == 1 else 'Reprobado'

    # Mostrar la observación específica
    print(f"Observación específica:\n{specific_observation.iloc[0]}")

    # Mostrar las reglas seleccionadas por el modelo
    print("Reglas seleccionadas por el modelo:")
    for idx, rule in enumerate(model.selected_rules):
        print(f"Regla {idx + 1}: {rule}")
    
    # Identificar las reglas activas manualmente
    active_rules = []
    for idx, rule in enumerate(model.selected_rules):
        if rule.covers(specific_observation.iloc[0]):
            active_rules.append(idx)

    # Mostrar cuáles reglas fueron activas
    print(f"Reglas activas: {[idx + 1 for idx in active_rules]}")

    # Crear el grafo
    dot = Digraph(comment='IDS - Local Explanation', graph_attr={'size': '10,10'})

    # Extraer las reglas y predicciones del DataFrame
    rules = rules_df[rule_col].tolist()
    predictions = rules_df[prediction_col].tolist()

    # Mapeo de etiquetas si no se ha proporcionado
    if labels_map is None:
        labels_map = {
            'Aprobado': {'id': 'A', 'color': 'lightgreen'},
            'Reprobado': {'id': 'B', 'color': 'lightcoral'}
        }

    # Agregar nodos de reglas con doble círculo si están activas
    for idx, rule in enumerate(rules, start=1):
        color = "yellow" if idx - 1 in active_rules else "lightblue"
        shape = "doublecircle" if idx - 1 in active_rules else "circle"
        dot.node(str(idx), f"{idx}", shape=shape, style="filled", fillcolor=color)

    # Imprimir la clase predicha en la consola para depuración
    print(f"Clase predicha según IDS: {predicted_class_str}")

    # Nodos de predicción final ("Aprobado" y "Reprobado")
    for label, info in labels_map.items():
        fillcolor = "yellow" if label == predicted_class_str else info['color']
        dot.node(info['id'], info['id'], shape='box', style="filled", fillcolor=fillcolor)

    # Conectar todas las reglas con su predicción correspondiente
    for idx in range(len(rules)):
        predicted_class_for_rule = predictions[idx]
        if idx in active_rules:
            dot.edge(str(idx + 1), labels_map[predicted_class_for_rule]['id'])
        else:
            dot.edge(str(idx + 1), labels_map[predicted_class_for_rule]['id'], style="dashed")  

    # Renderizar el gráfico en memoria
    dot.format = "png"
    dot_data = dot.pipe()

    # Mostrar el grafo utilizando matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].axis("off")
    axs[1].axis("off")

    # Leer la imagen directamente desde la memoria
    graph_img = Image.open(BytesIO(dot_data))

    # Crear la tabla de referencia de variables con la clase literal en la definición
    definitions = [
        [
            f"{idx+1}",
            f"{rule.split('entonces')[0].strip()} entonces {prediction}"
        ]
        for idx, (rule, prediction) in enumerate(zip(rules, predictions))
    ]

    # Agregar las etiquetas de clase "A" para Aprobado y "B" para Reprobado en la tabla de referencia
    definitions += [[info['id'], label] for label, info in labels_map.items()]

    table_ax = axs[1]
    table_ax.axis("off")
    table = table_ax.table(cellText=definitions, colLabels=["ID", "Definición"], loc="center", cellLoc="left", colWidths=[0.15, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1])

    # Resaltar las filas correspondientes a las reglas activas
    for active_rule_idx in active_rules:
        table[(active_rule_idx + 1, 0)].set_facecolor('yellow')  # Resaltar ID
        table[(active_rule_idx + 1, 1)].set_facecolor('yellow')  # Resaltar Definición

    # Resaltar la clase predicha en la tabla si está habilitado
    if highlight_predicted_in_table:
        # Iterar sobre las celdas de la tabla para encontrar la clase predicha y resaltarla
        for key, cell in table.get_celld().items():
            if cell.get_text().get_text() == predicted_class_str and key[0] > len(rules):
                cell.set_facecolor('yellow')

                # Resaltar la celda a la izquierda (ID)
                if key[1] == 1:  # Verificar que es la columna de "Definición"
                    table[(key[0], 0)].set_facecolor('yellow')  # Resaltar la celda de la columna "ID"

    # Mostrar la imagen del grafo y la tabla
    axs[0].imshow(graph_img)
    plt.tight_layout()
    plt.show()

def explain_global_ids(model, rules_df, test_features=None, labels_map=None, highlight_predicted_in_table=False):
    """
    Genera una explicación global para el modelo IDS, opcionalmente resaltando reglas activas y la clase predicha solo en la tabla.

    Parámetros:
    - model: Modelo IDS entrenado.
    - rules_df: DataFrame que contiene las reglas y sus predicciones.
    - test_features: Diccionario con las características de la observación específica (opcional para resaltar reglas activas).
    - labels_map: Mapeo opcional de etiquetas para usar colores específicos.
    - highlight_predicted_in_table: Booleano para resaltar la clase predicha en la tabla de definiciones.
    """
    # Crear el grafo
    dot = Digraph(comment='IDS - Global Explanation', graph_attr={'size': '10,10'})

    # Extraer las reglas y predicciones del DataFrame
    rules = rules_df['rule'].tolist()
    predictions = rules_df['prediction'].tolist()

    # Mapeo de etiquetas si no se ha proporcionado
    if labels_map is None:
        labels_map = {
            'Aprobado': {'id': 'A', 'color': 'lightgreen'},
            'Reprobado': {'id': 'B', 'color': 'lightcoral'}
        }

    # Si se proporcionan características de prueba, identificar las reglas activas y obtener la predicción
    active_rules = []
    predicted_class_str = None  # Clase predicha en formato string
    if test_features is not None:
        specific_observation = pd.DataFrame([test_features])
        active_rules = [idx for idx, rule in enumerate(model.selected_rules) if rule.covers(specific_observation.iloc[0])]

        # Obtener la clase predicha usando el modelo IDS
        predicted_class = model.predict(specific_observation)[0]
        predicted_class_str = 'Aprobado' if predicted_class == 1 else 'Reprobado'

    # Agregar nodos de reglas
    for idx, rule in enumerate(rules, start=1):
        color = "lightblue"  
        shape = "circle"  
        dot.node(str(idx), str(idx), shape=shape, style='filled', fillcolor=color)

    # Nodos de predicción final ("Aprobado" y "Reprobado")
    for label, info in labels_map.items():
        fillcolor = info['color']  
        dot.node(info['id'], info['id'], shape='box', style='filled', fillcolor=fillcolor)

    # Conectar todas las reglas con su predicción correspondiente
    for idx in range(len(rules)):
        predicted_class_for_rule = predictions[idx]
        dot.edge(str(idx + 1), labels_map[predicted_class_for_rule]['id'])

    # Renderizar el gráfico en memoria
    dot.format = "png"
    dot_data = dot.pipe()

    # Mostrar el grafo utilizando matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].axis("off")
    axs[1].axis("off")

    # Leer la imagen directamente desde la memoria
    graph_img = Image.open(BytesIO(dot_data))

    definitions = [
        [
            f"{idx+1}",
            f"{rule.split('entonces')[0].strip()} entonces {prediction}"
        ]
        for idx, (rule, prediction) in enumerate(zip(rules, predictions))
    ]

    # Agregar las etiquetas de clase "A" para Aprobado y "B" para Reprobado en la tabla de referencia
    definitions += [[info['id'], label] for label, info in labels_map.items()]

    table_ax = axs[1]
    table_ax.axis("off")
    table = table_ax.table(cellText=definitions, colLabels=["ID", "Definición"], loc="center", cellLoc="left", colWidths=[0.15, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1])

    # Resaltar las filas correspondientes a las reglas activas si están definidas
    if active_rules:
        for active_rule_idx in active_rules:
            table[(active_rule_idx + 1, 0)].set_facecolor('yellow')  # Resaltar ID
            table[(active_rule_idx + 1, 1)].set_facecolor('yellow')  # Resaltar Definición

    # Resaltar la clase predicha en la tabla si está habilitado
    if highlight_predicted_in_table and predicted_class_str is not None:
        # Resaltar la clase predicha en la tabla
        for key, cell in table.get_celld().items():
            if cell.get_text().get_text() == predicted_class_str and key[0] > len(rules):
                cell.set_facecolor('yellow')
                if key[1] == 1:  # Verificar que es la columna de "Definición"
                    table[(key[0], 0)].set_facecolor('yellow')  # Resaltar la celda de la columna "ID"

    axs[0].imshow(graph_img)
    plt.tight_layout()
    plt.show()
