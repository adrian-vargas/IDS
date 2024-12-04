# ids/metrics.py

import pandas as pd
import numpy as np

# Función para calcular las métricas para un modelo IDS
def calculate_ids_interpretability_metrics(ids_model, X_train, y_train):
    """
    Calcula métricas de interpretabilidad para un modelo IDS, como precisión, parsimonia, cobertura, Gini y sparsidad.

    Args:
        ids_model (IDSModel): Modelo IDS entrenado.
        X_train (DataFrame): Conjunto de características de entrenamiento.
        y_train (Series): Etiquetas de entrenamiento.

    Returns:
        metrics_df (DataFrame): DataFrame con las métricas de las reglas.
        summary (dict): Diccionario con la profundidad total (máxima longitud de las reglas) y el número de reglas.
    """
    rules = []
    num_rules = len(ids_model.selected_rules)

    for rule in ids_model.selected_rules:
        conditions = rule.conditions
        outcome = rule.class_label
        sparsity = len(conditions)  # La longitud de la regla se considera su "profundidad"
        parsimony = 1 / (sparsity + 1)  # A menor longitud de la regla, mayor parsimonia

        # Muestras cubiertas por la regla
        covered_samples = [i for i, row in X_train.iterrows() if rule.covers(row)]
        num_samples = len(covered_samples)

        # Cobertura
        coverage = num_samples / len(X_train) if len(X_train) > 0 else 0

        # Precisión
        if num_samples > 0:
            y_covered = y_train.iloc[covered_samples]
            correct_samples = (y_covered == outcome).sum()
            precision = correct_samples / num_samples

            # Índice de Gini
            class_counts = y_covered.value_counts(normalize=True)
            gini = 1 - np.sum(class_counts ** 2)
        else:
            precision = 0
            gini = 0

        # Agregar las métricas para la regla
        rules.append({
            'Precision': precision,
            'Parsimony': parsimony,
            'Coverage': coverage,
            'Gini': gini,
            'Sparsity': sparsity  # La longitud de la regla, o el número de condiciones
        })

    # Crear DataFrame con las métricas
    metrics_df = pd.DataFrame(rules)

    # La profundidad total en IDS se considera como la longitud máxima de las reglas
    max_rule_length = max([len(rule.conditions) for rule in ids_model.selected_rules])

    return metrics_df, {'Total Depth': max_rule_length, 'Number of Rules': num_rules}

# Función para calcular correct-cover e incorrect-cover en IDS
def calculate_correct_incorrect_cover(ids_model, X_test, y_test):
    """
    Calcula correct-cover(r) e incorrect-cover(r) para cada regla en el modelo IDS.

    Args:
        ids_model (IDSModel): Modelo IDS entrenado.
        X_test (DataFrame): Conjunto de datos de prueba.
        y_test (Series): Etiquetas verdaderas.

    Returns:
        correct_cover (list): Lista con el número de instancias correctamente cubiertas por cada regla.
        incorrect_cover (list): Lista con el número de instancias incorrectamente cubiertas por cada regla.
    """
    correct_cover = []
    incorrect_cover = []

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    for idx, rule in enumerate(ids_model.selected_rules):
        covered_samples = [i for i, row in X_test.iterrows() if rule.covers(row)]

        correct = sum([1 for i in covered_samples if i < len(y_test) and y_test.iloc[i] == rule.class_label])
        incorrect = len(covered_samples) - correct

        correct_cover.append(correct)
        incorrect_cover.append(incorrect)

        # Imprimir los resultados parciales por cada regla
        print(f"Rule {idx + 1}:")
        print(f"  Correctly covered samples: {correct}")
        print(f"  Incorrectly covered samples: {incorrect}")
        print("-" * 40)

    return correct_cover, incorrect_cover

# Función para calcular las propiedades de size, length, cover y overlap
def calculate_rule_properties(model, X_train, model_type="tree"):
    """
    Calcula las propiedades de size, length, cover y overlap para el modelo dado.

    Args:
        model: Modelo entrenado (puede ser un árbol de decisión o un modelo basado en reglas).
        X_train (DataFrame): Conjunto de entrenamiento.
        model_type (str): Tipo de modelo ('tree' para árboles de decisión, 'rules' para IDS).

    Returns:
        dict: Diccionario con las propiedades calculadas.
    """
    if model_type == "rules":
        # Para IDS o modelos basados en reglas
        rules = model.selected_rules
        size = len(rules)  # Número total de reglas

        # Longitud promedio de las reglas
        lengths = [len(rule.conditions) for rule in rules]
        avg_length = sum(lengths) / len(lengths) if lengths else 0

        # Cálculo del cover: Número de muestras cubiertas por las reglas
        covered_samples = set()
        for rule in rules:
            for i, row in X_train.iterrows():
                if rule.covers(row):
                    covered_samples.add(i)
        cover = len(covered_samples)

        # Cálculo del overlap: Muestras cubiertas por más de una regla
        overlap_count = 0
        for i, row in X_train.iterrows():
            covering_rules = sum([rule.covers(row) for rule in rules])
            if covering_rules > 1:
                overlap_count += 1

        return {
            'size': size,
            'avg_length': avg_length,
            'cover': cover,
            'overlap': overlap_count
        }

    elif model_type == "tree":
        # Para árboles de decisión
        tree_ = model.tree_
        size = model.get_n_leaves()  # Número de reglas (hojas)

        # Longitud promedio: Número promedio de nodos (profundidad) desde la raíz hasta las hojas
        depths = []
        def recurse(node, depth):
            if tree_.children_left[node] == -1:  # Hoja
                depths.append(depth)
            else:
                recurse(tree_.children_left[node], depth + 1)
                recurse(tree_.children_right[node], depth + 1)

        recurse(0, 0)
        avg_length = sum(depths) / len(depths) if depths else 0

        # Cover: Número de muestras cubiertas por el árbol completo (todas las hojas)
        cover = X_train.shape[0]

        # Overlap en un árbol es cero ya que las reglas no se solapan en un árbol de decisión puro
        overlap = 0

        return {
            'size': size,
            'avg_length': avg_length,
            'cover': cover,
            'overlap': overlap
        }
    
def calculate_ids_probabilities(ids_model, X_test, y_test, target_class):
    """
    Calcula las probabilidades de predicción para un conjunto de pruebas basado en el modelo IDS,
    enfocándose en la clase objetivo especificada.

    Args:
        ids_model (IDSModel): Modelo IDS entrenado.
        X_test (DataFrame): Conjunto de datos de prueba.
        y_test (Series): Etiquetas verdaderas para calcular la precisión de las reglas.
        target_class (int): Clase objetivo para la cual calcular las probabilidades.

    Returns:
        prob_ids (list): Lista de probabilidades estimadas para la clase objetivo.
    """
    prob_ids = []

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    for i, row in X_test.iterrows():
        # Obtener reglas aplicables a la fila actual
        applicable_rules = [rule for rule in ids_model.selected_rules if rule.covers(row)]

        if applicable_rules:
            rule_precisions = []
            for rule in applicable_rules:
                covered_samples = [idx for idx, train_row in X_test.iterrows() if rule.covers(train_row)]
                if covered_samples:
                    if all(idx < len(y_test) for idx in covered_samples):
                        correct_samples = sum([1 for idx in covered_samples if y_test.iloc[idx] == rule.class_label])
                        precision = correct_samples / len(covered_samples)

                        # Considerar solo reglas que predicen la clase objetivo
                        if rule.class_label == target_class:
                            rule_precisions.append(precision)

            if rule_precisions:
                prob_ids.append(max(rule_precisions))
            else:
                prob_ids.append(0.0)  # Si no hay precisión calculada para la clase objetivo, asignar 0
        else:
            prob_ids.append(0.0)  # Si no hay reglas aplicables, asignar 0 para la clase objetivo

    return prob_ids
