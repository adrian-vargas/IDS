# ids/utils.py

import numpy as np
from apyori import apriori
from .models.rule import Rule

def generate_candidate_rules(df, min_support=0.05, min_confidence=0.6, max_length=3):
    """
    Genera reglas candidatas utilizando el algoritmo Apriori.
    """
    # Obtener los nombres de las características excluyendo el objetivo
    feature_names = df.columns.tolist()
    feature_names.remove('target')

    # Preparar las transacciones
    transactions = []
    for i in range(df.shape[0]):
        transaction = []
        for feature in feature_names:
            transaction.append(f"{feature}={df.iloc[i][feature]}")
        transaction.append(f"target={df.iloc[i]['target']}")
        transactions.append(transaction)

    # Ejecutar el algoritmo Apriori
    results = list(apriori(transactions, min_support=min_support, min_confidence=min_confidence, max_length=max_length))

    # Generar reglas candidatas
    candidate_rules = []
    for result in results:
        for ordered_stat in result.ordered_statistics:
            if len(ordered_stat.items_base) > 0 and 'target=' in list(ordered_stat.items_add)[0]:
                conditions = []
                for item in ordered_stat.items_base:
                    feature_value = item.split('=')
                    if len(feature_value) == 2:
                        feature = feature_value[0]
                        value = feature_value[1]
                        conditions.append((feature, value))
                class_label = int(list(ordered_stat.items_add)[0].split('=')[1])
                rule = Rule(conditions, class_label)
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
        cover = df.apply(lambda x: rule.covers(x), axis=1)
        correct = df['target'].astype(int) == rule.class_label
        correct_cover = cover & correct
        rule_covers[idx] = cover.astype(int)
        rule_correct_covers[idx] = correct_cover.astype(int)
        rule_lengths[idx] = len(rule)
        rule_errors[idx] = (cover & (~correct)).astype(int)
    return rule_covers, rule_correct_covers, rule_lengths, rule_errors
