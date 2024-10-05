# ids/ids.py

import numpy as np
import pandas as pd
from .utils import generate_candidate_rules, calculate_rule_metrics
from .models.rule import Rule
from pulp import LpProblem, LpVariable, LpBinary, lpSum, LpMinimize, PULP_CBC_CMD, LpStatus
from itertools import combinations

class IDSModel:
    def __init__(self, lambda1=0.1, lambda2=0.1, lambda3=1.0, lambda4=1.0,
                 min_support=0.05, min_confidence=0.6, max_rule_length=3):
        """
        Inicializa el modelo IDS con los hiperparámetros dados.
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_rule_length = max_rule_length
        self.rules = []
        self.selected_rules = []

    def fit(self, X, y):
        """
        Entrena el modelo IDS con los datos proporcionados.
        """
        # Combinar X e y en un DataFrame
        df = X.copy()
        df['target'] = y.values
        df = df.astype(str)

        # Generar reglas candidatas
        self.rules = generate_candidate_rules(df, self.min_support, self.min_confidence, self.max_rule_length)

        if not self.rules:
            raise ValueError("No se generaron reglas candidatas. Ajusta los parámetros de soporte y confianza.")

        # Calcular métricas de las reglas
        rule_covers, rule_correct_covers, rule_lengths, rule_errors = calculate_rule_metrics(self.rules, df)

        # Formular y resolver el problema de optimización
        prob, rule_vars = self._formulate_optimization_problem(rule_covers, rule_correct_covers, rule_lengths, df)
        prob = self._solve_optimization_problem(prob)

        # Obtener las reglas seleccionadas
        self.selected_rules = self._get_optimal_rules(prob, rule_vars)

    def _formulate_optimization_problem(self, rule_covers, rule_correct_covers, rule_lengths, df):
        """
        Formula el problema de optimización para seleccionar el subconjunto óptimo de reglas.
        """
        num_rules = len(self.rules)
        num_samples = df.shape[0]

        # Variables de decisión
        rule_vars = LpVariable.dicts("Rule", range(num_rules), cat=LpBinary)
        sample_error_vars = LpVariable.dicts("SampleError", range(num_samples), cat=LpBinary)

        # Función objetivo
        prob = LpProblem("IDS", LpMinimize)

        # Penalización por solapamiento
        overlap_penalty = []
        for i, j in combinations(range(num_rules), 2):
            if self.rules[i].class_label != self.rules[j].class_label:
                overlap = np.minimum(rule_covers[i], rule_covers[j])
                if overlap.sum() > 0:
                    overlap_var = LpVariable(f"Overlap_{i}_{j}", cat=LpBinary)
                    prob += overlap_var <= rule_vars[i]
                    prob += overlap_var <= rule_vars[j]
                    prob += overlap_var >= rule_vars[i] + rule_vars[j] - 1
                    overlap_penalty.append(overlap_var)

        prob += lpSum([rule_vars[r] for r in range(num_rules)]) * self.lambda1 + \
                lpSum([rule_lengths[r] * rule_vars[r] for r in range(num_rules)]) * self.lambda2 + \
                lpSum([sample_error_vars[s] for s in range(num_samples)]) * self.lambda3 + \
                lpSum(overlap_penalty) * self.lambda4

        # Restricciones
        for s in range(num_samples):
            correct_cover = lpSum([rule_correct_covers[r][s] * rule_vars[r] for r in range(num_rules)])
            prob += sample_error_vars[s] >= 1 - correct_cover

        return prob, rule_vars

    def _solve_optimization_problem(self, prob):
        """
        Resuelve el problema de optimización utilizando el solver de PuLP.
        """
        solver = PULP_CBC_CMD(msg=0)
        prob.solve(solver)
        if LpStatus[prob.status] != 'Optimal':
            raise ValueError("No se encontró una solución óptima al problema de optimización.")
        return prob

    def _get_optimal_rules(self, prob, rule_vars):
        """
        Extrae las reglas seleccionadas después de resolver el problema de optimización.
        """
        selected_rules = []
        for r in rule_vars:
            if rule_vars[r].varValue == 1:
                selected_rules.append(self.rules[r])
        return selected_rules

    def predict(self, X):
        """
        Realiza predicciones en los datos proporcionados utilizando las reglas seleccionadas.
        """
        df_X = X.astype(str)
        predictions = []
        for _, row in df_X.iterrows():
            x = row.to_dict()
            votes = []
            for rule in self.selected_rules:
                if rule.covers(x):
                    votes.append(rule.class_label)
            if votes:
                pred = max(set(votes), key=votes.count)
                predictions.append(pred)
            else:
                predictions.append(0)  # Clase por defecto si ninguna regla coincide
        return np.array(predictions)

    def print_rules(self):
        """
        Imprime las reglas seleccionadas por el modelo.
        """
        print("Reglas seleccionadas:")
        for rule in self.selected_rules:
            print(rule)
    
    def print_rules_in_dt_format(self):
        """
        Imprime las reglas seleccionadas en el formato solicitado similar a DT.
        """
        formatted_rules = []
        for rule in self.selected_rules:
            conditions = rule.conditions
            outcome = rule.class_label

            # Construir la regla formateada
            formatted_rule = "si " + " y ".join([f"{feature} ≤ {value}" if isinstance(value, (int, float)) and float(value) <= 0.5 else f"{feature} > {value}" for feature, value in conditions])
            
            formatted_rule += f" entonces {outcome}"
            formatted_rules.append(formatted_rule)

        # Imprimir las reglas formateadas
        print("\nReglas de decisión en el formato solicitado:")
        for rule in formatted_rules:
            print(rule)
