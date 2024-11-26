# ids/ids.py

import numpy as np
import pandas as pd
from pulp import LpProblem, LpVariable, LpBinary, lpSum, LpMinimize, PULP_CBC_CMD, LpStatus
from itertools import combinations
from sklearn.base import BaseEstimator
from .utils import generate_candidate_rules, calculate_rule_metrics
from .models.rule import Rule

class IDSModel(BaseEstimator):
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
        self.boolean_cols = None

    def fit(self, X, y):
        """
        Entrena el modelo IDS con los datos proporcionados.
        """
        # Identificar columnas booleanas en X (asumiendo valores 0/1)
        self.boolean_cols = [col for col in X.columns if set(X[col].unique()).issubset({0, 1})]
        print(f"Columnas identificadas como booleanas: {self.boolean_cols}")

        # Combinar X e y en un DataFrame
        df = X.copy()
        df['target'] = y.values
        df = df.astype(str)  # Convertir a cadenas para Apriori

        # Generar reglas candidatas
        self.rules = generate_candidate_rules(df, self.min_support, self.min_confidence, self.max_rule_length, boolean_cols=self.boolean_cols)

        if not self.rules:
            raise ValueError("No se generaron reglas candidatas. Ajusta los parámetros de soporte y confianza.")

        # Calcular métricas de las reglas
        rule_covers, rule_correct_covers, rule_lengths, rule_errors = calculate_rule_metrics(self.rules, df)

        # Formular y resolver el problema de optimización
        prob, rule_vars = self._formulate_optimization_problem(rule_covers, rule_correct_covers, rule_lengths, df)
        prob = self._solve_optimization_problem(prob)

        # Obtener las reglas seleccionadas
        self.selected_rules = self._get_optimal_rules(prob, rule_vars)

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
                if rule.covers(x):  # Eliminado boolean_cols=self.boolean_cols
                    votes.append(rule.class_label)
            if votes:
                pred = max(set(votes), key=votes.count)
                predictions.append(pred)
            else:
                predictions.append(0)  # Clase por defecto si ninguna regla coincide
        return np.array(predictions)

    def get_params(self, deep=True):
        """
        Devuelve los parámetros del modelo para GridSearchCV.
        """
        return {
            'lambda1': self.lambda1,
            'lambda2': self.lambda2,
            'lambda3': self.lambda3,
            'lambda4': self.lambda4,
            'min_support': self.min_support,
            'min_confidence': self.min_confidence,
            'max_rule_length': self.max_rule_length
        }

    def set_params(self, **params):
        """
        Ajusta los parámetros del modelo para GridSearchCV.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

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

    def print_rules(self, X_train=None, y_train=None, label_mapping=None):
        if label_mapping is None:
            label_mapping = {0: 'Reprobado', 1: 'Aprobado'}  # Mapeo predeterminado para 0 y 1

        if X_train is None or y_train is None:
            print("Advertencia: No se proporcionaron X_train e y_train. Las reglas se imprimirán sin precisión ni muestras.")
            # Si no se proporcionan X_train e y_train, imprime las reglas sin precisión ni muestras
            for rule in self.selected_rules:
                print(rule)  # Utiliza el método __repr__ de Rule
            return

        # Si se proporcionan X_train e y_train, imprime las reglas con precisión y muestras
        print("\nReglas seleccionadas con precisión y muestras:")
        for rule in self.selected_rules:
            outcome = rule.class_label

            # Determinar las muestras cubiertas por la regla
            covered_samples = [i for i, row in X_train.iterrows() if rule.covers(row)]
            num_samples = len(covered_samples)
            if num_samples > 0:
                correct_samples = sum([1 for i in covered_samples if y_train.iloc[i] == outcome])
                precision = correct_samples / num_samples
            else:
                precision = 0

            # Imprimir la regla formateada utilizando __repr__
            formatted_rule = f"{rule} (Precisión: {precision:.2f}, Muestras: {num_samples})"
            print(formatted_rule)

