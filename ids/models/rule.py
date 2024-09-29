# ids/models/rule.py

class Rule:
    def __init__(self, conditions, class_label):
        """
        Inicializa una regla con sus condiciones y la etiqueta de clase.
        """
        self.conditions = conditions  # Lista de tuplas (feature, value)
        self.class_label = class_label

    def covers(self, x):
        """
        Verifica si la regla cubre una muestra dada.
        """
        return all(str(x.get(feature, -1)) == str(value) for feature, value in self.conditions)

    def __len__(self):
        """
        Devuelve la longitud de la regla (número de condiciones).
        """
        return len(self.conditions)

    def __repr__(self):
        """
        Representación en cadena de la regla.
        """
        conditions_str = ' AND '.join([f"{feature}={value}" for feature, value in self.conditions])
        return f"IF {conditions_str} THEN {self.class_label}"
