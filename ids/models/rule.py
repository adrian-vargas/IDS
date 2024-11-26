# ids/models/rule.py

class Rule:
    def __init__(self, conditions, class_label, boolean_cols=None):
        self.conditions = conditions  # Lista de tuplas (feature, value)
        self.class_label = class_label
        self.boolean_cols = boolean_cols if boolean_cols is not None else []

    def covers(self, x):
        for feature, value in self.conditions:
            feature_value = x.get(feature, -1)
            try:
                feature_value = float(feature_value)
                value = float(value)
                if feature in self.boolean_cols:
                    if int(round(feature_value)) != int(value):
                        return False
                else:
                    if feature_value <= value:
                        return False
            except ValueError:
                if str(feature_value) != str(value):
                    return False
        return True

    def __len__(self):
        """
        Devuelve la longitud de la regla (número de condiciones).
        """
        return len(self.conditions)

    def __repr__(self):
        # Mapeo de etiquetas
        label_mapping = {0: 'Reprobado', 1: 'Aprobado'}

        # Función para formatear valores
        def format_value(value):
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            elif isinstance(value, int):
                return str(value)
            else:
                return str(value)

        conditions_str = ' y '.join([
            f"{feature}={format_value(value)}" if feature in self.boolean_cols else f"{feature} > {format_value(value)}"
            for feature, value in self.conditions
        ])
        class_label_str = label_mapping.get(self.class_label, self.class_label)
        return f"si {conditions_str} entonces {class_label_str}"