import pandas as pd

class ModelRecommender:
    """Analyzes a dataset and suggests suitable ML models."""

    def __init__(self, dataset, target):
        self.dataset = dataset.copy()
        self.target = target

    def detect_problem_type(self):

        if self.dataset is None or self.dataset.empty:
            return {}

        # check if the target column exists
        if self.target not in self.dataset.columns:
            raise ValueError({f"Target column '{self.target}' doesn't exist in the dataset."})

        target_data = self.dataset[self.target]

        # count unique target_data values
        unique_count = target_data.nunique()

        # validate unique_count of target
        if unique_count <= 1:
            raise ValueError(
                f"Target column '{self.target}' must contain at least two unique values."
            )

        # check what dtype target is numeric
        if not pd.api.types.is_numeric_dtype(target_data):
            if unique_count == 2:
                return {
                    "task_type": "classification",
                    "subtype": "binary",
                    "target": self.target,
                    "num_classes": unique_count
                }
            elif unique_count > 2:
                return {
                    "task_type": "classification",
                    "subtype": "multiclass",
                    "target": self.target,
                    "num_classes": unique_count
                }

        # check if target is integer-like/discrete or continuous
        # removing missing values
        clean_target = target_data.dropna()

        is_integer_like = (clean_target % 1 == 0).all()    # is every numeric val a whole number?

        # Numeric target with exactly two values
        if unique_count == 2:
            return {
                "task_type": "classification",
                "subtype": "binary",
                "target": self.target,
                "num_classes": unique_count
            }

        # Numeric target with a small num of discrete classes
        elif is_integer_like and unique_count <= 20:
            return{
                "target_type": "classification",
                "subclass": "multiclass",
                "target": self.target,
                "num_classes": unique_count
            }

        # Otherwise assume a continuous numeric target
        else:
            return {
                "target_type": "regression",
                "subclass": None,
                "target": self.target,
                "num_classes": None
            }
        