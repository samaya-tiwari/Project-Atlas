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
                "task_type": "classification",
                "subtype": "multiclass",
                "target": self.target,
                "num_classes": unique_count
            }

        # Otherwise assume a continuous numeric target
        else:
            return {
                "task_type": "regression",
                "subtype": None,
                "target": self.target,
                "num_classes": None
            }

    def recommend_models(self):
        problem_info = self.detect_problem_type()

        task_type = problem_info["task_type"]
        subtype = problem_info["subtype"]

        recommendations = []

        if task_type == "classification" and subtype == "binary":
            recommendations = [
                {
                    "model": "logistic_regression",
                    "reason": "Strong and interpretable baseline for binary classification."
                },
                {
                    "model": "decision_tree",
                    "reason": "Can capture non-linear relationships and is easy to interpret."
                },
                {
                    "model": "random_forest",
                    "reason": "Handles nonlinear patterns and features interactions well."
                },
                {
                    "model": "knn",
                    "reason": "Useful when similar observations tend to belong to the same class."
                },
                {
                    "model": "svc",
                    "reason": "Can perform well on classification problems with properly scaled features."
                }
            ]

        elif task_type == "classification" and subtype == "multiclass":
            recommendations = [
                {
                    "model": "logistic_regression",
                    "reason": "Can handle multiclass classification using strategies such as multinomial classification."
                },
                {
                    "model": "decision_tree",
                    "reason": "Can naturally separate data into multiple classes using decision rules."
                },
                {
                    "model": "random_forest",
                    "reason": "Combines multiple decision trees and performs well on many multiclass tabular datasets."
                },
                {
                    "model": "knn",
                    "reason": "Can classify observations into multiple classes based on nearby training examples."
                },
                {
                    "model": "svc",
                    "reason": "Can support multiclass classification by combining multiple binary classification decisions."
                }
            ]

        elif task_type == "regression":
            recommendations = [
                {
                    "model": "linear_regression",
                    "reason": "Provides a simple and interpretable baseline for continuous targets."
                },
                {
                    "model": "decision_tree_regressor",
                    "reason": "Can capture nonlinear relationships without requiring feature scaling."
                },
                {
                    "model": "random_forest_regressor",
                    "reason": "Handles nonlinear patterns and feature interactions well on tabular data."
                },
                {
                    "model": "knn_regressor",
                    "reason": "Useful when observations with similar features tend to have similar target values."
                },
                {
                    "model": "svr",
                    "reason": "Can model complex relationships, especially when numerical features are properly scaled."
                }
            ]

        return {
            "task_type": task_type,
            "subtype": subtype,
            "target": self.target,
            "recommended_models": recommendations
        }