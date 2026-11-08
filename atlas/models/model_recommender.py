import pandas as pd

class ModelRecommender:
    """Analyzes a dataset and suggests suitable ML models."""

    def __init__(self, dataset, target=None):
        self.dataset = dataset.copy()
        self.target = None

        if target is not None:
            self.set_target(target)

    def detect_problem_type(self):

        if self.dataset is None or self.dataset.empty:
            return {}

        if self.target is None:
            raise ValueError(
                "No target column has been selected. "
                "Provide a target or self_target() first."
            )

        # check if the target column exists
        if self.target not in self.dataset.columns:
            raise ValueError(f"Target column '{self.target}' doesn't exist in the dataset.")

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
        dataset_info = self.analyze_dataset()

        task_type = problem_info["task_type"]
        subtype = problem_info["subtype"]

        dataset_size = dataset_info["dataset_size"]
        
        feature_size = dataset_info["feature_size"]

        recommendations = []

        if task_type == "classification" and subtype == "binary":
            recommendations = [
                {
                    "model": "logistic_regression",
                    "score": 8,
                    "reason": "Strong and interpretable baseline for binary classification."
                },
                {
                    "model": "decision_tree",
                    "score": 6,
                    "reason": "Can capture non-linear relationships and is easy to interpret."
                },
                {
                    "model": "random_forest",
                    "score": 8,
                    "reason": "Handles nonlinear patterns and feature interactions well."
                },
                {
                    "model": "knn",
                    "score": 7,
                    "reason": "Useful when similar observations tend to belong to the same class."
                },
                {
                    "model": "svc",
                    "score": 7,
                    "reason": "Can perform well on classification problems with properly scaled features."
                }
            ]

        elif task_type == "classification" and subtype == "multiclass":
            recommendations = [
                {
                    "model": "logistic_regression",
                    "score": 7,
                    "reason": "Can handle multiclass classification using strategies such as multinomial classification."
                },
                {
                    "model": "decision_tree",
                    "score": 6,
                    "reason": "Can naturally separate data into multiple classes using decision rules."
                },
                {
                    "model": "random_forest",
                    "score": 8,
                    "reason": "Combines multiple decision trees and performs well on many multiclass tabular datasets."
                },
                {
                    "model": "knn",
                    "score": 7,
                    "reason": "Can classify observations into multiple classes based on nearby training examples."
                },
                {
                    "model": "svc",
                    "score": 7,
                    "reason": "Can support multiclass classification by combining multiple binary classification decisions."
                }
            ]

        elif task_type == "regression":
            recommendations = [
                {
                    "model": "linear_regression",
                    "score": 7,
                    "reason": "Provides a simple and interpretable baseline for continuous targets."
                },
                {
                    "model": "decision_tree_regressor",
                    "score": 6,
                    "reason": "Can capture nonlinear relationships without requiring feature scaling."
                },
                {
                    "model": "random_forest_regressor",
                    "score": 8,
                    "reason": "Handles nonlinear patterns and feature interactions well on tabular data."
                },
                {
                    "model": "knn_regressor",
                    "score": 6,
                    "reason": "Useful when observations with similar features tend to have similar target values."
                },
                {
                    "model": "svr",
                    "score": 7,
                    "reason": "Can model complex relationships, especially when numerical features are properly scaled."
                }
            ]

        # Adjusting model scores based on dataset charactersitics
        for recommendation in recommendations:
            model = recommendation["model"]
            recommendation["adjustments"] = []

            # dataset size adjustments
            if dataset_size == "small":

                if model in {"knn", "knn_regressor"}:
                    recommendation["score"] += 1
                    recommendation["adjustments"].append(
                        "Small datasets are suitable for neighbor-based models."
                    )

                if model in {"svc", "svr"}:
                    recommendation["score"] += 1
                    recommendation["adjustments"].append(
                        "Support vector models can perform well on smaller datasets."
                    )

            elif dataset_size == "large":

                if model in {"knn", "knn_regressor"}:
                    recommendation["score"] -= 2
                    recommendation["adjustments"].append(
                        "KNN can become expensive on large datasets."
                    )

                if model in {"svc", "svr"}:
                    recommendation["score"] -= 2
                    recommendation["adjustments"].append(
                        "Support vector models can become computationally expensive on large datasets."
                    )

                if model in {"logistic_regression", "linear_regression"}:
                    recommendation["score"] += 1
                    recommendation["adjustments"].append(
                        "Linear models generally scale well to larger datasets."
                    )

            # Feature size adjustments
            if feature_size == "high":
                if model in {"knn", "knn_regressor"}:
                    recommendation["score"] -= 2
                    recommendation["adjustments"].append(
                        "KNN can struggle when the dataset has many features."
                    )

                if model in {"logistic_regression", "linear_regression"}:
                    recommendation["score"] += 1
                    recommendation["adjustments"].append(
                        "Linear models can work well with higher-dimensional datasets."
                    )

            # keeping score within a clean 1-10 range
            recommendation["score"] = max(1, min(10, recommendation["score"]))

        recommendations.sort(
            key=lambda model: model["score"],
            reverse=True
        )

        return {
            "task_type": task_type,
            "subtype": subtype,
            "target": self.target,
            "recommended_models": recommendations
        }

    def recommend_target(self):
        """Recommends likely target columns using simple dataset heuristics."""

        if self.dataset is None or self.dataset.empty:
            return {}

        candidates = []

        target_name_hints = {
            "target",
            "label",
            "outcome",
            "class",
            "churn",
            "price",
            "score",
            "fraud",
            "status",
            "diagnosis",
            "default",
            "survived",
            "approved",
            "result"
        }

        for column in self.dataset.columns:
            column_data = self.dataset[column]
            unique_count = column_data.nunique()
            non_missing_count = column_data.notna().sum()

            # skip useless columns
            if unique_count <= 1:
                continue

            if non_missing_count == 0:
                continue

            unique_ratio = unique_count / non_missing_count

            missing_percentage = (column_data.isna().sum() / len(self.dataset)) * 100

            score = 0
            reasons = []
            column_name = column.lower()

            # Column name signal
            if column_name in target_name_hints:
                score += 3
                reasons.append("Column name suggests a possible prediction target.")

            # Identifier case
            if (column_name == "id" or column_name.endswith("_id") or "uuid" in column_name or "guid" in column_name):
                score -= 5
                reasons.append("Column appears to be an identifier.")

            # Classification-like target
            if 2<=unique_count<=20 and unique_ratio <= 0.5:
                score += 2
                reasons.append("Column contains a small number of distinct values.")

            # Continuous Numeric Target
            elif pd.api.types.is_numeric_dtype(column_data):
                score+=1
                reasons.append("Numeric column could represent a continuous regression target.")

            # missing value signal
            if missing_percentage <= 10:
                score += 1
                reasons.append("Column has relatively few missing values.")

            elif missing_percentage > 50:
                score -= 2
                reasons.append("Column contains many missing values.")

            # last column signal
            if column == self.dataset.columns[-1]:
                score += 1
                reasons.append("Column is the final column in the dataset.")

            candidates.append({
                "column": column,
                "score": score,
                "unique_count": unique_count,
                "unique_ratio": float(round(unique_ratio, 3)),
                "missing_percentage": float(round(missing_percentage, 2)),
                "reasons": reasons
            })

        candidates.sort(
            key= lambda candidate: candidate["score"],
            reverse=True
        )

        if not candidates:
            return {
                "recommended_target": None,
                "candidates": []
            }

        return {
            "recommended_target": candidates[0]["column"],
            "candidates": candidates
        }
    

    def analyze_dataset(self):
        """This analyzes the dataset and returns useful characteristics."""

        if self.target is None:
            raise ValueError(
                "No target column has been selected. "
                "Provide a target or call set_target() first."
            )

        # drop the target column from the dataset to analyze
        features = self.dataset.drop(columns=[self.target])

        # dataset size
        num_rows = len(self.dataset)
        num_features = features.shape[1]

        # Numeric features
        numeric_features = features.select_dtypes(
            include=["number"]
        )
        num_numeric_features = numeric_features.shape[1]

        # Non-numeric/Categorical features
        categorical_features = features.select_dtypes(
            exclude=["number"]
        )
        num_categorical_features = categorical_features.shape[1]

        # determining dataset size
        if num_rows < 1000:
            dataset_size = "small"
        elif num_rows <= 50000:
            dataset_size = "medium"
        else:
            dataset_size = "large"

        # determine feature size
        if num_features < 10:
            feature_size = "low"
        elif num_features <= 50:
            feature_size = "medium"
        else:
            feature_size = "high"

        return{
            "num_rows": num_rows,
            "num_features": num_features,
            "num_numeric_features": num_numeric_features,
            "num_categorical_features": num_categorical_features,
            "dataset_size": dataset_size,
            "feature_size": feature_size
        }

    def set_target(self, target):
        """Validates and sets the target column."""

        if not isinstance(target, str):
            raise TypeError("Target must be a column name as a string.")

        if target not in self.dataset.columns:
            raise ValueError(
                f"Target column '{target}' does not exist in the dataset."
            )

        target_data = self.dataset[target]

        unique_count = target_data.nunique()

        if unique_count <= 1:
            raise ValueError(
                f"Target column '{target}' must contain at least two unique values."
            )

        self.target = target

        return {
            "target": self.target,
            "unique_count": unique_count
        }