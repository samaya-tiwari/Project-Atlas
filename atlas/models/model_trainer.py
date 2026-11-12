import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

class ModelTrainer:
    """Prepares datasets and trains ML models"""
    def __init__(self, dataset, target=None):
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("Dataset must be a pandas DataFrame.")

        if dataset.empty:
            raise ValueError("Dataset cannot be empty.")

        self.dataset = dataset.copy()
        self.target = None

        self.X = None
        self.y = None

        self.feature_names = []
        self.target_info = {}

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        if target is not None:
            self.set_target(target)

    def set_target(self, target):
        """Validates and sets the target column."""

        if not isinstance(target, str):
            raise TypeError("The target must be a column name as a string.")

        if target not in self.dataset.columns:
            raise ValueError(f"Target column '{target}' does not exist in the dataset.")

        target_data = self.dataset[target]
        non_missing_target = target_data.dropna()

        if non_missing_target.empty:
            raise ValueError(f"Target column '{target}' contains no usable values.")

        unique_count = non_missing_target.nunique()

        if unique_count <= 1:
            raise ValueError(f"Target column '{target}' must contain at least two unique values.")

        missing_count = target_data.isna().sum()

        missing_percentage = (missing_count/len(self.dataset)) * 100

        self.target = target

        self.target_info = {
            "target": target,
            "unique_count": unique_count,
            "missing_count": int(missing_count),
            "missing_percentage": float(round(missing_percentage, 2))
        }

        # Reset previously prepared data if target changes
        self.X = None
        self.y = None
        self.feature_names = []

        return self.target_info

    def prepare_feature_target(self):
        """Separates the database into model features X and target y."""

        if self.target is None:
            raise ValueError("No target column has been selected. "
                             "Call set_target() before preparing training data.")

        # Do not invent missing target values
        # Rows without a known target cannot be used for supervised training
        
        training_data = self.dataset.dropna(
            subset=[self.target]
        ).copy()

        if training_data.empty:
            raise ValueError("No usable rows remain after removing missing target values.")

        # Separate features and target
        X = training_data.drop(
            columns=[self.target]
        ).copy()

        y = training_data[self.target].copy()

        if X.empty or X.shape[1] == 0:
            raise ValueError("No input features remain after removing the target column.")

        # Extra safety check layer
        if self.target in X.columns:
            raise RuntimeError(
                "Target column was not properly removed from the training features."
            )

        # Target should not contain any missing values now
        if y.isna().any():
            raise RuntimeError(
                "Target still contains some missing values after preparation."
            )

        # Preprocessing should already have handled missing feature values
        missing_features = X.columns[
            X.isna().any()
        ].tolist()

        if missing_features:
            raise ValueError(
                "Input feautres still contian missing values: "
                f"{missing_features}. "
                "Run preprocessing before model training."
            )

        # Majority of sklearn models require numeric/boolean features
        unsupported_features = []

        for column in X.columns:
            columns_data = X[column]

            if not (pd.api.types.is_bool_dtype(columns_data) or pd.api.types.is_numeric_dtype(columns_data)):
                unsupported_features.append(column)

        if unsupported_features:
            raise ValueError(
                "Input features contain non-numeric columns: "
                f"{unsupported_features}. "
                "Encode categorical features before model training."
            )

        # Detect infinity values because many sklearn models reject them
        numeric_data = X.select_dtypes(
            include=["number"]
        )

        if not numeric_data.empty:
            has_infinte_values = np.isinf(numeric_data.to_numpy()).any()

            if has_infinte_values:
                raise ValueError(
                    "Input features contain infinite values. Clean them before model training."
                )

        self.X = X
        self.y = y
        self.feature_names = X.columns.tolist()

        return self.X, self.y

    def split_data(self, test_size = 0.20, random_state=42, stratify=False):
        """Splits the prepared features and target into training and testing sets."""

        if self.X is None or self.y is None:
            raise ValueError(
                "Training data has not been prepared. "
                "Call perpare_features_target() first."
            )

        if not 0 < test_size < 1:
            raise ValueError(
                "test_size must be between 0 and 1."
            )

        stratify_data = None

        if stratify:
            stratify_data = self.y

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state= random_state,
            stratify=stratify_data
        )

        return {
            "training_rows": len(self.X_train),
            "testing_rows": len(self.X_test),
            "test_size": test_size,
            "random_state": random_state,
            "stratified": stratify
        }