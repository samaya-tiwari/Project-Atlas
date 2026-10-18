import pandas as pd
from itertools import combinations

class FeatureEngineer:
    """Creates, evaluates, and selects useful dataset features."""

    def __init__(self, dataset):
        self.dataset = dataset.copy()

    def drop_features(self, columns):
        """Drops explicitly selected features from the dataset."""

        if self.dataset is None or self.dataset.empty:
            return {}

        if not isinstance(columns, list):
            raise TypeError("columns must be a list.")

        for column in columns:
            if column not in self.dataset.columns:
                raise ValueError(
                    f"Column '{column}' does not exist in the dataset."
                )

        self.dataset = self.dataset.drop(columns=columns)

        return self.dataset

    def detect_constant_features(self):
        """Detects features that contain only one unique value."""

        if self.dataset is None or self.dataset.empty:
            return []

        const_feature = []

        for column in self.dataset.columns:
            unique_count = self.dataset[column].nunique()

            if unique_count <=1:
                const_feature.append(column)

        return const_feature

    def detect_near_constant_feature(self, threshold=0.95):
        """Detected the near-constant features that certainly dominates the rest of the values."""

        if self.dataset is None or self.dataset.empty:
            return {}

        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1.")

        near_const_features = {}

        for column in self.dataset.columns:
            val_proportions = self.dataset[column].value_counts(normalize=True, dropna=True)

            if val_proportions.empty:
                continue

            dominant_value = val_proportions.index[0]
            dominant_proportion = val_proportions.iloc[0]

            if dominant_proportion >= threshold:
                near_const_features[column] = {
                    "dominant_values" : dominant_value,
                    "proportion" : round(float(dominant_proportion), 4)
                }

            return near_const_features

    def detect_high_correlation_features(self, threshold= 0.90):
        """Detects pairs of numerical features with high absolute correlation."""

        if self.dataset is None or self.dataset.empty:
            return []

        if not 0 < threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")

        numeric_data = self.dataset.select_dtypes(include=["number"])

        if numeric_data.shape[1] < 2:
            return []

        correlation_matrix = numeric_data.corr()

        high_correlation_pairs = []

        for feature_1, feature_2 in combinations(numeric_data.columns, 2):    # checks two features once only

            correlation = correlation_matrix.loc[feature_1, feature_2]

            if pd.isna(correlation):  
                continue

            if abs(correlation) >= threshold:       # both strong +ve and -ve relationships are detected
                high_correlation_pairs.append({
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": round(float(correlation), 4)
                })

        return high_correlation_pairs