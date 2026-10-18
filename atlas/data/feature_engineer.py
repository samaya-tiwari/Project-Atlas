import pandas as pd
from itertools import combinations
import re

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

            if pd.isna(correlation):     # handles constant numeric columns where correlation may be NaN
                continue

            if abs(correlation) >= threshold:       # both strong +ve and -ve relationships are detected
                high_correlation_pairs.append({
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": round(float(correlation), 4)
                })

        return high_correlation_pairs

    def detect_identifier_features(self, uniqueness_threshold=0.95):
        """Detects identifier-like features and potentially redundant identifiers."""

        if self.dataset is None or self.dataset.empty:
            return {}

        if not 0 < uniqueness_threshold <= 1:
            raise ValueError(
                "uniqueness_threshold must be between 0 and 1."
            )

        identifier_features = {}

        id_name_patterns = {
            "id",
            "student_id",
            "customer_id",
            "user_id",
            "uuid",
            "guid"
        }

        uuid_pattern = re.compile(
            r"^[0-9a-fA-F]{8}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{12}$"
        )

        for column in self.dataset.columns:
            non_missing = self.dataset[column].dropna()

            if non_missing.empty:
                continue

            unique_ratio = (
                non_missing.nunique() / len(non_missing)
            )

            column_name = column.lower()

            name_suggest_id = (
                column_name in id_name_patterns or column_name.endswith("_id")
            )

            uuid_like = False

            if pd.api.types.is_object_dtype(non_missing):
                string_values = non_missing.astype(str)

                uuid_matches = string_values.apply(
                    lambda value: bool(uuid_pattern.match(value))
                )

                uuid_like = uuid_matches.mean() >= uniqueness_threshold

            if unique_ratio >= uniqueness_threshold and (name_suggest_id or uuid_like):
                reasons = []

                if name_suggest_id:
                    reasons.append("identifier-like column name")

                if uuid_like:
                    reasons.append("UUID-like values")

                reasons.append("high uniqueness")

                identifier_features[column] = {
                    "unique_ratio": round(float(unique_ratio), 4),
                    "reasons": reasons
                }
        potentially_redundant = []

        identifier_columns = list(identifier_features.keys())

        if len(identifier_columns) > 1:
            potentially_redundant = identifier_columns

        return {
            "identifier_features": identifier_features,
            "potentially_redundant_identifier": potentially_redundant
        }