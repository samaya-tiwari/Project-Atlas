from atlas.data.dataset_loader import DatasetLoader
from atlas.data.preprocessor import Preprocessor
from atlas.data.feature_engineer import FeatureEngineer

loader = DatasetLoader()
loader.load("datasets/sample.csv")

preprocessor = Preprocessor(loader.dataset)

cleaned_df = preprocessor.clean_inconsistent_data(
    strip_whitespace=True,
    normalize_case="lower"
)

# convert enrollment_date to real datetime dtype
cleaned_df = preprocessor.prepare_datetime(
    ["enrollment_date"]
)

feature_engineer = FeatureEngineer(cleaned_df)

config = {
    "datetime_features": {
        "columns": ["enrollment_date"],
        "features": [
            "year",
            "month",
            "dayofweek"
        ],
        "drop_original": False
    },

    "drop_features": [
        "name"
    ],

    "analyze_drop_candidates": {
        "near_constant_threshold": 0.95,
        "correlation_threshold": 0.90,
        "identifier_uniqueness_threshold": 0.95
    }
}

result = feature_engineer.engineer_features(config)

print(result)
print(feature_engineer.feature_report)