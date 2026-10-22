from pprint import pprint
from atlas.data.dataset_loader import DatasetLoader
from atlas.data.preprocessor import Preprocessor
from atlas.data.feature_engineer import FeatureEngineer

# LOAD DATASET
loader = DatasetLoader()
loader.load("datasets/sample.csv")

print("-----Original Data-----")
print(loader.dataset.to_string(index=False))


# PREPROCESS DATASET
preprocessor = Preprocessor(loader.dataset)

preprocessor_config = {
    "inconsistent_data": {
        "strip_whitespace": True,
        "normalize_case": "lower"
    },

    "datetime": {
        "columns": ["enrollment_date"]
    },

    "missing_values": {
        "strategies": {
            "age": "median",
            "gpa": "median",
            "city": "mode",
            "income": "median"
        }
    },

    "remove_duplicates": True
}


cleaned_df = preprocessor.preprocess(preprocessor_config)

print("\n-----AFTER PREPROCESSING-----")
print(cleaned_df.to_string(index=False))


# FEATURE ENGINEERING
feature_engineer = FeatureEngineer(cleaned_df)

feature_config = {
    "datetime_features": {
        "columns": ["enrollment_date"],
        "features": [
            "year", "month", "dayofweek", "quarter", "is_weekend"
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


engineered_df = feature_engineer.engineer_features(
    feature_config
)

print("\n-----FINAL ENGINEERED DATA-----")
print(engineered_df.to_string(index=False))

print("\n-----FINAL ANALYSIS REPORT-----")
pprint(feature_engineer.feature_report)