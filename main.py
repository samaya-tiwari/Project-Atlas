from atlas.data.dataset_loader import DatasetLoader
from atlas.data.preprocessor import Preprocessor


loader = DatasetLoader()
loader.load("datasets/sample.csv")

preprocessor = Preprocessor(loader.dataset)


config = {
    "inconsistent_data": {
        "strip_whitespace": True,
        "normalize_case": "lower",
        "replacements": None,
        "numeric_rules": {
            "age": {
                "min": 0,
                "max": 120
            }
        }
    },

    "datetime": {
        "columns": ["enrollment_date"]
    },

    "missing_values": {
        "strategies": {
            "age": "median",
            "income": "median",
            "city": "mode"
        },
        "max_drop_percentage": 20,
        "constant_values": None
    },

    "remove_duplicates": True,

    "categorical_encoding": {
        "strategies": {
            "major": "one_hot",
            "city": "one_hot"
        }
    },

    "numerical_scaling": {
        "strategies": {
            "age": "standardize",
            "income": "normalize"
        }
    }
}


result = preprocessor.preprocess(config)

print(result)