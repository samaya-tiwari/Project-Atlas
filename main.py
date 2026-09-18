import pandas as pd

from atlas.data.preprocessor import Preprocessor

df = pd.DataFrame({
    "city": [
        " London ",
        "london",
        "LONDON ",
        "?",
        " Paris "
    ],
    "status": [
        "ACTIVE",
        "active",
        " Active ",
        "unknown",
        "INACTIVE"
    ],
    "age": [
        21,
        -5,
        150,
        24,
        30
    ],
    "score": [
        88,
        92,
        -10,
        105,
        76
    ]
})

preprocessor = Preprocessor(df)

result = preprocessor.clean_inconsistent_data(
    strip_whitespace=None,
    normalize_case="lower",
    replacements={
        "city": {
            "?": "unknown"
        },
        "status": {
            "unknown": "inactive"
        }
    },

    numeric_rules={
        "age": {
            "min": 0,
            "max": 120
        },
        "score": {
            "min": 0,
            "max": 100
        }
    }
)

print(df)

print("CLEANED DATA:")
print(result)