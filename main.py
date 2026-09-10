# test

import pandas as pd

from atlas.data.preprocessor import Preprocessor

df = pd.DataFrame({
    "age": [20, 22, None, 24, 26],
    "salary": [50000, None, 60000, 55000, 65000],
    "city": ["London", "Paris", None, "London", "Tokyo"],
    "status": ["Active", None, "Active", "Inactive", "Active"],
    "score": [10, 20, None, 30, 40]
})

print("ORIGINAL DATA:")
print(df)

# MEAN
preprocessor = Preprocessor(df)

result = preprocessor.handle_missing_values(
    {"salary" : "drop"},
    max_drop_percentage=10
)

print("\nMEAN TEST:")
print(result)