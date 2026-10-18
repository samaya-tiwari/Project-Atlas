import pandas as pd

from atlas.data.feature_engineer import FeatureEngineer


df = pd.DataFrame({
    "student_id": [1001, 1002, 1003],
    "name": ["Alice", "Bob", "Charlie"],
    "age": [20, 21, 22],
    "gpa": [3.8, 3.2, 3.5],
    "hours_studied": [12, 8, 10]
})

feature_engineer = FeatureEngineer(df)

print("BEFORE:")
print(feature_engineer.dataset)

result = feature_engineer.select_features([
    "age",
    "gpa",
    "fake_col"
])

print("\nAFTER:")
print(result)