import pandas as pd
from atlas.data.feature_engineer import FeatureEngineer

df = pd.DataFrame({
    "student_id": [1001, 1002, 1003, 1004],
    "uuid": [
        "550e8400-e29b-41d4-a716-446655440000",
        "550e8400-e29b-41d4-a716-446655440001",
        "550e8400-e29b-41d4-a716-446655440002",
        "550e8400-e29b-41d4-a716-446655440003"
    ],
    "age": [20, 21, 22, 23],
    "gpa": [3.4, 3.7, 3.1, 3.8]
})

feature_engineer = FeatureEngineer(df)

result = feature_engineer.detect_identifier_features()

print(result)