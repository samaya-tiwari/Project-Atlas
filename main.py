import pandas as pd
from atlas.data.feature_engineer import FeatureEngineer

df = pd.DataFrame({
    "enrollment_date": pd.to_datetime([
        "2024-08-20",
        "2024-08-24",
        "2025-01-15"
    ])
})


feature_engineer = FeatureEngineer(df)

result = feature_engineer.create_datetime_features(
    columns=["enrollment_date"],
    features=[
        "year",
        "month",
        "day",
        "dayofweek",
        "quarter",
        "is_weekend"
    ]
)

print(result)