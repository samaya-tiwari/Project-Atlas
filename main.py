import pandas as pd
from atlas.data.feature_engineer import FeatureEngineer

df = pd.DataFrame({
    "status": [
        "active",
        "active",
        "active",
        "active",
        "active",
        "inactive",
        "active"
    ],
    "age": [20, 21, 22, 23, 24, 25, 26]
})

feature_engineer = FeatureEngineer(df)

result = feature_engineer.detect_near_constant_feature(
    threshold=0.80
)

print(result)