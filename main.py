import pandas as pd

from atlas.data.feature_engineer import FeatureEngineer


df = pd.DataFrame({
    "height_cm": [160, 165, 170, 175, 180, 185],
    "height_inches": [63.0, 65.0, 66.9, 68.9, 70.9, 72.8],
    "age": [18, 25, 32, 41, 55, 67],
    "years_until_retirement": [47, 40, 33, 24, 10, -2],
    "score": [72, 88, 65, 91, 77, 84]
})

feature_engineer = FeatureEngineer(df)

result = feature_engineer.detect_high_correlation_features(
    threshold=0.9999
)


print(result)
