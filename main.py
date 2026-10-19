import pandas as pd
from pprint import pprint

from atlas.data.feature_engineer import FeatureEngineer


df = pd.DataFrame({
    # Identifier-like feature
    "student_id": list(range(1001, 1021)),

    # Second identifier-like feature (UUID)
    "uuid": [
        f"550e8400-e29b-41d4-a716-{i:012d}"
        for i in range(20)
    ],

    # Constant feature
    "school": ["USM"] * 20,

    # Near-constant feature: 19/20 = 95%
    "status": [
        "active", "active", "active", "active", "active",
        "active", "active", "active", "active", "active",
        "active", "active", "active", "active", "active",
        "active", "active", "active", "active", "inactive"
    ],

    # Highly correlated numerical features
    "height_cm": [
        160, 162, 164, 166, 168,
        170, 172, 174, 176, 178,
        180, 182, 184, 186, 188,
        190, 192, 194, 196, 198
    ],

    "height_inches": [
        63.0, 63.8, 64.6, 65.4, 66.1,
        66.9, 67.7, 68.5, 69.3, 70.1,
        70.9, 71.7, 72.4, 73.2, 74.0,
        74.8, 75.6, 76.4, 77.2, 78.0
    ]
})


feature_engineer = FeatureEngineer(df)

result = feature_engineer.get_drop_candidates()

print("===== DROP CANDIDATES =====")
pprint(result)