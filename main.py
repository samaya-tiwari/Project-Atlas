import pandas as pd

from atlas.data.preprocessor import Preprocessor

df = pd.DataFrame({
    "enrollment_date": [
        "2024-08-20",
        "08/21/2024",
        "not-a-date",
        "23-Aug-2024"
    ]
})

preprocessor = Preprocessor(df)

result = preprocessor.prepare_datetime([
    "enrollment_date"
])

print(result)
print(result.dtypes)