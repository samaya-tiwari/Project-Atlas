import pandas as pd

from atlas.data.preprocessor import Preprocessor

df = pd.DataFrame({
    "name": ["Sam", "Alex", "Sam"],
    "age": [20, 22, 20]
})

preprocessor = Preprocessor(df)

print("BEFORE:")
print(preprocessor.dataset)

result = preprocessor.remove_duplicates()

print("\nAFTER:")
print(result)