import pandas as pd
from atlas.data.preprocessor import Preprocessor

df = pd.DataFrame({
    "name": ["Samaya", "Swastika", "Aaron", "Sri"],
    "city": ["London", "Paris", "Tokyo", "London"]
})

preprocessor = Preprocessor(df)

result = preprocessor.encode_categorical({
    "city" : "one_hot"
})

print(result)
print(preprocessor.label_mapping)