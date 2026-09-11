from atlas.data.dataset_loader import DatasetLoader
from atlas.data.preprocessor import Preprocessor

loader = DatasetLoader()
loader.load("datasets/sample.csv")
preprocessor = Preprocessor(loader.dataset)


preprocessor.handle_missing_values({
    "age" : "median",
    "income" : "median"
})

result = preprocessor.scale_numerical({
    "age" : "standardize",
    "income" : "normalize"
})

print(result[["age", "income"]])
print(preprocessor.scalers)