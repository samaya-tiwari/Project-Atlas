# test

from atlas.data.dataset_loader import DatasetLoader
from atlas.eda.eda_engine import EDAEngine

print("Starting test...")

loader = DatasetLoader()
loader.load("datasets/sample.csv")

eda = EDAEngine(loader.dataset)

result = eda.get_dimensionality()

print("Dimensionality: ")
print(result)

print("Summary: ")
print(eda.get_summary())