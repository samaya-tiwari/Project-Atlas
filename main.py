# test

from atlas.data.dataset_loader import DatasetLoader

loader = DatasetLoader()
dataset= loader.load("datasets/sample.csv")

print(loader.validate_file())
print(loader.get_summary())