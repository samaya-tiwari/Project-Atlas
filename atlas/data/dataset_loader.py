# Dataset Loader

import pandas as pd

class DatasetLoader:
    """Loads datasets and provides basic info for Project Atlas."""
    

    """Initialize an empty DatasetLoader."""
    def __init__(self):
        self.dataset = None
        self.filename = None
        

    def load(self, filename: str):

        '''Load a CSV dataset.'''

        # saves the filename to self.filename
        self.filename = filename
        self.dataset = pd.read_csv(filename)

        # returns self.dataset
        return self.dataset
    
    
    def get_shape(self):
        return self.dataset.shape     # returns the shape of the data
    
    
    
    def get_column_names(self):
        pass
    def get_memory_usage(self):
        pass
    def detect_file_type(self):
        pass
    def validate_file(self):
        pass
    def get_summary(self):
        pass