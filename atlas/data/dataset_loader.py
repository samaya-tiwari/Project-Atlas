# Dataset Loader

import pandas as pd
from pathlib import Path

class DatasetLoader:
    """Loads datasets and provides basic info for Project Atlas."""
    

 
    def __init__(self):
        """Initialize an empty DatasetLoader."""
        self.dataset = None
        self.filename = None
        

    def load(self, filename: str):

        '''Load a supported dataset into memory.'''

        # saves the filename to self.filename
        self.filename = Path(filename)
        extension = self.detect_file_type()

        if extension == ".csv":
            self.dataset = pd.read_csv(self.filename)
        elif extension == ".json":
            self.dataset = pd.read_json(self.filename)
        elif extension == ".xlsx":
            self.dataset = pd.read_excel(self.filename)
        else:
            raise ValueError("Unspported file type")

        # returns self.dataset
        return self.dataset
    
    
    def get_shape(self):
        return self.dataset.shape     # returns the shape of the data
    
    
    def get_column_names(self):
        """Return the dataset column names in a list."""
        ind_list = self.dataset.columns
        return ind_list.to_list()
        
    
    def get_memory_usage(self):
        """Return the total memory usage of the dataset."""
        return self.dataset.memory_usage(index=True, deep=True).sum()


    def detect_file_type(self):
        """Return the file extension of the loaded dataset."""
        return self.filename.suffix.lower()


        
    def validate_file(self):
        pass
    def get_summary(self):
        pass