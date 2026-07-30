# Dataset Loader

import pandas as pd
from pathlib import Path

class DatasetLoader:
    """Loads datasets and provides basic info for Project Atlas."""
    
    SUPPORTED_FILE_TYPES = {
        ".csv",
        ".json",
        ".xlsx",
    }
 
    def __init__(self):
        """Initialize an empty DatasetLoader."""
        self.dataset = None
        self.filename = None
        

    def load(self, filename: str):

        '''Load a supported dataset into memory.'''

        # saves the filename to self.filename
        self.filename = Path(filename)
        self.validate_file()

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
        """Validates that the file exists, is readable, is not empty, and has supported extension."""
        # whether a file is provided
        if self.filename is None:
            raise ValueError("No file has been provided.")
        # if the file exists
        if not self.filename.exists():
            raise FileNotFoundError(
                f"File does not exist: {self.filename}"
            )
        # if the path is a file
        if not self.filename.is_file():
            raise ValueError(f"Path is not a file: {self.filename}")

        extension = self.detect_file_type()
        # whether it is a supported extension
        if extension not in self.SUPPORTED_FILE_TYPES:
            raise ValueError(f'Unsupported file type: {extension}')
        # if the file is empty
        if self.filename.stat().st_size == 0:
            raise ValueError(f"File is empty: {self.filename}")

        return True 

    def get_summary(self):
        """Return a summary of the loaded dataset."""

        if self.dataset is None:
            raise ValueError("No dataset has been loaded.")

        return {
            "filename" : self.filename.name,
            "file_type" : self.detect_file_type(),
            "shape" : self.get_shape(),
            "columns" : self.get_column_names(),
            "memory_usage_bytes" : int(self.get_memory_usage()),
        }