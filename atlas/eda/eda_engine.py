class EDAEngine:
    """Analyze the loaded dataset."""

    def __init__(self, dataset):
        self.dataset = dataset
    
    def get_missing_values(self):
        missing_counts = self.dataset.isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        missing_counts = missing_counts.sort_values(ascending=False)

        missing_info = {}

        for column, count in missing_counts.items():
            percentage = (count / len(self.dataset)) * 100

            missing_info[column] = {
                "count" : int(count),
                "percentage" : round(float(percentage), 2)
            }

        return missing_info

    
    def get_duplicate_rows(self):

        if self.dataset is None or self.dataset.empty:
            return {"count" : 0, "percentage" : 0.0, "row_indices" : []}
        
        duplicated_info = self.dataset.duplicated(keep="first")  # this ignores the first occurance

        row_indices = self.dataset.index[duplicated_info].tolist()

        count = len(row_indices)

        total_rows = len(self.dataset)
        percentage = float((total_rows/row_indices) * 100)

        return {
            "count" : count,
            "percentage" : percentage,
            "row_indices" : row_indices
        }


    def get_data_types(self, dataset):
        pass
    def get_unique_values(self, dataset):
        pass
    def get_numerical_summary(self, dataset):
        pass
    def get_categorical_summary(self, dataset):
        pass
    def get_correlation_matrix(self, dataset):
        pass
    def get_dimensionality(self, dataset):
        pass
    
    def get_summary(self, dataset):
        pass