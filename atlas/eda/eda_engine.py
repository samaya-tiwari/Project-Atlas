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
        percentage = round((total_rows/row_indices) * 100)

        return {
            "count" : count,
            "percentage" : percentage,
            "row_indices" : row_indices
        }


    def get_data_types(self):

        TYPE_MAPPING = {
            "int64": "integer",
            "int32": "integer",
            "float64": "float",
            "float32": "float",
            "bool": "boolean",
            "datetime64[ns]": "datetime",
            "timedelta64[ns]": "timedelta",
            "object": "text",
            "category": "categorical" 
        }

        if self.dataset is None or self.dataset.empty:
            return {}

        data_types = {}

        for column in self.dataset.columns:
            dtype_str = str(self.dataset[column].dtype)
            semantic_type = TYPE_MAPPING.get(dtype_str, "Unknown / Mixed")

            data_types[column] = {
                "dtype": dtype_str,
                "semantic_type": semantic_type
            }

        return data_types

        
    def get_unique_values(self):

        if self.dataset is None or self.dataset.empty:
            return {}

        unique_vals = {}

        for column in self.dataset.columns:
            count = self.dataset[column].nunique()

            unique_vals[column] = {
                    "unique_count": int(count)
                }
        return unique_vals
    
    def get_numerical_summary(self):

        if self.dataset is None or self.dataset.empty:
            return {}

        numerical_summary = {}

        numeric_data = self.dataset.select_dtypes(include='number')
        if numeric_data.empty:
            return {}
        describe_stats = numeric_data.describe()

        for column in numeric_data.columns:
            count = describe_stats.loc["count", column]
            mean = describe_stats.loc["mean", column]
            minimum = describe_stats.loc["min", column]
            std = describe_stats.loc["std", column]
            q1 = describe_stats.loc["25%", column]
            median = describe_stats.loc["50%", column]
            q3 = describe_stats.loc["75%", column]
            maximum = describe_stats.loc["max", column]

            variance = numeric_data[column].var()
            skewness = numeric_data[column].skew()
            kurtosis = numeric_data[column].kurt()
            iqr = describe_stats.loc["75%", column] - describe_stats.loc["25%", column]
        
            numerical_summary[column] =  {
                "count": int(count),
                "mean": float(mean),
                "min": float(minimum),
                "std": float(std),
                "25%": float(q1),
                "median": float(median),
                "75%": float(q3),
                "max": float(maximum),
                "variance": float(variance),
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "iqr": float(iqr),
            }

        return numerical_summary
       
    def get_categorical_summary(self):
        categorical_cols = self.dataset.select_dtypes(include= ["object", "category", "bool"]).columns.tolist()

        if not categorical_cols:
            categorical_dict = {}
        else:
            categorical_dict = {
                col: {
                    "count": int(self.dataset[col].count()),
                    "unique_count":int(self.dataset[col].nunique()),
                    "most_frequent": self.dataset[col].mode().iloc[0] if not self.dataset[col].mode().empty else None,
                    "frequency" : int(self.dataset[col].value_counts().iloc[0]) if not self.dataset[col].value_counts().empty else 0,
                    "missing_count": int(self.dataset[col].isna().sum())
                }
                for col in categorical_cols
            }
            return categorical_dict
    def get_correlation_matrix(self, dataset):
        pass
    def get_dimensionality(self, dataset):
        pass
    
    def get_summary(self, dataset):
        pass