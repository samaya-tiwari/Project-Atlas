import pandas as pd

class Preprocessor:
    """Cleans and prepares datasets for analysis and machine learning."""

    MISSING_VALUE_STRATEGIES = {
        "mean",
        "median",
        "mode",
        "drop",
        "constant"
    }

    CATEGORICAL_ENCODING_STRATEGIES = { 
        "label", "one_hot"
    }

    def __init__(self, dataset):
        self.dataset = dataset.copy()
        self.label_mapping = {}

    def handle_missing_values(self, strategies, max_drop_percentage = 20, constant_values=None):
        # check whether the dataset is missing/empty
        if self.dataset is None or self.dataset.empty:
            return {}

        if not isinstance(strategies, dict):
            raise TypeError("strategies must be a dictionary.")

        if constant_values is None:
            constant_values = {}

        for column, strategy in strategies.items():

            # check if column exists
            if column not in self.dataset.columns:
                raise ValueError(f"Column '{column}' does not exist in the dataset.")

            # check if strategy is supported
            if strategy not in self.MISSING_VALUE_STRATEGIES:
                raise ValueError(
                    f"Unsupported missing-value strategy '{strategy}' for column '{column}'. "
                    "Supported strategies: mean, median, mode, drop, constant."
                )

            # validation: check if strategy requires numeric data
            if strategy in ['mean', 'median']:

                # verify that the column is numerical
                if not pd.api.types.is_numeric_dtype(self.dataset[column]):
                    raise ValueError(
                        f"Cannot apply '{strategy}' strategy to column '{column}'. "
                        f"The column data type is '{self.dataset[column].dtype}', but it must be numeric."
                    )

        for column, strategy in strategies.items():

            if strategy == 'drop':
            
                # calculate the missing percentage 
                missing_percentage = (self.dataset[column].isna().sum() / len(self.dataset)) * 100

                if missing_percentage > max_drop_percentage:
                    raise ValueError(f"Cannot drop rows for column '{column}'. "
                                    f"{missing_percentage:.2f}% of the values are missing, "
                                    f"which exceeds the allowed threshold of {max_drop_percentage}%.")

                self.dataset = self.dataset.dropna(subset=[column])
                continue

            if strategy == "mean":
                fill_value = self.dataset[column].mean()

            elif strategy == 'median':
                fill_value = self.dataset[column].median()

            elif strategy == 'mode':
                mode_series = self.dataset[column].mode()
                if mode_series.empty:
                    raise ValueError(
                        f"Cannot use mode strategy for column '{column} '"
                        "because the column has no non-missing values."
                    )

                fill_value = mode_series.iloc[0]

           
            elif strategy == "constant":
                if column not in constant_values:
                    raise ValueError(
                        f"No constant value provided for column '{column}'."
                    )

                fill_value = constant_values[column]

            self.dataset[column] = self.dataset[column].fillna(fill_value)

        return self.dataset

    def remove_duplicates(self):
        """Removes the duplicates in the dataset while keeping the first element."""
        if self.dataset is None or self.dataset.empty:
            return {}

        # counts duplicates before removal
        duplicate_count = self.dataset.duplicated(keep="first").sum()

        # remove duplicate rows keeping the first occurence
        self.dataset = self.dataset.drop_duplicates(keep="first")

        return self.dataset

    def encode_categorical(self, strategies):
        """Encodes columns using specified categorical encoding strategies."""

        # base validatoin check for the dataset
        if self.dataset is None or self.dataset.empty:
            raise ValueError("Dataset is empty or not initialized.")

        if not isinstance(strategies, dict):
            raise TypeError(f"Strategies must be a dictionary.")

        # validation loop
        for column, strategy in strategies.items():

            if column not in self.dataset.columns:
                raise ValueError (f"Column '{column}' does not exist in the dataset.")

            # checking if the strategy is supported
            if strategy not in self.CATEGORICAL_ENCODING_STRATEGIES:
                raise ValueError(f"Unknown encoding strategy '{strategy}' for column '{column}'. "
                                 f"Supported strategies are: {self.CATEGORICAL_ENCODING_STRATEGIES}"
                                 )

            # verify the column is categorical
            if pd.api.types.is_numeric_dtype(self.dataset[column]):
                raise ValueError(
                    f"Cannot apply categorical encoding to column '{column}'. "
                    f"The column is numeric, but must be categorical."
                )

            # execution loop
        for column, strategy in strategies.items():
            if strategy == "label":

                categories = self.dataset[column].astype("category").cat.categories

                mapping = {
                    category : code
                    for code, category in enumerate(categories)
                }

                self.label_mapping[column] = mapping
                self.dataset[column] = self.dataset[column].map(mapping)

            elif strategy == "one_hot":

                encoded_cols = pd.get_dummies(self.dataset[column], prefix=column, dtype=int)

                self.dataset = pd.concat([self.dataset, encoded_cols], axis=1)
                self.dataset = self.dataset.drop(columns=[column])

        return self.dataset