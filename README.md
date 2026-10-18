# Project Atlas

> An AI-assisted data science platform for automating dataset analysis, preprocessing, feature engineering, model selection, evaluation, explainability, and reporting.

## 🚧 Project Status

**Atlas is currently under active development.**

The goal of Atlas is to build an AI Data Scientist that can inspect a dataset, identify data-quality issues, prepare the data for machine learning, recommend appropriate modeling approaches, evaluate models, and eventually expose these capabilities to AI assistants through MCP.

## Current Pipeline

```text
Dataset
   ↓
Dataset Loader          ✅
   ↓
EDA Engine              ✅
   ↓
Preprocessing Engine    ✅
   ↓
Feature Engineering     🚧
   ↓
Model Recommendation    ⏳
   ↓
Model Training          ⏳
   ↓
Evaluation              ⏳
   ↓
Explainability          ⏳
   ↓
Report Generation       ⏳
   ↓
MCP Integration         ⏳
```

## Features Implemented
### Dataset Loading
- CSV loading
- JSON loading
- Excel loading
- File validation
- Dataset metadata and summary
### Exploratory Data Analysis
- Missing-value analysis
- Duplicate-row analysis
- Data-type analysis
- Unique-value analysis
- Numerical summaries
- Categorical summaries
- Correlation analysis
- Dataset dimensionality
- Aggregated EDA summary
### Data Preprocessing
- Missing-value handling
  - Mean
  - Median
  - Mode
  - Constant replacement
  - Controlled row dropping
- Duplicate removal
- Categorical encoding
  - Label encoding
  - One-hot encoding
- Numerical scaling
  - StandardScaler
  - MinMaxScaler
- Datetime conversion
- Inconsistent-data cleaning
- Config-driven preprocessing pipeline
### Feature Engineering
Currently in development.

Implemented so far:
- Explicit feature removal
- Constant-feature detection
- Near-constant feature detection
- Highly correlated feature detection
- Identifier-like feature detection
- Redundant identifier detection

## Example
```
from atlas.data.dataset_loader import DatasetLoader
from atlas.data.preprocessor import Preprocessor

loader = DatasetLoader()
loader.load("datasets/sample.csv")

preprocessor = Preprocessor(loader.dataset)

config = {
    "missing_values": {
        "strategies": {
            "age": "median",
            "income": "median"
        }
    },

    "remove_duplicates": True,

    "numerical_scaling": {
        "strategies": {
            "age": "standardize",
            "income": "normalize"
        }
    }
}

cleaned_data = preprocessor.preprocess(config)
```


## Project Structure
```
atlas/
├── data/           # Dataset loading, preprocessing, feature engineering
├── eda/            # Exploratory data analysis
├── models/         # Model training and recommendation
├── evaluation/     # Model evaluation
├── visualization/  # Visualization utilities
├── mcp/            # MCP integration
└── utilities/      # Shared utilities
```

## Roadmap
Atlas is being developed incrementally.
Current focus:
#### Feature Engineering and Feature Selection
Future development will include:
- Automated model recommendation
- Classification and regression training
- Model comparison
- Feature importance
- Explainability
- Automated reports
- MCP tools for AI assistants
## Technologies
- Python
- pandas
- NumPy
- scikit-learn
Additional technologies will be introduced as the project grows.
## Motivation
Data scientists repeatedly perform many of the same tasks when working with new datasets: inspecting data quality, cleaning missing values, encoding features, scaling numerical variables, comparing models, and interpreting results.
Atlas aims to provide a structured system that automates these repetitive parts of the workflow while keeping decisions transparent and configurable.