# House Price Linear Regression

A house-price prediction project with a NumPy implementation of linear
regression trained using gradient descent. The training pipeline handles
missing numeric values, standardizes features and the target, evaluates
predictions, and saves the trained model with its preprocessor.

## Requirements

- Python 3.10 or later
- Packages listed in `requirements.txt`

## Setup

From the project root, create and activate a virtual environment, then install
the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The training dataset is expected at `data/house_predict.csv` and uses
`SalePrice` as its target column.

## Train the model

Run the training module from the project root:

```powershell
python -m src.training.train
```

The script prints train and test metrics and writes the model artifact to
`artifacts/linear_regression.joblib`.

## Run cross-validation

Cross-validation uses five shuffled folds by default. Each fold fits its own
preprocessor on that fold's training data to avoid data leakage:

```python
from src.core.config import FEATURE_COLUMNS_TO_DROP
from src.data.loader import load_dataset
from src.evaluations.cross_validation import cross_validate_regression

X, y = load_dataset()
X = X.select_dtypes(include="number").drop(
    columns=FEATURE_COLUMNS_TO_DROP,
    errors="ignore",
)
results = cross_validate_regression(X, y)
print(results["summary"])
```

## Notebooks

The `notebook/` directory contains the project notebooks for training and
exploratory house-price analysis.

## Project structure

```text
data/                 Training dataset
notebook/             Jupyter notebooks
src/core/             Project configuration
src/data/             Dataset loading
src/evaluations/      Regression metrics and cross-validation
src/models/           Linear regression implementation
src/preprocessing/    Missing-value handling and standardization
src/training/         Training entry point
```

Generated model artifacts, Python bytecode, virtual environments, and Jupyter
checkpoint directories are excluded by `.gitignore`.
