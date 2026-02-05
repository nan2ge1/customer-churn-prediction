# --------------------------------------------------------------------------- #
# config.py – Central place for constants used across the pipeline.
# --------------------------------------------------------------------------- #

DATA_PATH = "data/customer_churn.csv"

# Column that uniquely identifies each customer (dropped before modelling)
ID_COL = "CustomerID"

# Target variable
TARGET = "Churn"

# Feature groups
CATEGORICAL_FEATURES = ["ContractType", "PaymentMethod"]
NUMERICAL_FEATURES = [
    "ContractDuration",
    "MonthlyCharges",
    "InternetUsageGB",
    "SupportCalls",
]

# Train/test split settings
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Model hyper-parameters
N_ESTIMATORS = 1000

# Model output
MODEL_DIR = "models"
