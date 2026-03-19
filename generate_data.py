"""
generate_data.py
----------------
Generates a realistic synthetic customer churn dataset.
Run this FIRST if you don't have your own CSV file.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 2000

age              = np.random.randint(18, 70, N)
tenure_months    = np.random.randint(1, 72, N)
monthly_charges  = np.round(np.random.uniform(20, 120, N), 2)
total_charges    = np.round(monthly_charges * tenure_months + np.random.normal(0, 50, N), 2)
num_products     = np.random.choice([1, 2, 3, 4], N, p=[0.4, 0.35, 0.15, 0.10])
support_calls    = np.random.poisson(2, N)
satisfaction     = np.random.choice([1, 2, 3, 4, 5], N, p=[0.10, 0.15, 0.30, 0.30, 0.15])
contract_type    = np.random.choice(["Month-to-Month", "One Year", "Two Year"], N, p=[0.55, 0.25, 0.20])
payment_method   = np.random.choice(["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"], N)
internet_service = np.random.choice(["DSL", "Fiber Optic", "No"], N, p=[0.34, 0.44, 0.22])

# Churn probability influenced by real-world factors
churn_score = (
    0.3 * (monthly_charges / 120)
    - 0.25 * (tenure_months / 72)
    + 0.2 * (support_calls / 10)
    - 0.15 * (satisfaction / 5)
    + 0.1 * (num_products == 1).astype(float)
    + 0.15 * (contract_type == "Month-to-Month").astype(float)
    + np.random.normal(0, 0.1, N)
)
churn_prob  = 1 / (1 + np.exp(-5 * (churn_score - 0.3)))
churn       = (np.random.rand(N) < churn_prob).astype(int)

# Inject ~5% missing values into two columns
for col_arr in [monthly_charges, support_calls]:
    mask = np.random.rand(N) < 0.05
    col_arr = col_arr.astype(float)
    col_arr[mask] = np.nan

df = pd.DataFrame({
    "Age":              age,
    "Tenure_Months":    tenure_months,
    "Monthly_Charges":  monthly_charges,
    "Total_Charges":    total_charges,
    "Num_Products":     num_products,
    "Support_Calls":    support_calls,
    "Satisfaction_Score": satisfaction,
    "Contract_Type":    contract_type,
    "Payment_Method":   payment_method,
    "Internet_Service": internet_service,
    "Churn":            churn,
})

out = Path(__file__).parent / "customer_churn.csv"
df.to_csv(out, index=False)
print(f"✅  Dataset saved → {out}  ({N} rows, churn rate: {churn.mean():.1%})")
