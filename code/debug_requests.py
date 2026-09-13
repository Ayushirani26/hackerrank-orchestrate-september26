import os
import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)

sample_df = pd.read_csv(os.path.join(BASE_DIR, "dataset", "sample_requests.csv"))

target_ids = ['request_25']
filtered = sample_df[sample_df['request_id'].isin(target_ids)]

for idx, row in filtered.iterrows():
    print(f"=== {row['request_id']} ===")
    print(f"User ID: {row['user_id']}")
    print(f"Req Amount: {row['requested_amount']} | Req Date: {row['request_date']} | Desired Date: {row['desired_completion_date']}")
    print(f"TRUTH Status: {row['affordability_status']}")
    print(f"TRUTH Method: {row['recommended_payment_method']}")
    print(f"TRUTH Plan: {row['payment_plan']}")
    print(f"TRUTH Safe Pay: {row['amount_safe_to_pay']}")
    print(f"TRUTH Earliest Date: {row['earliest_date_for_full_payment']}")
    print(f"TRUTH Spending Changes: {row['spending_changes_needed']}")
    print(f"TRUTH Spending Changes: {row['spending_changes_needed']}")
    print("-" * 50)