import os
import pandas as pd
from main import load_datasets, parse_message_overrides, apply_message_overrides, evaluate_request

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)

requests, profiles, events, exchange_rates, payment_options, messages, images = load_datasets(os.path.join(BASE_DIR, "dataset"))
sample_df = pd.read_csv(os.path.join(BASE_DIR, "dataset", "sample_requests.csv"))

overrides = parse_message_overrides(messages)
print("\n--- REQUEST 02 DEBUG ---")

req = sample_df[sample_df["request_id"] == "request_03"].iloc[0]

print("Request:")
print(req.to_dict())

print("\nPayment options:")
print(
    payment_options[
        payment_options["request_id"] == "request_03"
    ].to_string(index=False)
)
events_updated = apply_message_overrides(events, overrides)

preds = []
for idx, req in sample_df.iterrows():
    res = evaluate_request(req, profiles, events_updated, exchange_rates, payment_options)
    preds.append(res)

pred_df = pd.DataFrame(preds)

merged = sample_df.merge(pred_df, on="request_id", suffixes=("_truth", "_pred"))

print(f"--- Benchmark Validation on {len(merged)} Sample Requests ---")

# Status match
status_match = (merged['affordability_status_truth'] == merged['affordability_status_pred']).sum()
print(f"Affordability Status Accuracy: {status_match}/{len(merged)} ({status_match/len(merged)*100:.1f}%)")

# Method match
method_match = (merged['recommended_payment_method_truth'] == merged['recommended_payment_method_pred']).sum()
print(f"Payment Method Accuracy: {method_match}/{len(merged)} ({method_match/len(merged)*100:.1f}%)")

# Earliest Date match
date_match = (merged['earliest_date_for_full_payment_truth'].fillna('') == merged['earliest_date_for_full_payment_pred'].fillna('')).sum()
print(f"Earliest Date Accuracy: {date_match}/{len(merged)} ({date_match/len(merged)*100:.1f}%)")

print("\n--- First 5 Mismatches (if any) ---")
mismatches = merged[merged['affordability_status_truth'] != merged['affordability_status_pred']]
cols = ['request_id', 'affordability_status_truth', 'affordability_status_pred', 'recommended_payment_method_truth', 'recommended_payment_method_pred']
print(mismatches[cols].head(5).to_string(index=False))

print("\n--- ALL MISMATCHES ---")
print(merged[
    (merged['affordability_status_truth'] != merged['affordability_status_pred']) |
    (merged['recommended_payment_method_truth'] != merged['recommended_payment_method_pred']) |
    (merged['earliest_date_for_full_payment_truth'].fillna('') != merged['earliest_date_for_full_payment_pred'].fillna(''))
][[
    'request_id',
    'affordability_status_truth',
    'affordability_status_pred',
    'recommended_payment_method_truth',
    'recommended_payment_method_pred',
    'earliest_date_for_full_payment_truth',
    'earliest_date_for_full_payment_pred'
]].to_string(index=False))
