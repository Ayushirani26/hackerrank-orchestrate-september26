import os
import re
import pandas as pd
from datetime import datetime, timedelta

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

def load_datasets(dataset_dir=DATASET_DIR):
    requests = pd.read_csv(os.path.join(dataset_dir, "requests.csv"))
    profiles = pd.read_csv(os.path.join(dataset_dir, "financial_profiles.csv"))
    events = pd.read_csv(os.path.join(dataset_dir, "financial_events.csv"))
    exchange_rates = pd.read_csv(os.path.join(dataset_dir, "exchange_rates.csv"))
    payment_options = pd.read_csv(os.path.join(dataset_dir, "request_payment_options.csv"))
    messages = pd.read_csv(os.path.join(dataset_dir, "messages.csv"))
    images = pd.read_csv(os.path.join(dataset_dir, "images.csv"))
    return requests, profiles, events, exchange_rates, payment_options, messages, images

def fill_missing_amounts_from_images(events_df, images_df):
    events_copy = events_df.copy()
    if 'image_id' in events_copy.columns and not images_df.empty:
        img_map = dict(zip(images_df['image_id'], images_df['extracted_amount']))
        for idx, row in events_copy.iterrows():
            if pd.isna(row['amount']) and pd.notna(row.get('image_id')):
                extracted = img_map.get(row['image_id'])
                if pd.notna(extracted):
                    events_copy.at[idx, 'amount'] = extracted
    return events_copy

def convert_currency(amount, from_curr, to_curr, rate_date, exchange_rates_df):
    if from_curr == to_curr or pd.isna(amount):
        return amount
    match = exchange_rates_df[
        (exchange_rates_df['from_currency'] == from_curr) & 
        (exchange_rates_df['to_currency'] == to_curr) & 
        (exchange_rates_df['rate_date'] == rate_date)
    ]
    if not match.empty:
        return amount * match.iloc[0]['rate']
    match_fallback = exchange_rates_df[
        (exchange_rates_df['from_currency'] == from_curr) & 
        (exchange_rates_df['to_currency'] == to_curr)
    ]
    if not match_fallback.empty:
        return amount * match_fallback.iloc[0]['rate']
    return amount

def parse_message_overrides(messages_df):
    overrides = {}
    for idx, row in messages_df.iterrows():
        rel_event = row['related_event_id']
        text = str(row['message_text']).lower()
        if pd.notna(rel_event):
            if rel_event not in overrides:
                overrides[rel_event] = {}
            if any(w in text for w in ['cancel', 'cancelled', 'refunded', 'void', 'waived']):
                overrides[rel_event]['status'] = 'cancelled'
            elif any(w in text for w in ['paid', 'settled', 'completed']):
                overrides[rel_event]['status'] = 'settled'
    return overrides

def apply_message_overrides(events_df, overrides):
    events_copy = events_df.copy()
    for event_id, override in overrides.items():
        mask = events_copy['event_id'] == event_id
        for col, val in override.items():
            events_copy.loc[mask, col] = val
    return events_copy

def parse_frequency(freq_val):
    if pd.isna(freq_val):
        return 30
    if isinstance(freq_val, (int, float)):
        return int(freq_val)
    freq_str = str(freq_val).strip().lower()
    if 'month' in freq_str:
        return 30
    if 'week' in freq_str and 'bi' in freq_str:
        return 14
    if 'week' in freq_str:
        return 7
    if 'day' in freq_str:
        return 1
    try:
        return int(float(freq_str))
    except ValueError:
        return 30
def simulate_cashflow(user_id, start_date_str, profiles, events, exchange_rates,
                      payment_schedule=None, disabled_event_ids=None):

    user_profile = profiles[profiles['user_id'] == user_id].iloc[0]

    initial_balance = user_profile['current_available_balance']
    min_keep = user_profile['minimum_balance_to_keep']
    home_curr = user_profile['home_currency']

    disabled_set = set(disabled_event_ids) if disabled_event_ids else set()

    user_events = events[events['user_id'] == user_id].copy()

    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")

    daily_balances = {}
    current_balance = initial_balance
    lowest_balance = initial_balance

    sched_map = payment_schedule if payment_schedule else {}

    for day in range(91):

        cur_dt = start_dt + timedelta(days=day)
        cur_date_str = cur_dt.strftime("%Y-%m-%d")

        if cur_date_str in sched_map:
            current_balance -= sched_map[cur_date_str]

        day_events = user_events[
            (
                (
                    (user_events['event_date'] == cur_date_str) &
                    (user_events['event_type'] != 'refund')
                )
                |
                (
                    (user_events['event_type'] == 'refund') &
                    (user_events['status'] == 'pending') &
                    (user_events['settlement_date'] == cur_date_str)
                )
                |
                (
                    (user_events['event_type'] == 'refund') &
                    (user_events['status'] != 'pending') &
                    (user_events['event_date'] == cur_date_str)
                )
            )
            & (user_events['status'] != 'cancelled')
        ]

        for _, ev in day_events.iterrows():

            if ev['event_id'] in disabled_set:
                continue

            amt = ev['amount']

            if pd.isna(amt):
                continue

            amt_home = convert_currency(
                amt,
                ev['currency'],
                home_curr,
                cur_date_str,
                exchange_rates
            )

            if ev['direction'] == 'inflow' or ev['event_type'] == 'income':
                current_balance += amt_home

            elif ev['direction'] == 'outflow':
                current_balance -= amt_home

        daily_balances[cur_date_str] = current_balance

        if current_balance < lowest_balance:
            lowest_balance = current_balance

    return daily_balances, lowest_balance, min_keep

def compute_safe_pay_and_earliest_date(req_row, daily_balances, min_keep):
    req_amt = float(req_row['requested_amount'])

    min_balance = min(daily_balances.values())
    amount_safe_to_pay = max(0.0, min(req_amt, min_balance - min_keep))

    earliest_date = ""
    sorted_dates = sorted(daily_balances.keys())

    for d_str in sorted_dates:
        balance = daily_balances[d_str]

        if balance - req_amt >= min_keep:
            dt = datetime.strptime(d_str, "%Y-%m-%d")

            if dt.day <= 15:
                earliest_date = dt.replace(day=15).strftime("%Y-%m-%d")
            else:
                next_month = dt.replace(day=28) + timedelta(days=4)
                earliest_date = next_month.replace(day=15).strftime("%Y-%m-%d")

            break

    if earliest_date == "":
        desired_date = str(req_row['desired_completion_date'])

        if desired_date in daily_balances:
            if daily_balances[desired_date] - req_amt >= min_keep:
                earliest_date = desired_date

    return round(amount_safe_to_pay, 2), earliest_date

def build_installment_plan(opt_row):
    num_p = int(opt_row['number_of_payments']) if pd.notna(opt_row['number_of_payments']) else 1
    p_amt = float(opt_row['payment_amount']) if pd.notna(opt_row['payment_amount']) else 0.0

    freq = parse_frequency(opt_row.get('payment_frequency_days', 30))

    start_date_val = (
        str(opt_row['first_payment_date'])
        if pd.notna(opt_row['first_payment_date'])
        else "2026-09-01"
    )

    try:
        start_dt = datetime.strptime(start_date_val, "%Y-%m-%d")
    except ValueError:
        start_dt = datetime.strptime("2026-09-01", "%Y-%m-%d")

    plan_parts = []
    sched = {}

    for i in range(num_p):
        cur_dt = start_dt + timedelta(days=i * freq)
        dt_str = cur_dt.strftime("%Y-%m-%d")

        plan_parts.append(f"{dt_str}:{p_amt:.2f}")
        sched[dt_str] = sched.get(dt_str, 0.0) + p_amt

    return "|".join(plan_parts), sched

def evaluate_request(req_row, profiles, events, exchange_rates, payment_options):
    u_id = req_row['user_id']
    req_id = req_row['request_id']
    profile_row = profiles[profiles['user_id'] == u_id].iloc[0]
    
    daily_balances, lowest_bal, min_keep = simulate_cashflow(
        u_id, req_row['request_date'], profiles, events, exchange_rates
    )
    
    safe_pay, earliest_full_date = compute_safe_pay_and_earliest_date(
        req_row, daily_balances, min_keep
    )
    
    allowed_methods = [m.strip() for m in re.split(r'[,|]', str(profile_row['payment_methods_user_will_consider']))]
    print("\nMETHOD DEBUG")
    print("Request:", req_id)
    print("Allowed methods:", allowed_methods)
    req_amt = req_row['requested_amount']
    req_date = req_row['request_date']
    desired_comp_date = req_row['desired_completion_date']
    
    status = "not_affordable"
    rec_method = "not_recommended"
    plan = "none"
    spending_changes = "none"
    explanation = ""

   # Strategy 1: full_payment today
    if "full_payment" in allowed_methods and safe_pay >= req_amt:
        status = "affordable_now"
        rec_method = "full_payment"
        plan = f"{req_date}:{req_amt:.2f}"
        earliest_full_date = req_date
        explanation = f"Full payment of {req_amt:.2f} is safe on request date."

    # Strategy 2: installments options
    elif "installments" in allowed_methods:
        req_opts = payment_options[payment_options['request_id'] == req_id]

        for _, opt_row in req_opts.iterrows():
            opt_plan_str, opt_sched = build_installment_plan(opt_row)

            _, opt_lowest, _ = simulate_cashflow(
                u_id,
                req_date,
                profiles,
                events,
                exchange_rates,
                payment_schedule=opt_sched
            )

            if opt_lowest >= min_keep:
                status = "affordable_with_plan"
                rec_method = "installments"
                plan = opt_plan_str
                explanation = "Installment plan selected safely completing the request."
                break

    # Strategy 3: installments
    elif "installments" in allowed_methods and not (
    "partial_payment" in allowed_methods
    and req_row['allows_partial_payment']
    and safe_pay > 0
    and earliest_full_date != ""
    and earliest_full_date <= desired_comp_date):
       req_opts = payment_options[payment_options['request_id'] == req_id]

       for _, opt_row in req_opts.iterrows():
        opt_plan_str, opt_sched = build_installment_plan(opt_row)

        _, opt_lowest, _ = simulate_cashflow(
            u_id,
            req_date,
            profiles,
            events,
            exchange_rates,
            payment_schedule=opt_sched
        )

        if opt_lowest >= min_keep:
            status = "affordable_with_plan"
            rec_method = "installments"
            plan = opt_plan_str
            explanation = "Installment plan selected safely completing the request."
            break

    # Strategy 4: partial_payment
    if rec_method == "not_recommended" and "partial_payment" in allowed_methods and req_row['allows_partial_payment'] and safe_pay > 0 and earliest_full_date != "" and earliest_full_date <= desired_comp_date:
        status = "affordable_with_plan"
        rec_method = "partial_payment"
        rem_amt = req_amt - safe_pay
        plan = f"{req_date}:{safe_pay:.2f}|{earliest_full_date}:{rem_amt:.2f}"
        explanation = f"Partial payment recommended: pay safe amount {safe_pay:.2f} today, pay remaining on {earliest_full_date}."

    # Strategy 5: wait
    elif rec_method == "not_recommended" and "full_payment" in allowed_methods and earliest_full_date != "" and earliest_full_date <= desired_comp_date:
        status = "affordable_later"
        rec_method = "wait"
        plan = f"{earliest_full_date}:{req_amt:.2f}"
        explanation = f"Wait until {earliest_full_date} to make full payment safely."

    if rec_method == "not_recommended" and status == "not_affordable":
        status = "not_affordable"
        rec_method = "not_recommended"
        plan = "none"
        explanation = "Request cannot be safely completed by desired date while maintaining minimum balance."

    return {
        "request_id": req_id,
        "amount_safe_to_pay": safe_pay,
        "affordability_status": status,
        "recommended_payment_method": rec_method,
        "payment_plan": plan,
        "earliest_date_for_full_payment": earliest_full_date,
        "spending_changes_needed": spending_changes,
        "decision_explanation": explanation
    }

def main():
    requests, profiles, events, exchange_rates, payment_options, messages, images = load_datasets()
    events = fill_missing_amounts_from_images(events, images)
    overrides = parse_message_overrides(messages)
    events_updated = apply_message_overrides(events, overrides)
    
    results = [evaluate_request(req, profiles, events_updated, exchange_rates, payment_options) for _, req in requests.iterrows()]
    output_df = pd.DataFrame(results)
    output_path = os.path.join(BASE_DIR, "dataset", "output.csv")
    output_df.to_csv(output_path, index=False)
    print(f"Updated output saved to: {output_path}")

if __name__ == "__main__":
    main()