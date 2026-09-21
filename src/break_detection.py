import numpy as np
import pandas as pd


def cusum_detect(series, threshold_std=5.0, drift_std=0.5):
    """
    Two-sided CUSUM break detection on a 1D series.
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    if n < 30:
        return None, None

    mean = values.mean()
    std = values.std()
    if std == 0:
        return None, None

    drift = drift_std * std
    threshold = threshold_std * std

    pos_cusum = np.zeros(n)
    neg_cusum = np.zeros(n)

    for i in range(1, n):
        diff = values[i] - mean
        pos_cusum[i] = max(0, pos_cusum[i-1] + diff - drift)
        neg_cusum[i] = min(0, neg_cusum[i-1] + diff + drift)

    break_idx = None
    for i in range(n):
        if pos_cusum[i] > threshold or abs(neg_cusum[i]) > threshold:
            break_idx = i
            break

    return break_idx, {'pos_cusum': pos_cusum, 'neg_cusum': neg_cusum, 'threshold': threshold}


def run_cusum_on_candidates(long_df, candidates, threshold_std=5.0, drift_std=0.5):
    """
    Run CUSUM break detection across a list of item_ids.
    """
    results = []
    for item_id in candidates:
        item_df = long_df[long_df['item_id'] == item_id].sort_values('date').reset_index(drop=True)
        series = item_df['sales']

        break_idx, _ = cusum_detect(series, threshold_std=threshold_std, drift_std=drift_std)

        if break_idx is not None:
            results.append({
                'item_id': item_id,
                'break_detected': True,
                'break_date': item_df.loc[break_idx, 'date'],
                'break_index': break_idx,
                'series_length': len(series)
            })
        else:
            results.append({
                'item_id': item_id,
                'break_detected': False,
                'break_date': pd.NaT,
                'break_index': None,
                'series_length': len(series)
            })

    return pd.DataFrame(results)


def validate_against_christmas(break_df, long_df):
    """
    Sanity check: verify Dec 25 reads as anomalous (zero-sales) across items/years.
    """
    checks = []
    for item_id in break_df['item_id'].unique():
        item_df = long_df[long_df['item_id'] == item_id]
        christmases = item_df[item_df['date'].dt.strftime('%m-%d') == '12-25']
        for _, row in christmases.iterrows():
            checks.append({
                'item_id': item_id,
                'year': row['date'].year,
                'sales_on_christmas': row['sales'],
                'is_zero': row['sales'] == 0
            })
    return pd.DataFrame(checks)