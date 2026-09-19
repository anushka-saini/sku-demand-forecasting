from statsmodels.tsa.stattools import adfuller
import pandas as pd

def run_adf_tests(long_df, candidates, min_length=30):
    """ADF stationarity test across a list of item_ids."""
    results = []
    for item_id in candidates:
        series = long_df[long_df['item_id'] == item_id].set_index('date')['sales']
        if len(series) < min_length:
            continue
        try:
            stat, p_value = adfuller(series)[:2]
            results.append({'item_id': item_id, 'adf_stat': stat, 'p_value': p_value})
        except Exception as e:
            print(f"Failed on {item_id}: {e}")
    df = pd.DataFrame(results)
    df['is_stationary'] = df['p_value'] < 0.05
    return df