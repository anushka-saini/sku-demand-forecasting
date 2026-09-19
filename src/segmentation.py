
import pandas as pd
import numpy as np

def trim_leading_zeros(df):
    """Cut each item's series to its first real sale date."""
    trimmed = []
    for item_id, grp in df.groupby('item_id'):
        grp = grp.sort_values('date')
        first_sale_idx = grp['sales'].to_numpy().nonzero()[0]
        if len(first_sale_idx) == 0:
            continue
        trimmed.append(grp.iloc[first_sale_idx[0]:])
    return pd.concat(trimmed, ignore_index=True)

def scope_and_aggregate(sales, state_id, cat_id):
    """Scope to one state+category, aggregate stores into item-level."""
    scoped = sales[(sales['state_id'] == state_id) & (sales['cat_id'] == cat_id)].copy()
    day_cols = [c for c in scoped.columns if c.startswith('d_')]
    item_level = scoped.groupby('item_id')[day_cols].sum().reset_index()
    return item_level, day_cols

def abc_xyz_segment(long_df):
    """Run ABC-XYZ segmentation on a long-format sales dataframe."""
    item_stats = long_df.groupby('item_id')['sales'].agg(
        total_sales='sum', mean_sales='mean', std_sales='std'
    ).reset_index()
    item_stats['cv'] = item_stats['std_sales'] / item_stats['mean_sales']
    item_stats = item_stats.sort_values('total_sales', ascending=False).reset_index(drop=True)
    item_stats['cum_pct'] = item_stats['total_sales'].cumsum() / item_stats['total_sales'].sum()
    item_stats['abc_class'] = pd.cut(item_stats['cum_pct'], bins=[0, 0.7, 0.9, 1.0], labels=['A','B','C'])
    item_stats['xyz_class'] = pd.cut(item_stats['cv'], bins=[0, 0.5, 1.0, np.inf], labels=['X','Y','Z'])
    item_stats['segment'] = item_stats['abc_class'].astype(str) + item_stats['xyz_class'].astype(str)
    return item_stats
