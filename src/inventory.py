import numpy as np
from scipy import stats


def z_value(service_level):
    """Standard normal quantile for a cycle service level (e.g. 0.95 -> 1.645)."""
    return stats.norm.ppf(service_level)


def safety_stock_normal(sigma_daily, lead_time, service_level):
    """
    Normal-error safety stock: z * sigma_daily * sqrt(L). Assumes daily forecast
    errors are independent with constant variance over the lead time.
    """
    if not np.isfinite(sigma_daily):
        return 0.0
    return max(0.0, z_value(service_level) * sigma_daily * np.sqrt(lead_time))


def safety_stock_empirical(lead_time_error_sums, service_level):
    """
    Distribution-free safety stock for intermittent demand: the service-level
    quantile of observed lead-time error sums (actual - forecast over L days).
    """
    e = np.asarray(lead_time_error_sums, dtype=float)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return 0.0
    return max(0.0, float(np.quantile(e, service_level)))


def lead_time_error_sums(daily_errors_by_window, lead_time):
    """Rolling L-day sums of daily errors, computed within each window (no crossing windows)."""
    sums = []
    for e in daily_errors_by_window:
        e = np.asarray(e, dtype=float)
        if len(e) >= lead_time:
            sums.append(np.convolve(e, np.ones(lead_time), mode='valid'))
    return np.concatenate(sums) if sums else np.array([])


def forecast_path(window_fcs, t_next, n_days, period=7):
    """
    Forecasts for days t_next .. t_next+n_days-1, using only the forecast issued
    at the latest origin before t_next. window_fcs: list of 28-day forecast
    arrays (window w covers days 28w .. 28w+27). Days past the end of that
    window repeat the window's last week.
    """
    w_len = len(window_fcs[0])
    w = min(t_next // w_len, len(window_fcs) - 1)
    f = np.asarray(window_fcs[w], dtype=float)
    pos = t_next - w * w_len + np.arange(n_days)
    beyond = pos >= w_len
    pos[beyond] = w_len - period + (pos[beyond] - w_len) % period
    return f[pos], w


def simulate_policy(demand, window_fcs, safety_stocks, lead_time=7, review_days=7):
    """
    Daily-review reorder-point policy with lost sales.

    Each day: receive arrivals, serve demand (unmet demand is lost), then at end
    of day compute
        ROP = forecast demand over the next L days + SS
        S   = ROP + forecast demand for the following `review_days` days
    and, if inventory position (on hand + on order) <= ROP, order S - position.
    An order placed at the end of day t is available at the start of day t+L.
    safety_stocks: one value per forecast window (re-estimated at each origin).
    Starts with S on hand and nothing on order.
    """
    demand = np.asarray(demand, dtype=float)
    T = len(demand)
    arrivals = np.zeros(T + lead_time + 1)

    f0, _ = forecast_path(window_fcs, 0, lead_time + review_days)
    on_hand = f0.sum() + safety_stocks[0]
    on_order = 0.0
    out = {k: np.zeros(T) for k in ['on_hand', 'sales', 'lost', 'order_qty', 'rop']}
    for t in range(T):
        on_hand += arrivals[t]
        on_order -= arrivals[t]
        sales = min(on_hand, demand[t])
        on_hand -= sales
        f, w = forecast_path(window_fcs, t + 1, lead_time + review_days)
        rop = f[:lead_time].sum() + safety_stocks[w]
        position = on_hand + on_order
        q = 0.0
        if position <= rop:
            q = rop + f[lead_time:].sum() - position
            arrivals[t + lead_time] += q
            on_order += q
        out['on_hand'][t], out['sales'][t], out['lost'][t] = on_hand, sales, demand[t] - sales
        out['order_qty'][t], out['rop'][t] = q, rop
    return out


def policy_kpis(sim, demand, price, holding_rate_annual, stockout_multiple):
    """
    KPIs and costs for one simulated policy.
      holding cost  = holding_rate_annual / 365 * unit price * end-of-day units on hand
      stockout cost = stockout_multiple * unit price * units of lost sales
    Unit price (sell price) is used as the value basis for both; ordering cost is ignored.
    """
    demand, price = np.asarray(demand, dtype=float), np.asarray(price, dtype=float)
    holding = np.sum(holding_rate_annual / 365.0 * price * sim['on_hand'])
    stockout = np.sum(stockout_multiple * price * sim['lost'])
    return {'fill_rate': sim['sales'].sum() / demand.sum() if demand.sum() > 0 else np.nan,
            'stockout_days': int(np.sum(sim['lost'] > 0)),
            'lost_units': sim['lost'].sum(),
            'avg_inventory_units': sim['on_hand'].mean(),
            'avg_inventory_value': np.mean(sim['on_hand'] * price),
            'n_orders': int(np.sum(sim['order_qty'] > 0)),
            'holding_cost': holding, 'stockout_cost': stockout, 'total_cost': holding + stockout}
