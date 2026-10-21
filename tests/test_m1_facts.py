"""Milestone 1 synopsis numbers, checked against the real data. Skipped if data/raw is not downloaded."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
pytestmark = pytest.mark.skipif(not (RAW / "sales_train_evaluation.csv").exists(), reason="raw M5 data not downloaded")


@pytest.fixture(scope="module")
def raw():
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    calendar = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(RAW / "sell_prices.csv")
    return sales, calendar, prices


@pytest.fixture(scope="module")
def ca_items(raw):
    """California sales summed over the four stores, one row per product."""
    sales, calendar, _ = raw
    days = [c for c in sales.columns if c.startswith("d_")]
    return sales[sales["state_id"] == "CA"].groupby(["item_id", "cat_id"])[days].sum()


def test_file_shapes(raw):
    sales, calendar, prices = raw
    assert sales.shape == (30490, 1947)
    assert calendar.shape == (1969, 14)
    assert prices.shape == (6841121, 4)


def test_1941_days_from_29_jan_2011_to_22_may_2016(raw):
    sales, calendar, _ = raw
    days = [c for c in sales.columns if c.startswith("d_")]
    assert len(days) == 1941
    dates = calendar.set_index("d").loc[days, "date"]
    assert dates.iloc[0] == pd.Timestamp("2011-01-29") and dates.iloc[-1] == pd.Timestamp("2016-05-22")


def test_no_empty_sales_cells(raw):
    assert raw[0].isna().sum().sum() == 0


def test_3049_california_products(ca_items):
    assert len(ca_items) == 3049


def test_1565_products_start_with_more_than_30_zero_days(ca_items):
    vals = ca_items.to_numpy()
    first_sale = np.argmax(vals > 0, axis=1)
    assert int(((first_sale > 30) & (vals > 0).any(axis=1)).sum()) == 1565


def test_726_selected_items():
    n = sum(pd.read_csv(ROOT / "data" / "interim" / f"{c}_candidates.csv")["item_id"].nunique()
            for c in ["foods", "hobbies", "household"])
    assert n == 726


def test_prices_missing_until_a_product_goes_on_sale(raw):
    """No store-item sells anything in a week before its first price row."""
    sales, calendar, prices = raw
    days = [c for c in sales.columns if c.startswith("d_")]
    vals = sales[days].to_numpy()
    sold = (vals > 0).any(axis=1)
    week_of_day = calendar.set_index("d").loc[days, "wm_yr_wk"].to_numpy()
    first_sale_week = pd.Series(week_of_day[np.argmax(vals > 0, axis=1)], index=sales["item_id"] + "|" + sales["store_id"])[sold]
    first_price_week = prices.groupby(prices["item_id"] + "|" + prices["store_id"])["wm_yr_wk"].min()
    assert (first_sale_week >= first_price_week.reindex(first_sale_week.index)).all()


def test_sales_are_highest_at_the_weekend_in_every_category(raw, ca_items):
    _, calendar, _ = raw
    days = ca_items.columns
    weekday = calendar.set_index("d").loc[days, "weekday"].to_numpy()
    for cat, g in ca_items.groupby(level="cat_id"):
        by_day = pd.Series(g.sum().to_numpy(), index=weekday).groupby(level=0).mean()
        weekend = by_day[["Saturday", "Sunday"]]
        assert weekend.min() > by_day.drop(["Saturday", "Sunday"]).max(), cat
