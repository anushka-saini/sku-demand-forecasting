# Dashboard demo kit

Every number below is shown somewhere in the app at its default settings. Defaults are a 95% service level, a 7-day lead time, LightGBM, and item FOODS_1_004.

## 3-minute walkthrough

**0:00 – Overview (30 s)**
> "This project asks whether Walmart items' demand shifts in detectable ways, and whether a forecast that retrains after a shift leads to better stocking decisions. The headline at the top is generated from the results, and it's a negative finding. Retraining after a break made forecasts slightly *worse*: median MASE changed by +0.0033, with a 95% interval of +0.0015 to +0.0053, across 488 items. A random training cut on items with no break changed it by +0.0009, which is essentially nothing. I analysed 726 items, and 67.2% had a break by the last forecast date."

**0:30 – Data & method (20 s)**
> "Here's the pipeline. One data bug mattered a lot. FOODS_1_004 shows 403 days of zero sales before it was ever sold. Left in, every new product would look like it had a huge structural break at launch. 1,565 items had more than 30 of these leading zero days, so I trimmed every series to its first sale."

**0:50 – Segmentation (15 s)**
> "I modelled only the AX and AY items: high volume and reasonably steady. That's 726 of 3,049 items, but 66% of units sold. You can click any item here and open it in the forecast explorer."

**1:05 – Break detection (35 s)**
> "Breaks are found with a weekly CUSUM whose threshold I calibrated on simulated series with no breaks. It gives false alarms on 4.8% of them. PELT, a second method, agrees on 83.7% of items, with a kappa of 0.62. The honest limitation is this chart. When I inject *no* shift at all, CUSUM still alarms 45% of the time, because the series drift on their own. And only 5.7% of breaks sit near a sustained price change, against 4.2% expected by chance. So most breaks have no identifiable cause in this data."

**1:40 – Forecast comparison (40 s)**
> "LightGBM is the best static model, with a median MASE of 0.744 against 0.976 for seasonal naive. In the confidence-interval chart, adaptive retraining doesn't significantly help in any of the 8 model-category groups. The orange intervals sit at or right of zero. The placebo shows that shortening the training window at random doesn't help either. Break items also aren't harder to forecast. After adjusting for volume, variability and history length, the difference is +2.8%, and the interval includes zero. Down here, the item explorer shows actual sales against the static and adaptive forecasts, with 90% bands."

**2:20 – Inventory decision (30 s)**
> "This simulates every item's stock day by day. At a 95% target, static LightGBM reaches a 97.69% fill rate holding $161,205 of inventory, at a total cost of $68,708. Seasonal naive needs $193,251 of inventory for 97.28%. Adaptive costs +12.4% more than static. But item by item that difference isn't significant (p = 0.307), because a few items drive it. The costs are assumptions, labelled as such. Across every cost combination in the grid, adaptive costs 8.3% to 16.0% more."

**2:50 – Assumptions & limitations (10 s)**
> "This last page lists the scope, the assumed parameters, what didn't work, and what would change the conclusion. For example, the 90% bands only cover 84.5% of days, so safety stock comes out a bit too low."

## Five likely interview questions about the dashboard

1. **"Why is the headline negative? Isn't that a failed project?"**
   The question was whether adapting helps. The data says no for this setup: +0.0033 MASE, interval above zero, and the placebo near zero. A dashboard that hid that would mislead a planner. The positive, actionable result is that a good static model beats the baseline: 0.744 vs 0.976 MASE, with less inventory and a lower cost.

2. **"How do I know the numbers in the app are right?"**
   The app never models anything; it only reads result files from the notebooks. A test recomputes the per-item adaptive and placebo medians from the error table and checks they match the notebook's statistical tests. Another test renders every page. Every headline and chart title is built from the data, not typed in.

3. **"What does the stockout-risk flag mean, and is it reliable?"**
   It's a derived rule, labelled as such on the page: the item's simulated fill rate fell below the target service level. At 95% with a 7-day lead time, 95 items are flagged for static LightGBM. It depends on the simulation's assumptions (lead time, lost sales), so it's a prompt for review, not a measured fact.

4. **"Why should a planner trust the cost numbers?"**
   They shouldn't treat them as measured. Holding cost (25% a year), stockout cost (0.5× price) and lead time are assumptions, shown with an ASSUMED tag, and a sensitivity grid sits next to them. The ranking (adaptive costs more than static) held in every combination tested. The cheapest service level was 99% in every case, which would change if holding costs were much higher.

5. **"Why Streamlit with pre-computed files instead of live models?"**
   The backtest takes about 30 minutes, which is too slow for a page load. Pre-computing keeps every page under a second in testing, keeps the app reproducible from a fresh clone (about 22 MB of committed results), and guarantees the dashboard shows exactly what the notebooks produced.

## Captions

**GitHub repo description (≤ 350 chars):**
> Detecting structural breaks in 726 Walmart (M5) SKU demand series and testing whether retraining after a break improves forecasts and inventory decisions. Calibrated CUSUM + PELT, rolling-origin backtest (LightGBM MASE 0.744 vs 0.976 naive), placebo-controlled test, inventory simulation, and an explainable Streamlit dashboard.

**Resume line:**
> Built an explainable Streamlit dashboard over a 726-SKU Walmart demand study. It showed that a static LightGBM cut median MASE from 0.976 to 0.744 vs seasonal naive, while retraining after detected breaks gave no gain under a placebo-controlled test. Every figure is generated from tested, pre-computed results.
