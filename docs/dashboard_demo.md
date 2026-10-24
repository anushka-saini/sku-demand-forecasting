# Dashboard demo kit

Every number here is in `outputs/results_facts.json` and is shown in the app at its default settings: 95% target service level, 7-day lead time, base assumed costs, all categories.

## 2-minute walkthrough (follows the page order)

Overview (25 s)
> "The question is whether retail demand shifts in ways we can detect, and whether retraining after a shift improves forecasts and stocking. The four verdicts at the top are computed from the results, not typed. The main one is No: none of four retraining methods helped. That negative result is the finding. The gains came from a better static forecast and better-calibrated safety stock."

Data, Explore, Zeros and trimming (10 s)
> "I modelled the 726 high-volume California items: 23.8% of items but 66.4% of units sold. Before anything else, I trimmed pre-launch zeros, which would otherwise look like huge launch shifts."

Breaks (25 s)
> "Shifts are real but the detector is coarse. On break-free series that keep each item's seasonality, CUSUM raises false alarms 11.9% of the time, not the 4.8% the simpler test suggested. After removing chance alarms, about 63.5% of items have a real shift, with a CI from 59.8 to 67.2%. A sudden 20% shift is caught only 18% of the time, and only 171 of 488 breaks pass every quality check."

ML and comparison (15 s)
> "LightGBM is the best model: median error 0.744, against 0.755 for ETS and 0.976 for repeating last week. It runs 3.6% low overall. Its uncalibrated 90% bands covered only 85.5% of days; conformal calibration brings that to 89.9%."

Does retraining help? (15 s)
> "Restarting the model from the shift made error slightly worse, +0.0033 MASE, with a CI wholly above zero. A placebo, cutting the training window at random on items without a shift, also changed nothing. And it isn't because shifted items are harder: +2.8% after adjustment, with a CI that includes zero."

Uncertainty and stock (20 s)
> "Costs here are assumptions, shown with an orange frame. At a 95% target, LightGBM-based reorder points hold 16.6% less stock at 10.1% lower cost than the baseline. Sizing safety stock from calibrated bands cuts cost by another 31.1%. Adaptive costs more than static in all three seasons tested."

Planner (5 s)
> "This is the Monday list: 38 items below a 95% fill-rate target with calibrated safety stock (95 with the textbook formula), each with reasons and a CSV export."

Texas and Wisconsin (5 s)
> "Everything was rerun on Texas and Wisconsin with all settings frozen, and all 14 conclusion checks held."

## Five likely interview questions

1. "Why didn't retraining help?"
   Four methods were tested: restart from the shift, down-weight old data, a shift feature, and a blend. None met the pre-registered bar, and restarting was significantly worse (+0.0033 MASE). My best explanation, which I haven't tested, is that LightGBM's rolling-mean features already follow level changes, so discarding history loses seasonality and price information without gaining anything. The placebo supports this: random truncation on items without a shift also did nothing.

2. "Why should I trust your detector?"
   Partly, and the dashboard says so. I measured false alarms on break-free series that keep each item's yearly seasonality and autocorrelation: 11.9% for CUSUM. So I report a chance-adjusted share of 63.5% with a CI rather than the raw 67.8%. PELT agrees on 83.7% of items. Injecting known shifts shows the limits: a 20% step is caught 18% of the time. I trust it for "large shifts are common", not for spotting small ones.

3. "Your first false-alarm rate was 4.8%. Why did it change?"
   That 4.8% was measured on the same simple simulation used to set the threshold, so it was about 5% by construction. A stricter, seasonality-preserving null gave 11.9%. I pre-registered the new check before running it, logged it, and corrected the README.

4. "How do you know the inventory savings aren't just your cost assumptions?"
   Fill rate, stockouts and inventory are measured in the simulation, and costs are assumed and labelled. The direction held across a grid of holding (15–35%/yr) and lost-sale (0.25–2× price) costs, three seasons, and two other states. A 99% target beats 95% unless a lost sale costs less than about a quarter of the annual holding rate.

5. "What would you do next?"
   Correct the 3.6% under-forecast, test adaptation only on large, confirmed shifts, try store-level and slower-moving items where shifts are relatively bigger, and replace assumed costs with real ones.

## Repository description
> Do demand shifts matter? Walmart M5 SKU study: calibrated shift detection (CUSUM + PELT), rolling-origin forecasting (LightGBM, ETS), four pre-registered retraining strategies (none helped), conformal safety stock, inventory simulation, replication in TX/WI, and an explainable Streamlit dashboard.

## Resume bullets
- Built a demand-forecasting and inventory pipeline on 726 Walmart SKUs. LightGBM cut median forecast error (MASE) from 0.976 to 0.744 vs a seasonal-naive baseline, and its reorder points held 16.6% less inventory at 10.1% lower simulated cost.
- Pre-registered and ran a robustness study: recalibrated a structural-break detector (false alarms 11.9% on a seasonal null; 63.5% of items with a real shift, 95% CI 59.8–67.2%), tested four retraining strategies with Holm-corrected paired tests (none helped), and replicated all 14 conclusion checks in two other states.
- Shipped a Streamlit dashboard whose verdicts and README numbers are generated from a single facts file and checked by tests, including a planner watchlist that cut items below a 95% target from 95 to 38 with conformal safety stock.
