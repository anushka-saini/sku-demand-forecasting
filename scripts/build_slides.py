"""Builds the presentation (PPTX, 14 slides for about 11 minutes) from the saved results.

Every number is read from outputs/tables/final_*.csv through src/final_text.py, the same source as the report and the
dashboard; the figures are the PNGs of notebook 13. Run after notebook 13 and build_report.py (which makes two figures):

    python scripts/build_slides.py
"""
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import final_text as FT  # noqa: E402
from config import FIGS, TABLES  # noqa: E402

f = FT.Final()
tx = f.all()
lb, nm = f.lb, f.nm
b, n, o = f.best, f.naive, f.old_best
facts = json.loads((ROOT / "outputs" / "results_facts.json").read_text())
ax_ay, ca_items = int(facts["scope"]["ax_ay_items"]["value"]), int(facts["scope"]["ca_items"]["value"])
sp = f.split.set_index("part")
cfgs = {k: json.loads((ROOT / "data" / "processed" / "final" / f"{k}_config.json").read_text())["config"] for k in ["lstm", "gru", "patchtst"]}
xp = json.loads((ROOT / "data" / "processed" / "final" / "xgb_params.json").read_text())
prop = json.loads((ROOT / "data" / "processed" / "final" / "prophet_config.json").read_text())
trials = len(__import__("pandas").read_csv(TABLES / "final_xgb_trials.csv"))
g = f.garch.set_index(["interval", "nominal"])
s_est = f.stock.set_index(["method", "model"])
tot = f.tot.set_index("model")

ACC = RGBColor(0x1F, 0x6F, 0x8B)
INK = RGBColor(0x22, 0x22, 0x22)
MUTE = RGBColor(0x66, 0x66, 0x66)
PH = RGBColor(0xB2, 0x6B, 0x00)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W, Hh = 13.333, 7.5


def fits(text, w_in, h_in, pt):
    """Rough check that the text fits its box: about 0.5 em per character, 1.25 line height."""
    cpl = max(1, int(w_in * 72 / (pt * 0.5)))
    lines = sum(max(1, -(-len(par) // cpl)) for par in text.split("\n"))
    return lines * pt * 1.25 / 72 <= h_in


def box(slide, x, y, w, h, text, pt=18, bold=False, color=INK, align=None, check=True):
    if check:
        assert fits(text, w, h, pt), f"text too long for its box ({pt}pt): {text[:60]}"
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size, p.font.bold, p.font.color.rgb, p.font.name = Pt(pt), bold, color, "Calibri"
        if align:
            p.alignment = align
        p.space_after = Pt(4)
    return tb


def bullets(slide, x, y, w, h, items, pt=18):
    text = "\n".join("• " + t for t in items)
    return box(slide, x, y, w, h, text, pt)


def title(slide, text, sub=None):
    bar = slide.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACC
    bar.line.fill.background()
    box(slide, 0.6, 0.3, 12.1, 1.0, text, pt=28, bold=True, color=ACC)
    if sub:
        box(slide, 0.6, 1.2, 12.1, 0.5, sub, pt=14, color=MUTE)


def pic(slide, path, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def foot(slide, text, k):
    box(slide, 0.6, 7.0, 11.5, 0.35, text, pt=10, color=MUTE, check=False)
    box(slide, 12.2, 7.0, 0.7, 0.35, str(k), pt=10, color=MUTE, align=PP_ALIGN.RIGHT, check=False)


def new(k, text, sub=None, note=None, source=None):
    s = prs.slides.add_slide(BLANK)
    title(s, text, sub)
    foot(s, source or "Walmart M5 data, California, 150 items, 28-day forecasts; test = six windows scored once", k)
    if note:
        s.notes_slide.notes_text_frame.text = note
    return s


def table(slide, x, y, w, rows, head, widths, pt=12, rh=0.34):
    shp = slide.shapes.add_table(len(rows) + 1, len(head), Inches(x), Inches(y), Inches(w), Inches(rh * (len(rows) + 1)))
    t = shp.table
    for j, wd in enumerate(widths):
        t.columns[j].width = Inches(wd)
    for j, h in enumerate(head):
        c = t.cell(0, j)
        c.text = h
        c.text_frame.paragraphs[0].font.size, c.text_frame.paragraphs[0].font.bold = Pt(pt), True
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.text = str(v)
            c.text_frame.paragraphs[0].font.size = Pt(pt)
    return t


pct = FT.pct
# 1 title ------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.12))
bar.fill.solid(); bar.fill.fore_color.rgb = ACC; bar.line.fill.background()
box(s, 0.8, 1.4, 11.7, 2.3, "Forecasting Retail SKU Demand:\nClassical, Machine-Learning and Deep Models Compared", pt=34, bold=True, color=ACC)
box(s, 0.8, 3.7, 11.7, 0.6, "Walmart M5 data, California, 150 high-volume items, 28 days ahead", pt=20, color=MUTE)
box(s, 0.8, 4.9, 11.7, 1.4, "[Team member 1 name, roll no.]\n[Team member 2 name, roll no.]\n[Team member 3 name, roll no.]\nCSET369, [University / department]", pt=18, color=PH)
s.notes_slide.notes_text_frame.text = "About 20 seconds. Introduce the team. Replace the placeholders."

# 2 problem ----------------------------------------------------------------
s = new(2, "Better demand forecasts mean less stock and fewer lost sales",
        note="About 45 seconds. The forecast feeds the order; error becomes either unsold stock or a lost sale.")
bullets(s, 0.7, 1.6, 12, 4.8, [
    "An order placed today must cover sales over the lead time; forecast error turns into unsold stock or lost sales.",
    "Question 1: which family forecasts best on the same items, split and metrics: classical time series, XGBoost/LightGBM, or neural networks?",
    "Question 2: our expectation was that a combination of models would beat every single model. Does it?",
    "Question 3: does a smaller error reduce cost in a stock simulation?",
    "Same level as our earlier work: one series per item (four stores added), 28 days ahead. We did not change it to make errors look smaller."], pt=20)

# 3 dataset ----------------------------------------------------------------
s = new(3, f"Data: {ax_ay} high-volume California items; 150 of them are modelled",
        note="About 45 seconds. Point at the shaded areas: validation then test.")
pic(s, FIGS / "final_dataset_line.png", 0.6, 1.5, w=8.3)
bullets(s, 9.1, 1.6, 3.9, 5, [
    f"{ax_ay} of {ca_items} items (classes AX, AY); 150 sampled, 50 per category",
    "Daily units, stores added; leading zeros trimmed",
    "About 1 item-day in 20 has no sales",
    "Known ahead: calendar, events, SNAP, price"], pt=16)

# 4 protocol ---------------------------------------------------------------
s = new(4, "Every choice is made on validation windows; the test is scored once",
        note="About 45 seconds. Stress: nothing after a forecast date is used. A test scrambles future sales and checks the features do not move.")
pic(s, FIGS / "final_workflow.png", 0.6, 1.5, w=12.1)
bullets(s, 0.7, 4.6, 12, 2.3, [
    f"Train before {sp.loc['train', 'last_day']}; 3 rolling validation windows of 28 days for all tuning; 6 test windows ({sp.loc['test (6 windows, scored once)', 'first_day']} to {sp.loc['test (6 windows, scored once)', 'last_day']})",
    "Ranking: MAE and RMSE. Also MAPE on non-zero days only, and WAPE as the zero-safe check",
    "One feature set for all learning models; scalers use training data only"], pt=17)

# 5 classical recap --------------------------------------------------------
cl = lb[lb["model"].isin(["seasonal_naive", "hw_add", "ets_boxcox", "theta", "arima", "sarima", "sarimax", "prophet", "combination"])].sort_values("rank")
s = new(5, f"Classical models recap: the best, {nm[o['model']].split(' (')[0]}, beats seasonal naive by {pct(1 - o['mae'] / n['mae'])} on MAE",
        note="About 45 seconds. These are from the earlier work and are unchanged; they are the reference for the new models.")
table(s, 0.7, 1.7, 8.2, [[nm[m].replace(" (SARIMAX, HW, Theta)", ""), f"{lb[lb['model'] == m]['mae'].iloc[0]:.2f}", f"{lb[lb['model'] == m]['rmse'].iloc[0]:.2f}"] for m in cl["model"]],
      ["Model", "MAE", "RMSE"], [5.2, 1.5, 1.5], pt=14)
bullets(s, 9.2, 1.7, 3.8, 4.5, ["Each model is fitted per item on the last 730 days",
                                "SARIMA orders chosen by AICc; Prophet, SARIMAX use a yearly term",
                                "Combination = average of SARIMAX, Holt-Winters, Theta"], pt=15)

# 6 advanced ---------------------------------------------------------------
s = new(6, "Advanced models: trees, neural nets, VAR, GARCH and three combinations",
        note="About 60 seconds. Give one line each. Neural nets: 3 seeds, mean and std.")
bullets(s, 0.7, 1.6, 12.2, 5.2, [
    f"XGBoost: Optuna, {trials} trials, 3 time-series folds, early stopping; LightGBM as a reference",
    f"Prophet with events as holidays plus SNAP and price; searched changepoint, mode, seasonality prior (chosen {prop['changepoint_prior_scale']}, {prop['seasonality_mode']})",
    f"LSTM, GRU, PatchTST (own PyTorch code, patches + transformer): small grids, early stopping, 3 seeds",
    "VAR on the 3 category totals: ADF check, lag by AIC, weekday/SNAP/event regressors",
    "GARCH(1,1) on the best model's residuals, for volatility and prediction intervals",
    "Combinations: average of top 3, stack with non-negative weights, hybrid (classical + XGBoost on residuals); built on validation only"], pt=18)

# 7 results bars -----------------------------------------------------------
s = new(7, f"{b['name']} ranks first on both MAE ({b['mae']:.2f}) and RMSE ({b['rmse']:.2f})",
        note="About 60 seconds. Read off the top three. Neural nets are behind the classical models.")
pic(s, FIGS / "final_mae_rmse_bars.png", 0.5, 1.4, w=8.3)
bullets(s, 9.0, 1.7, 4.0, 4.8, [f"{pct(1 - b['mae'] / n['mae'])} lower MAE than seasonal naive",
                                f"Best earlier model {nm[o['model']].split(' (')[0]}: {o['mae']:.2f} / {o['rmse']:.2f}",
                                "Neural nets: better than naive, behind the classical models",
                                "Whiskers: std over 3 seeds"], pt=16)

# 8 table + combos ---------------------------------------------------------
top = lb.head(7)
s = new(8, "No combination beat the best single model on both measures",
        note="About 60 seconds. Our hypothesis was a combination wins. It does not here: say so plainly.")
table(s, 0.6, 1.6, 7.9, [[int(r["rank"]), nm[r["model"]].split(" (")[0] if r["model"] != "combination" else "Combination", f"{r['mae']:.2f}", f"{r['rmse']:.2f}", f"{r['mape_nonzero']:.3f}", f"{r['wape']:.3f}"] for _, r in top.iterrows()],
      ["#", "Model", "MAE", "RMSE", "MAPE*", "WAPE"], [0.5, 3.8, 1, 1, 0.8, 0.8], pt=14, rh=0.42)
box(s, 0.6, 5.2, 7.9, 0.5, "*MAPE on non-zero days only (about 5% of item-days left out)", pt=12, color=MUTE)
bullets(s, 8.8, 1.6, 4.2, 5.2, ["Hypothesis: a combination wins. Result: not supported",
                                f"Best single: {b['name']}",
                                "Average of top 3 and the stack are close but not better on both",
                                "Hybrid is weaker: only 3 to 8 earlier windows to learn residuals"], pt=16)

# 9 DM + overlay -----------------------------------------------------------
dm = f.dm
sig_naive = bool(dm[(dm["model_b"] == "seasonal_naive") & (dm["loss"] == "absolute error")]["significant_5pct"].iloc[0])
p_sarima = dm[(dm["model_b"] == "sarima") & (dm["loss"] == "absolute error")]["p_value"].iloc[0]
s = new(9, "The gain over seasonal naive is real; the gain over the best classical models is not significant",
        note="About 60 seconds. Diebold-Mariano with Newey-West and the small-sample correction.")
pic(s, FIGS / "final_forecast_overlay.png", 0.5, 1.5, h=5.3)
bullets(s, 7.9, 1.7, 5.1, 5, [f"Versus seasonal naive: significant ({FT.fp(dm[(dm['model_b'] == 'seasonal_naive') & (dm['loss'] == 'absolute error')]['p_value'].iloc[0])})" if sig_naive else "Versus seasonal naive: not significant",
                              f"Versus SARIMA: p {FT.fp(p_sarima)} (absolute error); not significant at 5%",
                              f"Versus the earlier best: p {FT.fp(dm[(dm['model_b'] == o['model']) & (dm['loss'] == 'absolute error')]['p_value'].iloc[0])}; not significant",
                              "XGBoost has the lower MAE on most items, but the day-level gap is small"], pt=16)

# 10 failures --------------------------------------------------------------
seg = f.seg
gv = lambda sg, m=b["model"], k="mae": float(seg[(seg["segment"] == sg) & (seg["model"] == m)][k].iloc[0])
s = new(10, "Where models fail: weekends, zero-demand days and quiet spells, not SNAP days",
        note="About 60 seconds. Spikes and zeros are the main misses.")
pic(s, FIGS / "final_failure_segments.png", 0.5, 1.5, w=7.6)
bullets(s, 8.3, 1.6, 4.7, 5.2, [f"Event days: MAE {gv('event days'):.2f} vs {gv('no event'):.2f} on other days",
                                f"SNAP days: {gv('SNAP days'):.2f} vs {gv('non-SNAP days'):.2f}, no worse",
                                f"Weekends {gv('weekend'):.2f} vs weekdays {gv('weekday'):.2f}",
                                f"Zero-sales days: forecast {gv('days with zero sales', k='bias'):.1f} units too high",
                                f"After a zero week: {gv('zero week before origin'):.2f}, no better than naive ({gv('zero week before origin', 'seasonal_naive'):.2f})",
                                "Bias is negative: spikes are missed"], pt=15)

# 11 GARCH and VAR ---------------------------------------------------------
s = new(11, "GARCH did not improve intervals, and VAR on category totals did not help",
        note="About 60 seconds. GARCH models variance; it cannot lower MAE or RMSE by itself.")
pic(s, FIGS / "final_garch_volatility.png", 0.5, 1.5, w=7.4)
lm = f.archlm
bullets(s, 8.1, 1.6, 4.9, 5.2, [f"ARCH-LM: changing variance in {(lm['arch_lm_p'] < 0.05).mean() * 100:.0f}% of items",
                                f"80% band coverage: GARCH {g.loc[('GARCH(1,1)', 0.8), 'coverage'] * 100:.1f}% vs constant {g.loc[('constant std', 0.8), 'coverage'] * 100:.1f}%",
                                f"Interval score {g.loc[('GARCH(1,1)', 0.8), 'interval_score']:.1f} vs {g.loc[('constant std', 0.8), 'interval_score']:.1f}: no gain",
                                "Real volatility follows the calendar (weekends), which GARCH does not see",
                                f"VAR, category totals: MAE {tot.loc['var', 'mae']:.0f}, RMSE {tot.loc['var', 'rmse']:.0f} (best item model summed: {tot.loc[b['model'], 'mae']:.0f}, {tot.loc[b['model'], 'rmse']:.0f})"], pt=15)

# 12 stock -----------------------------------------------------------------
ref = "combination"
s = new(12, "Stock: a lower error pays off only with safety stock sized from the model's own errors",
        note="About 60 seconds. Same simulation as before: 7-day lead time, 95% target, assumed costs.", source="Stock simulation over 168 days with assumed costs; not a measured saving")
mods = ["seasonal_naive", "sarima", "combination", b["model"]]
table(s, 0.6, 1.7, 8.4, [[nm[m].split(" (")[0], f"${s_est.loc[('conformal', m), 'total_cost']:,.0f}", f"{s_est.loc[('conformal', m), 'fill_rate'] * 100:.1f}%",
                          f"${s_est.loc[('normal formula', m), 'total_cost']:,.0f}", f"{s_est.loc[('normal formula', m), 'fill_rate'] * 100:.1f}%"] for m in mods if ("conformal", m) in s_est.index],
      ["Model", "Calibrated cost", "Fill", "Textbook cost", "Fill"], [2.8, 1.8, 1.0, 1.8, 1.0], pt=14, rh=0.45)
bullets(s, 9.2, 1.7, 3.8, 5, [f"Calibrated: {b['name'].split(' (')[0]} {pct(s_est.loc[('conformal', b['model']), 'total_cost'] / s_est.loc[('conformal', ref), 'total_cost'] - 1, 1)} cheaper than the earlier best, but the interval includes zero",
                              "Textbook formula: more lost sales, higher cost",
                              "The safety-stock rule matters more than the choice among the top models"], pt=14)

# 13 challenges ------------------------------------------------------------
s = new(13, "Challenges and limitations",
        note="About 45 seconds. Be specific: sample size, six windows, neural nets under-tuned.")
bullets(s, 0.7, 1.6, 12.2, 5.3, [
    "150 items, one state, stores added up: slow movers and store level are untested",
    "Six test windows: top models are close; only some gaps are significant",
    "Neural nets trained once and rolled forward, small grids, CPU laptop: probably under-tuned, not a verdict on deep learning",
    "Hybrid had few residual windows; stack and top-3 chosen on three validation windows",
    "Prices and calendar assumed known ahead; stock costs are assumptions",
    "Hardest parts: keeping every choice on validation, leak-free features, run time, zeros in MAPE"], pt=18)

# 14 conclusion ------------------------------------------------------------
s = new(14, "Conclusion and future scope", note="About 45 seconds. Close with the answer to each question.")
bullets(s, 0.7, 1.6, 12.2, 5.3, [
    f"Best model: {b['name']}, MAE {b['mae']:.2f}, RMSE {b['rmse']:.2f}; clearly better than seasonal naive, not significantly better than the best classical models",
    "A combination did not win; neural nets did not catch the classical models here; GARCH and VAR did not help",
    "For stock, size safety stock from the model's own past errors",
    "Next: refit neural nets at every origin on a GPU; model weekend, event and zero spikes directly; calendar-aware safety stock; other states and slow movers"], pt=19)

OUT = ROOT / "outputs" / "report"
OUT.mkdir(parents=True, exist_ok=True)
prs.save(OUT / "final_slides.pptx")
print("slides:", len(prs.slides), "->", (OUT / "final_slides.pptx").relative_to(ROOT))
