"""
Start the dashboard, open every page in a headless browser with Playwright, and save full-page screenshots
to docs/screenshots/. Uses Playwright's Chromium if installed, otherwise the system Microsoft Edge.

Run: python scripts/screenshots.py          (light theme, docs/screenshots/)
     python scripts/screenshots.py --dark   (dark theme, docs/screenshots/dark/)
     python scripts/screenshots.py --pages overview,accuracy   (only these pages)
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
PORT = 8599
# page -> (url path, window height). Streamlit scrolls inside its own container, so a tall window is used
# instead of a "full page" screenshot.
PAGES = {"overview": ("", 1500), "data_scope": ("data_scope", 2300), "demand_patterns": ("demand_patterns", 2700),
         "accuracy": ("accuracy", 3300), "item_explorer": ("item_explorer", 2200), "demand_shifts": ("demand_shifts", 2700),
         "stock_impact": ("stock_impact", 2900), "method_limits": ("method_limits", 1700)}


def wait_for_server(timeout=90):
    for _ in range(timeout):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise SystemExit("dashboard did not start")


def launch(p):
    try:
        return p.chromium.launch()
    except Exception:
        return p.chromium.launch(channel="msedge")


def main():
    from playwright.sync_api import sync_playwright
    dark = "--dark" in sys.argv
    pages = PAGES
    if "--pages" in sys.argv:
        wanted = sys.argv[sys.argv.index("--pages") + 1].split(",")
        pages = {k: v for k, v in PAGES.items() if k in wanted}
    out = OUT / "dark" if dark else OUT
    out.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"),
                               "--server.port", str(PORT), "--server.headless", "true"], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_server()
        with sync_playwright() as p:
            browser = launch(p)
            for name, (path, height) in pages.items():
                page = browser.new_page(viewport={"width": 1440, "height": height}, color_scheme="dark" if dark else "light")
                page.goto(f"http://localhost:{PORT}/{path}", wait_until="networkidle")
                page.wait_for_selector("h1", timeout=60000)
                # wait until Streamlit has finished running the script
                page.wait_for_function("!document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=60000)
                time.sleep(2)                                   # let Plotly finish drawing
                page.screenshot(path=str(out / f"{name}.png"))
                page.close()
                print("saved", name)
            browser.close()
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
