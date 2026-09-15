"""Automated portal QA: loads every tab in Microsoft Edge (Playwright), records console errors,
takes screenshots (light/dark, desktop/phone) into outputs/portal_qa/.

    python -m http.server 8765 --directory portal   (in another shell)
    python scripts/qa_portal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "portal_qa"
OUT.mkdir(parents=True, exist_ok=True)
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/index.html"


def main():
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        for scheme, size, tag in (("light", (1600, 1000), "desktop_light"), ("dark", (1600, 1000), "desktop_dark"), ("light", (400, 860), "phone")):
            ctx = browser.new_context(viewport={"width": size[0], "height": size[1]}, color_scheme=scheme)
            page = ctx.new_page()
            page.on("console", lambda m, t=tag: errors.append((t, m.type, m.text)) if m.type in ("error", "warning") else None)
            page.on("pageerror", lambda e, t=tag: errors.append((t, "pageerror", str(e))))
            page.goto(URL, wait_until="networkidle")
            page.wait_for_timeout(6000)
            info = page.evaluate("""() => { const m = window.__ghmap; if (!m) return null;
                return { stations: m.queryRenderedFeatures({layers:['stations']}).length, adm1: m.queryRenderedFeatures({layers:['adm1-line']}).length }; }""")
            print(tag, "rendered:", info)
            # click a location to populate the time series
            box = page.locator("#map").bounding_box()
            if box:
                page.mouse.move(box["x"] + box["width"] * 0.48, box["y"] + box["height"] * 0.78)
                page.wait_for_timeout(500)
                page.mouse.click(box["x"] + box["width"] * 0.48, box["y"] + box["height"] * 0.78)
                page.wait_for_timeout(2500)
            page.screenshot(path=str(OUT / f"{tag}_monitor.png"), full_page=True)
            for tab in ("stations", "exposure", "validation", "methods"):
                page.click(f'.tabs button[data-tab="{tab}"]')
                page.wait_for_timeout(1500)
                if tab == "stations":
                    rows = page.locator("#station-table tbody tr")
                    if rows.count():
                        rows.first.click(); page.wait_for_timeout(1500)
                page.screenshot(path=str(OUT / f"{tag}_{tab}.png"), full_page=True)
            ctx.close()
        browser.close()
    print("console errors/warnings:")
    for e in errors:
        print("  ", e)


if __name__ == "__main__":
    main()
