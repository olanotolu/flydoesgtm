"""Capture the live-mode GIF + verify the frontend end-to-end.

  .venv/bin/python demo/capture_gif.py

The page auto-boots into a live Clay search. Fails loudly on browser
console errors — this is a verification run, not just a recording.
"""
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parents[1]
OUT = ROOT / "demo" / "frames"
GIF = ROOT / "demo" / "clayfly.gif"
URL = "http://127.0.0.1:8090"


def shot(page, i):
    page.screenshot(path=str(OUT / f"f{i:02d}.png"))


def main():
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.png"):
        f.unlink()
    errors = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("console", lambda m: errors.append(m.text)
                if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(2000)          # connectome loads, search fires
        shot(page, 0)
        page.wait_for_timeout(1500)
        shot(page, 1)
        page.wait_for_selector(".row", timeout=60000)
        page.wait_for_timeout(10 * 320 + 800)   # all rows streamed in
        shot(page, 2)

        # provenance on the first row
        page.click(".row >> nth=0")
        page.wait_for_timeout(700)
        shot(page, 3)

        # pre/post training comparison on the same company
        page.click("#compare-btn")
        page.wait_for_timeout(2500)
        shot(page, 4)

        for i in range(5, 10):
            page.wait_for_timeout(500)
            shot(page, i)

        browser.close()

    if errors:
        print("CONSOLE ERRORS:", *errors, sep="\n  ")
        sys.exit(1)

    frames = sorted(OUT.glob("*.png"))
    if not frames:
        sys.exit("no frames captured")
    imgs = [Image.open(f).convert("P", palette=Image.ADAPTIVE, colors=128)
            for f in frames]
    imgs[0].save(GIF, save_all=True, append_images=imgs[1:],
                 duration=900, loop=0, optimize=True)
    size = GIF.stat().st_size / 1024
    print(f"OK: {len(imgs)} frames -> {GIF} ({size:.0f} KB)")


if __name__ == "__main__":
    main()
