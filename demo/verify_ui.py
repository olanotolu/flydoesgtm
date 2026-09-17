"""Assertion-based UI verification for the auto-boot live mode."""
import sys
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8090"


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector(".row", timeout=60000)
        page.wait_for_timeout(10 * 320 + 600)

        rows = page.query_selector_all(".row")
        assert len(rows) == 10, f"expected 10 rows, got {len(rows)}"
        first = rows[0].inner_text()
        assert "IGNORE" in first or "EMAIL" in first \
            or "RESEARCH" in first, "no decision badge"
        assert page.text_content("#accounts").strip() == "10"

        page.click(".row >> nth=0")
        page.wait_for_timeout(600)
        assert "open" in page.get_attribute("#drawer", "class")
        assert len(page.query_selector_all("#sig-bars .hbar")) == 6
        assert len(page.query_selector_all("#act-bars .hbar")) == 7
        assert len(page.query_selector_all("#pool-rows .poolrow")) == 6

        page.click("#compare-btn")
        page.wait_for_timeout(3000)
        badges = page.query_selector_all("#compare-out .badge")
        assert len(badges) == 2, f"compare badges={len(badges)}"

        highlighted = page.evaluate(
            "() => highlightChannels.size + highlightPool.size")
        assert highlighted > 0, "no population highlight on selection"

        assert not errors, errors
        browser.close()
    print("UI VERIFIED: auto-boot live search, provenance drawer, "
          "pre/post compare, population highlights")


if __name__ == "__main__":
    main()
