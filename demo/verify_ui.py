"""Small browser smoke test for the recorded, no-send showcase."""
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8090"


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector(".row", timeout=60000)
        rows = page.locator(".row")
        assert rows.count() == 5, f"expected five recorded rows, got {rows.count()}"
        assert page.locator("#accounts").text_content().strip() == "5"
        assert page.locator("#queue-label").text_content().strip() == "READ-ONLY"
        rows.nth(0).click()
        assert page.locator("#timeline").locator(".step").count() >= 3
        assert page.locator("#contact-state").text_content().strip() == "NO SEND PATH"
        page.locator("#live-mode").click()
        assert page.locator("#live-mode").get_attribute("aria-pressed") == "true"
        assert not errors, errors
        browser.close()
    print("UI VERIFIED: recorded proof, timeline, live toggle, and no-send state")


if __name__ == "__main__":
    main()
