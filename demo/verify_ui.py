"""Small browser smoke test for the single-screen recorded, no-send showcase."""
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8090"


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(URL, wait_until="networkidle")
        page.keyboard.press("Enter")
        page.wait_for_selector("#review-question", timeout=60000)
        assert page.locator(".signal-det h2").text_content().strip() == "Anthropic launches Fable 5.1 and Mythos 5.1"
        assert page.locator(".signal-tags span").count() == 4, "expected four signal tags"
        assert page.locator(".source-card").count() == 5, "expected five listening sources"
        assert page.locator(".action-card").count() == 4, "expected four GTM plan steps"
        assert page.locator("#firing-list .firing-row").count() >= 1, "expected firing signal rows"
        assert page.locator("#decision-chip").text_content().strip() != ""
        assert page.locator("#evidence-verdict").text_content().strip() != ""
        page.locator("#think-run").click()
        page.wait_for_selector("#fly-decision:not([hidden])", timeout=30000)
        page.wait_for_timeout(2500)
        assert page.locator("#think-bars .think-bar").count() == 6, "expected six sensory energy bars"
        assert page.locator("#think-readout .think-line").count() >= 3, "expected streamed reasoning lines"
        assert "The fly says YES" in page.locator("#fly-decision-verdict").text_content()
        assert page.locator("#fly-decision-motion li").count() == 3, "expected three motion steps"
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(400)
        scroll_width = page.evaluate("() => document.documentElement.scrollWidth")
        assert scroll_width <= 390, f"mobile horizontal overflow: {scroll_width}"
        assert not errors, errors
        browser.close()
    print("UI VERIFIED: single-screen review flow, ported RUN thinking theater, no page errors")


if __name__ == "__main__":
    main()
