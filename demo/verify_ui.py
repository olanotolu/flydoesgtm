"""Small browser smoke test for the single-screen recorded, no-send showcase."""
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8090"


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(URL, wait_until="domcontentloaded")
        page.keyboard.press("Enter")
        page.wait_for_selector("#review-question", timeout=60000)
        headline = page.locator(".signal-det h2").text_content().strip()
        assert headline.startswith("Anthropic") and len(headline) > 12, f"signal card headline missing: {headline!r}"
        assert page.locator(".signal-tags span").count() >= 1, "expected signal tags on the card"
        assert page.locator(".source-card").count() == 7, "expected seven listening sources"
        assert page.locator("#pack-switch button").count() == 3, "expected three evidence packs"
        assert page.locator(".action-card").count() == 4, "expected four GTM plan steps"
        assert page.locator("#firing-list .firing-row").count() >= 1, "expected firing signal rows"
        assert page.locator("#decision-chip").text_content().strip() != ""
        assert page.locator("#evidence-verdict").text_content().strip() != ""
        page.locator("#think-run").click()
        page.wait_for_selector("#fly-decision:not([hidden])", timeout=30000)
        page.wait_for_timeout(2500)
        assert page.locator("#think-bars .think-bar").count() == 6, "expected six sensory energy bars"
        assert page.locator("#think-readout .think-line").count() >= 3, "expected streamed reasoning lines"
        verdict = page.locator("#fly-decision-verdict").text_content()
        assert verdict.startswith("The fly says"), f"verdict missing: {verdict!r}"
        assert page.locator("#fly-decision-motion li").count() == 3, "expected three motion steps"
        na_sub = page.locator("#firing-sub").text_content()
        assert "active" in na_sub and "spikes" in na_sub
        assert page.locator("#firing-list .firing-row").count() >= 1
        assert "MaleCNS" in page.locator("#activity-title").text_content()
        assert "ms" in page.locator("#activity-window").text_content()
        lit = page.locator("#vision-strip span.is-lit").count()
        assert 0 <= lit <= 16
        assert page.locator("#activity-window").text_content().strip() != ""
        assert page.locator("#activity-play").count() == 1, "expected neuron activity play control"
        assert page.locator("#activity-scrub").count() == 1, "expected neuron activity scrubber"
        assert page.locator("#activity-legend").text_content().strip() != ""
        first_lit = page.locator("#brain-map").get_attribute("data-lit")
        assert first_lit is not None and first_lit.isdigit(), "brain map did not report lit neurons"
        page.locator("#activity-play").click()
        page.wait_for_timeout(1200)
        advanced = page.locator("#activity-step").text_content()
        assert "step" in advanced, f"play did not advance the activity step: {advanced!r}"
        assert page.locator("#activity-play").text_content().strip() in ("▶", "❚❚")
        assert "Hz" in page.locator("#activity-legend").text_content()
        base_note = page.locator("#fly-decision-baseline")
        assert base_note.is_visible() and "baseline" in base_note.text_content(), "expected baseline comparison line"
        page.locator("#pack-switch button[data-pack='warm']").click()
        page.wait_for_timeout(300)
        headline = page.locator(".signal-det h2").text_content().strip()
        assert "Linear" in headline, f"pack switch did not update card: {headline!r}"
        assert page.locator(".source-card").count() == 5, "expected five Linear sources"
        assert page.locator("#fly-decision").get_attribute("hidden") is not None, "decision card did not reset on switch"
        page.locator("#think-run").click()
        page.wait_for_selector("#fly-decision:not([hidden])", timeout=30000)
        page.wait_for_timeout(1500)
        verdict = page.locator("#fly-decision-verdict").text_content()
        assert verdict.startswith("The fly says"), f"warm pack verdict missing: {verdict!r}"
        assert "baseline" in page.locator("#fly-decision-baseline").text_content(), "expected baseline note on warm pack"
        page.locator("#pack-switch button[data-pack='cold']").click()
        page.wait_for_timeout(300)
        headline = page.locator(".signal-det h2").text_content().strip()
        assert "23andMe" in headline, f"cold pack card missing: {headline!r}"
        page.locator("#think-run").click()
        page.wait_for_selector("#fly-decision:not([hidden])", timeout=30000)
        page.wait_for_timeout(1500)
        verdict = page.locator("#fly-decision-verdict").text_content()
        assert verdict.startswith("The fly says"), f"cold pack verdict missing: {verdict!r}"
        page.locator("#pack-switch button[data-pack='anthropic']").click()
        page.wait_for_timeout(300)
        assert "Anthropic" in page.locator(".signal-det h2").text_content(), "switch back to Anthropic failed"
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(400)
        scroll_width = page.evaluate("() => document.documentElement.scrollWidth")
        assert scroll_width <= 390, f"mobile horizontal overflow: {scroll_width}"
        assert not errors, errors
        browser.close()
    print("UI VERIFIED: single-screen review flow, ported RUN thinking theater, no page errors")


if __name__ == "__main__":
    main()
