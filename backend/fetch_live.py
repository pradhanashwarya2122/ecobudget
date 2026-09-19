"""
Fetches a live webpage using headless Chromium (Playwright) instead of
relying on pre-downloaded static HTML snapshots. This lets EcoBudget
be tested against real, currently-live pages, including JS-rendered ones.
"""
from playwright.sync_api import sync_playwright


def fetch_rendered_html(url, wait_ms=2000):
    """
    Loads a URL in headless Chromium, waits for JS to render,
    and returns the final HTML.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, timeout=15000)
        page.wait_for_timeout(wait_ms)  # let JS-rendered content settle
        html = page.content()
        browser.close()
        return html


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://en.wikipedia.org/wiki/Louvre"
    html = fetch_rendered_html(url)
    print(f"Fetched {len(html)} characters from {url}")
    with open("live_test_output.html", "w") as f:
        f.write(html)
    print("Saved to live_test_output.html")
