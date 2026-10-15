import pytest


def test_responsive_css_hygiene():
    """
    Verify that app_multi.py includes proper CSS hygiene:
    - min-width: 0 on columns
    - overflow-x: hidden on page containers
    - max-width: 100% on block-container
    - responsive word-break on chat messages and titles
    - no fixed parent container clipping
    """
    with open("app_multi.py", "r", encoding="utf-8") as f:
        code = f.read()

    # Verify horizontal overflow prevention
    assert "overflow-x: hidden !important;" in code
    assert "max-width: 100vw !important;" in code

    # Verify min-width: 0 on Streamlit columns
    assert "min-width: 0 !important;" in code

    # Verify responsive padding and box-sizing on block container
    assert ".block-container" in code
    assert "box-sizing: border-box !important;" in code

    # Verify word break on chat messages and titles
    assert "word-break: break-word" in code or "word-break: break-word !important;" in code
    assert "overflow-wrap: break-word" in code


def test_page_navigation_and_state_sync():
    """
    Verify pagination and active citation state synchronization logic.
    """
    session_state = {
        "viewer_page": 1,
        "tb_input_page": 1,
        "active_citation": None
    }

    def set_viewer_page(p: int):
        session_state["viewer_page"] = int(p)
        session_state["tb_input_page"] = int(p)

    def on_next_page(max_p: int):
        new_p = min(max_p, session_state["viewer_page"] + 1)
        set_viewer_page(new_p)

    def on_prev_page():
        new_p = max(1, session_state["viewer_page"] - 1)
        set_viewer_page(new_p)

    total_pages = 5

    # Test initial
    assert session_state["viewer_page"] == 1
    assert session_state["tb_input_page"] == 1

    # Test next
    on_next_page(total_pages)
    assert session_state["viewer_page"] == 2
    assert session_state["tb_input_page"] == 2

    # Test next again
    on_next_page(total_pages)
    assert session_state["viewer_page"] == 3
    assert session_state["tb_input_page"] == 3

    # Test prev
    on_prev_page()
    assert session_state["viewer_page"] == 2
    assert session_state["tb_input_page"] == 2

    # Test citation jump
    mock_citation = {
        "citation_id": 4,
        "page_number": 4,
        "snippet": "Sample text snippet"
    }
    set_viewer_page(mock_citation["page_number"])
    session_state["active_citation"] = mock_citation

    assert session_state["viewer_page"] == 4
    assert session_state["tb_input_page"] == 4
    assert session_state["active_citation"]["citation_id"] == 4


@pytest.mark.browser
def test_browser_viewports_and_no_overflow():
    """
    Live browser integration test using Selenium Headless Chrome:
    Verifies that all 5 viewports (1440px, 1280px, 1024px, 768px, 390px)
    render without horizontal overflow (scrollWidth <= clientWidth).
    """
    try:
        import urllib.request
        # Check if Streamlit server is running on localhost:8501
        req = urllib.request.Request("http://localhost:8501", method="HEAD")
        with urllib.request.urlopen(req, timeout=3) as resp:
            server_available = resp.status == 200
    except Exception:
        pytest.skip("Streamlit server not currently active on http://localhost:8501; skipping browser test.")

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import time

    viewports = [
        ("1440px", 1440, 900, "desktop", False),
        ("1280px", 1280, 800, "desktop", False),
        ("1024px", 1024, 768, "medium", False),
        ("768px", 768, 1024, "medium", False),
        ("390px", 390, 844, "mobile", True),
    ]

    for name, w, h, mode, is_mobile in viewports:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        if is_mobile:
            options.add_experimental_option("mobileEmulation", {"deviceMetrics": {"width": w, "height": h, "pixelRatio": 3.0}})
        else:
            options.add_argument(f"--window-size={w},{h}")

        driver = webdriver.Chrome(options=options)
        try:
            driver.get(f"http://localhost:8501/?layout={mode}")
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, '//*[contains(text(), "Cite_RAG_AI") or contains(text(), "Evidence-Linked")]'))
            )
            time.sleep(1.5)

            # Check horizontal overflow
            scroll_w = driver.execute_script("return document.documentElement.scrollWidth")
            client_w = driver.execute_script("return document.documentElement.clientWidth")
            assert scroll_w <= client_w, f"Horizontal overflow detected at {name}: scrollWidth={scroll_w} > clientWidth={client_w}"

            # Check heading presence
            if mode == "desktop":
                headings = ["Document Library", "AI Research Assistant", "Precision Evidence Viewer"]
                for h_text in headings:
                    el = driver.find_element(By.XPATH, f'//*[contains(text(), "{h_text}")]')
                    assert el.is_displayed(), f"Heading '{h_text}' not displayed at {name}!"
            elif mode == "medium":
                headings = ["AI Research Assistant", "Precision Evidence Viewer"]
                for h_text in headings:
                    el = driver.find_element(By.XPATH, f'//*[contains(text(), "{h_text}")]')
                    assert el.is_displayed(), f"Heading '{h_text}' not displayed at {name}!"
            elif mode == "mobile":
                # On mobile tabs, verify tab labels are present
                tab_el = driver.find_element(By.XPATH, '//*[contains(text(), "AI Assistant")]')
                assert tab_el.is_displayed(), f"AI Assistant tab label not displayed at {name}!"

        finally:
            driver.quit()


@pytest.mark.browser
def test_initial_browser_load_no_query_param():
    """
    Verify that http://localhost:8501 (with no query params) loads smoothly and quickly,
    without sandboxed iframe navigation errors or delay loops.
    """
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:8501", method="HEAD")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status != 200:
                pytest.skip("Streamlit not running")
    except Exception:
        pytest.skip("Streamlit not running")

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import time

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1440,900")
    options.set_capability("goog:loggingPrefs", {"browser": "ALL"})

    driver = webdriver.Chrome(options=options)
    try:
        t0 = time.time()
        driver.get("http://localhost:8501")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[contains(text(), "Document Library")]'))
        )
        load_time = time.time() - t0
        assert load_time < 8.0, f"Initial page load took too long: {load_time:.2f}s"

        # Verify no SEVERE console navigation errors
        logs = driver.get_log("browser")
        severe_nav_errors = [
            l for l in logs 
            if l["level"] == "SEVERE" and "navigation" in l.get("message", "").lower()
        ]
        assert len(severe_nav_errors) == 0, f"Found severe navigation errors in console: {severe_nav_errors}"
    finally:
        driver.quit()
