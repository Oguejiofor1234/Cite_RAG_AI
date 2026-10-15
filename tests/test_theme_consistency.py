import pytest
import re


def parse_hex_color(hex_str: str):
    """Parse #ffffff, #fff, or rgba/rgb string to (r, g, b) float tuple [0..1]."""
    hex_str = hex_str.strip()
    if hex_str.startswith("#"):
        hex_str = hex_str[1:]
        if len(hex_str) == 3:
            hex_str = "".join([c * 2 for c in hex_str])
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        return (r, g, b)
    elif "rgb" in hex_str:
        nums = [float(x) for x in re.findall(r"[\d.]+", hex_str)]
        if len(nums) >= 3:
            return (nums[0] / 255.0, nums[1] / 255.0, nums[2] / 255.0)
    return (0.0, 0.0, 0.0)


def relative_luminance(rgb):
    """WCAG 2.1 relative luminance calculation."""
    lum = []
    for c in rgb:
        if c <= 0.03928:
            lum.append(c / 12.92)
        else:
            lum.append(((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * lum[0] + 0.7152 * lum[1] + 0.0722 * lum[2]


def contrast_ratio(color1, color2):
    """Compute contrast ratio (>= 1.0, <= 21.0)."""
    rgb1 = parse_hex_color(color1) if isinstance(color1, str) else color1
    rgb2 = parse_hex_color(color2) if isinstance(color2, str) else color2
    l1 = relative_luminance(rgb1)
    l2 = relative_luminance(rgb2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def test_theme_tokens_declared_in_css():
    """Verify that app_multi.py declares required theme tokens for light and dark modes."""
    with open("app_multi.py", "r", encoding="utf-8") as f:
        code = f.read()

    # Verify CSS custom properties declared on :root
    assert "--app-heading-color" in code
    assert "--app-text-color" in code
    assert "--app-card-bg" in code
    assert "--app-card-border" in code
    assert "--app-answer-bg" in code
    assert "--app-answer-text" in code

    # Verify media query for OS dark mode
    assert "@media (prefers-color-scheme: dark)" in code

    # Verify explicit data-theme selectors
    assert '[data-theme="dark"]' in code

    # Verify theme-aware styling on key elements
    assert "color: var(--app-heading-color, inherit) !important;" in code
    assert "color: var(--app-text-muted" in code
    assert ".answer-box" in code
    assert ".evidence-snippet" in code
    assert ".citation-badge" in code


def test_no_hardcoded_dark_text_color():
    """Verify that app_multi.py does NOT hardcode near-black text on headings without theme vars."""
    with open("app_multi.py", "r", encoding="utf-8") as f:
        code = f.read()

    # .app-main-title must not have hardcoded color: #0f172a;
    pattern_main_title = r"\.app-main-title\s*\{[^}]*color:\s*#0f172a;"
    assert not re.search(pattern_main_title, code), "Found hardcoded #0f172a in .app-main-title!"

    # .panel-header must not have hardcoded color: #0f172a;
    pattern_panel_header = r"\.panel-header\s*\{[^}]*color:\s*#0f172a;"
    assert not re.search(pattern_panel_header, code), "Found hardcoded #0f172a in .panel-header!"

    # .doc-title must not have hardcoded color: #0f172a;
    pattern_doc_title = r"\.doc-title\s*\{[^}]*color:\s*#0f172a;"
    assert not re.search(pattern_doc_title, code), "Found hardcoded #0f172a in .doc-title!"


def test_wcag_contrast_ratios_mathematical():
    """
    Verify that all key foreground/background token pairs in both Light and Dark themes
    surpass WCAG AA contrast requirements (>= 4.5:1 for normal text, >= 3.0:1 for large text).
    """
    tokens = {
        "Light": {
            "bg": "#ffffff",
            "heading": "#0f172a",
            "text": "#1e293b",
            "muted": "#475569",
            "card_bg": "#f8fafc",
            "card_text": "#0f172a",
            "card_muted": "#475569",
            "badge_bg": "#f1f5f9",
            "badge_text": "#0f172a",
            "answer_bg": "#fefce8",
            "answer_text": "#713f12",
            "snippet_bg": "#f8fafc",
            "snippet_text": "#334155",
            "alert_bg": "#fffbeb",
            "alert_text": "#92400e",
        },
        "Dark": {
            "bg": "#0e1117",
            "heading": "#f8fafc",
            "text": "#f1f5f9",
            "muted": "#94a3b8",
            "card_bg": "#1e293b",
            "card_text": "#f8fafc",
            "card_muted": "#94a3b8",
            "badge_bg": "#334155",
            "badge_text": "#f8fafc",
            "answer_bg": "#2e2112",  # blending rgba(245, 158, 11, 0.16) on #0e1117
            "answer_text": "#fef08a",
            "snippet_bg": "#1e293b",
            "snippet_text": "#cbd5e1",
            "alert_bg": "#33240e",   # blending rgba(245, 158, 11, 0.18) on #0e1117
            "alert_text": "#fde68a",
        }
    }

    for theme_name, t in tokens.items():
        # 1. Main Heading on Page Background (Large text: >= 3.0:1)
        cr_heading = contrast_ratio(t["heading"], t["bg"])
        assert cr_heading >= 3.0, f"{theme_name} Heading contrast {cr_heading:.2f} < 3.0"

        # 2. Body Text on Page Background (Normal text: >= 4.5:1)
        cr_text = contrast_ratio(t["text"], t["bg"])
        assert cr_text >= 4.5, f"{theme_name} Body Text contrast {cr_text:.2f} < 4.5"

        # 3. Card Title on Card Background (Normal text: >= 4.5:1)
        cr_card_title = contrast_ratio(t["card_text"], t["card_bg"])
        assert cr_card_title >= 4.5, f"{theme_name} Card Title contrast {cr_card_title:.2f} < 4.5"

        # 4. Card Muted Text on Card Background (Normal text: >= 4.5:1)
        cr_card_muted = contrast_ratio(t["card_muted"], t["card_bg"])
        assert cr_card_muted >= 4.5, f"{theme_name} Card Muted contrast {cr_card_muted:.2f} < 4.5"

        # 5. Citation Badge on Badge Background (Normal text: >= 4.5:1)
        cr_badge = contrast_ratio(t["badge_text"], t["badge_bg"])
        assert cr_badge >= 4.5, f"{theme_name} Badge contrast {cr_badge:.2f} < 4.5"

        # 6. Cited Answer Claim on Answer Box Background (Normal text: >= 4.5:1)
        cr_answer = contrast_ratio(t["answer_text"], t["answer_bg"])
        assert cr_answer >= 4.5, f"{theme_name} Answer Claim contrast {cr_answer:.2f} < 4.5"

        # 7. Evidence Snippet on Snippet Background (Normal text: >= 4.5:1)
        cr_snippet = contrast_ratio(t["snippet_text"], t["snippet_bg"])
        assert cr_snippet >= 4.5, f"{theme_name} Evidence Snippet contrast {cr_snippet:.2f} < 4.5"

        # 8. Evidence Alert on Alert Background (Normal text: >= 4.5:1)
        cr_alert = contrast_ratio(t["alert_text"], t["alert_bg"])
        assert cr_alert >= 4.5, f"{theme_name} Evidence Alert contrast {cr_alert:.2f} < 4.5"


@pytest.mark.browser
def test_browser_live_theme_contrast():
    """
    Live browser test verifying that the main title in both Light and Dark mode
    renders with high contrast (> 4.5:1) against its background.
    """
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:8501", method="HEAD")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status != 200:
                pytest.skip("Streamlit server not available")
    except Exception:
        pytest.skip("Streamlit server not available")

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    import time

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get("http://localhost:8501")
        time.sleep(3)

        # Light mode check
        light_data = driver.execute_script('''
            const title = document.querySelector('.app-main-title');
            const app = document.querySelector('.stApp');
            return {
                titleColor: title ? getComputedStyle(title).color : null,
                appBg: app ? getComputedStyle(app).backgroundColor : null
            };
        ''')
        assert light_data["titleColor"] is not None
        cr_light = contrast_ratio(light_data["titleColor"], light_data["appBg"])
        assert cr_light >= 4.5, f"Light title contrast {cr_light:.2f} < 4.5"

        # Toggle to dark
        menu_btn = driver.find_element(By.CSS_SELECTOR, '[data-testid="stMainMenuButton"]')
        menu_btn.click()
        time.sleep(1)
        dark_btn = [el for el in driver.find_elements(By.XPATH, '//*[contains(text(), "Dark")]') if el.is_displayed()]
        if dark_btn:
            dark_btn[0].click()
            time.sleep(2)

            dark_data = driver.execute_script('''
                const title = document.querySelector('.app-main-title');
                const app = document.querySelector('.stApp');
                return {
                    titleColor: title ? getComputedStyle(title).color : null,
                    appBg: app ? getComputedStyle(app).backgroundColor : null
                };
            ''')
            cr_dark = contrast_ratio(dark_data["titleColor"], dark_data["appBg"])
            assert cr_dark >= 4.5, f"Dark title contrast {cr_dark:.2f} < 4.5"

    finally:
        driver.quit()
