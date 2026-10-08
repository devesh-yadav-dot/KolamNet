"""Smoke test: the app boots on the bundled demo image and renders every tab.

Run with:  pip install pytest && pytest -q
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_runs_with_demo_image():
    at = AppTest.from_file(APP, default_timeout=180).run()

    assert not at.exception
    assert len(at.tabs) == 6

    dots = [m.value for m in at.metric if m.label == "Dots Extracted"]
    assert dots == ["12"]


def test_clean_vector_mode():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value("🎨 Clean Digital Vector / Sketch").run()

    assert not at.exception


def test_neural_button_without_key_shows_error():
    at = AppTest.from_file(APP, default_timeout=180).run()
    next(b for b in at.button if "Neural" in b.label).click().run()

    assert not at.exception
    assert any("API Key" in e.value for e in at.error)
