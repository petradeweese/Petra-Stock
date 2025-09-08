import app


def test_home_endpoint():
    resp = app.home(request=None)
    assert resp["template"] == "index.html"


def test_scanner_page_endpoint():
    resp = app.scanner_page(request=None)
    assert resp["template"] == "index.html"
