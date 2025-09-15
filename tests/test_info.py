import os
import sys
from pathlib import Path
from starlette.requests import Request

os.environ["SCAN_EXECUTOR_MODE"] = "thread"
sys.path.append(str(Path(__file__).resolve().parents[1]))

from routes import info_page, templates


def test_info_page_uses_template(monkeypatch):
    class DummyResponse:
        def __init__(self, name, context):
            self.template = type("T", (), {"name": name})
            self.context = context

    def dummy_template_response(request, name, context):
        return DummyResponse(name, context)

    monkeypatch.setattr(templates, "TemplateResponse", dummy_template_response)

    request = Request({"type": "http"})
    resp = info_page(request)
    assert resp.template.name == "info.html"
    assert resp.context["active_tab"] == "info"

