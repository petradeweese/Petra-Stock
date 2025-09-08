import sys, types, datetime, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Stub pandas
pandas_stub = types.ModuleType('pandas')
sys.modules.setdefault('pandas', pandas_stub)

# Stub pandas_market_calendars
pmc_stub = types.ModuleType('pandas_market_calendars')
class DummyCalendar:
    tz = datetime.timezone.utc
pmc_stub.get_calendar = lambda name: DummyCalendar()
sys.modules.setdefault('pandas_market_calendars', pmc_stub)

# Stub FastAPI and related components
fastapi_stub = types.ModuleType('fastapi')
class FastAPI:
    def __init__(self, *a, **kw):
        pass
    def mount(self, *a, **kw):
        pass
    def get(self, *a, **kw):
        def decorator(fn):
            return fn
        return decorator
    def post(self, *a, **kw):
        def decorator(fn):
            return fn
        return decorator
    def on_event(self, *a, **kw):
        def decorator(fn):
            return fn
        return decorator
fastapi_stub.FastAPI = FastAPI
class Request:
    pass
fastapi_stub.Request = Request
def Form(*a, **kw):
    return None
fastapi_stub.Form = Form
def Depends(dep):
    return dep
fastapi_stub.Depends = Depends
sys.modules.setdefault('fastapi', fastapi_stub)

responses_stub = types.ModuleType('fastapi.responses')
class HTMLResponse(dict):
    def __init__(self, content='', status_code=200):
        super().__init__(content=content)
        self.status_code = status_code
class RedirectResponse(dict):
    def __init__(self, url='', status_code=307):
        super().__init__(url=url)
        self.status_code = status_code
class JSONResponse(dict):
    def __init__(self, content=None, status_code=200):
        if content is None:
            content = {}
        super().__init__(content)
        self.status_code = status_code
responses_stub.HTMLResponse = HTMLResponse
responses_stub.RedirectResponse = RedirectResponse
responses_stub.JSONResponse = JSONResponse
sys.modules.setdefault('fastapi.responses', responses_stub)

static_stub = types.ModuleType('fastapi.staticfiles')
class StaticFiles:
    def __init__(self, directory, name=None):
        self.directory = directory
static_stub.StaticFiles = StaticFiles
sys.modules.setdefault('fastapi.staticfiles', static_stub)

templates_stub = types.ModuleType('fastapi.templating')
class Jinja2Templates:
    def __init__(self, directory):
        self.directory = directory
    def TemplateResponse(self, request, template_name, context):
        return {'template': template_name, **context}
templates_stub.Jinja2Templates = Jinja2Templates
sys.modules.setdefault('fastapi.templating', templates_stub)

# Stub pattern_finder_app to avoid heavy imports
auto_stub = types.ModuleType('pattern_finder_app')
sys.modules.setdefault('pattern_finder_app', auto_stub)
