from fastapi.testclient import TestClient

from kade.chat.service import ChatService
from kade.config import KadeConfig
from kade.runtime import KadeRuntime
from kade.ui.app import create_ui_app


def test_ui_app_startup_wiring() -> None:
    app = create_ui_app(KadeConfig())
    assert app.state.runtime is not None
    assert app.state.chat is not None


def test_dashboard_endpoint_shape() -> None:
    client = TestClient(create_ui_app(KadeConfig()))
    data = client.get('/api/dashboard').json()
    assert 'runtime_summary' in data
    assert 'provider_diagnostics' in data
    assert 'visual_explainability' in data


def test_command_endpoint_passthrough() -> None:
    client = TestClient(create_ui_app(KadeConfig()))
    out = client.post('/api/command', json={'command': 'status'}).json()
    assert out['executed_action'] == 'status'
    assert out['structured_result']['ok'] is True


def test_chat_endpoint_behavior() -> None:
    client = TestClient(create_ui_app(KadeConfig()))
    out = client.post('/api/chat', json={'message': 'What do you think about NVDA here?'}).json()
    assert out['executed_action'] == 'trade_idea'
    assert out['interpreted_intent']['params']['symbol'] == 'NVDA'


def test_natural_language_fallback_when_disabled() -> None:
    runtime = KadeRuntime()
    service = ChatService.build(
        runtime,
        KadeConfig(natural_language_chat_enabled=False),
    )
    reply = service.submit_message('Show me the best setups right now')
    assert reply.interpreted_intent.action == 'status'


def test_sparse_dashboard_rendering_fallback() -> None:
    app = create_ui_app(KadeConfig())
    vm = app.state.runtime.dashboard_payload()
    vm.pop('market_intelligence', None)
    app.state.runtime.state.latest_dashboard = vm
    client = TestClient(app)
    resp = client.get('/')
    assert resp.status_code == 200


def test_no_llm_trade_logic_replacement() -> None:
    runtime = KadeRuntime()
    service = ChatService.build(runtime, KadeConfig())
    reply = service.submit_message('Should I consider a put on NVDA within an hour?')
    payload = reply.deterministic_result['result']['payload']
    assert reply.deterministic_result['result']['deterministic_source'] == 'kade_structured_engines'
    assert 'trigger' in payload
    assert 'invalidation' in payload


def test_runtime_integration_with_handlers_and_history() -> None:
    client = TestClient(create_ui_app(KadeConfig()))
    client.post('/api/chat', json={'message': 'Show NVDA visually'})
    history = client.get('/api/history').json()
    assert len(history['items']) >= 1
    assert history['latest_symbol'] == 'NVDA'
