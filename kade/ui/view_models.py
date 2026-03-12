from __future__ import annotations

from typing import Any


def dashboard_vm(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": payload.get("runtime_summary", {}),
        "ticker_cards": payload.get("ticker_cards", []),
        "radar_queue": payload.get("radar_queue", []),
        "market_intelligence": payload.get("market_intelligence", {}),
        "premarket_gameplan": payload.get("premarket_gameplan", {}),
        "trade_idea": payload.get("trade_idea", {}),
        "target_move_board": payload.get("target_move_board", []),
        "trade_plan": payload.get("trade_plan", {}),
        "trade_plan_tracking": payload.get("trade_plan_tracking", {}),
        "trade_review": payload.get("trade_review", {}),
        "visual_explainability": payload.get("visual_explainability", {}),
        "strategy_intelligence": payload.get("strategy_intelligence", {}),
        "execution_monitor": payload.get("execution_monitor", {}),
        "timeline": payload.get("timeline", []),
        "provider_diagnostics": payload.get("provider_diagnostics", {}),
    }
