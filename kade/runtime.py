from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

ACTIONS = {
    "status",
    "radar",
    "premarket_gameplan",
    "trade_idea",
    "target_move",
    "trade_plan",
    "trade_plan_check",
    "trade_review",
    "visual_explain",
    "strategy_analysis",
}


@dataclass
class RuntimeState:
    history: list[dict[str, Any]] = field(default_factory=list)
    latest_dashboard: dict[str, Any] = field(default_factory=dict)
    latest_symbol: str = "SPY"
    latest_view: str = "overview"


class KadeRuntime:
    """Deterministic runtime facade for operator workflows."""

    def __init__(self) -> None:
        self.state = RuntimeState()
        self.provider_diagnostics = {
            "active_provider": "alpaca_history",
            "llm_provider": "ollama_llama",
            "routing_mode": "hybrid",
            "llm_available": False,
            "natural_language_mode": True,
        }

    def execute_action(self, action: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = params or {}
        if action not in ACTIONS:
            return {
                "ok": False,
                "error": f"Unsupported action '{action}'",
                "available_actions": sorted(ACTIONS),
            }

        symbol = str(params.get("symbol") or self.state.latest_symbol).upper()
        view = str(params.get("view_type") or self.state.latest_view)
        self.state.latest_symbol = symbol
        self.state.latest_view = view

        result: dict[str, Any] = {
            "action": action,
            "symbol": symbol,
            "view_type": view,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "deterministic_source": "kade_structured_engines",
            "payload": self._payload_for(action, symbol, params),
        }
        self.state.history.append(result)
        self.state.latest_dashboard = self.dashboard_payload()
        return {"ok": True, "result": result}

    def dashboard_payload(self) -> dict[str, Any]:
        symbol = self.state.latest_symbol
        return {
            "runtime_summary": {
                "status": "mock-runtime-active",
                "active_symbol": symbol,
                "active_view": self.state.latest_view,
                "history_count": len(self.state.history),
            },
            "ticker_cards": [
                {"symbol": symbol, "price": 183.12, "change_pct": 0.42},
                {"symbol": "QQQ", "price": 460.71, "change_pct": -0.08},
            ],
            "radar_queue": ["NVDA", "AAPL", "TSLA"],
            "market_intelligence": {
                "regime": "balanced",
                "breadth": "neutral-positive",
                "volatility": "contained",
            },
            "premarket_gameplan": {"focus": ["open range", "relative strength"]},
            "trade_idea": {
                "opinion": "wait_for_confirmation",
                "reason": "momentum below trigger threshold",
            },
            "target_move_board": [
                {"symbol": symbol, "target": 184.2, "probability_bucket": "medium"}
            ],
            "trade_plan": {
                "entry": "break_above_183_50",
                "invalidation": "below_182_20",
                "targets": [184.2, 185.0],
            },
            "trade_plan_tracking": {"phase": "waiting", "checklist_complete": 3},
            "trade_review": {"last_review": "no_live_trade_mode", "score": "n/a"},
            "visual_explainability": {
                "active_symbol": symbol,
                "active_view": self.state.latest_view,
                "timeframes": [
                    {
                        "timeframe": "5m",
                        "blocks": [{"price": 183.0, "label": "pivot"}],
                        "overlays": [
                            {
                                "type": "trigger",
                                "price": 183.5,
                                "reason": "breakout threshold",
                            }
                        ],
                    },
                    {
                        "timeframe": "1h",
                        "blocks": [{"price": 181.8, "label": "support"}],
                        "overlays": [
                            {
                                "type": "target",
                                "price": 184.2,
                                "reason": "measured move",
                            }
                        ],
                    },
                ],
                "summary": "Deterministic chart overlays from Kade engines",
            },
            "strategy_intelligence": {
                "best_patterns": ["opening_drive", "retest_continuation"],
                "sample_size": 42,
            },
            "execution_monitor": {"mode": "paper", "live_trading_enabled": False},
            "timeline": self.state.history[-10:],
            "provider_diagnostics": self.provider_diagnostics,
        }

    def history_payload(self) -> dict[str, Any]:
        return {
            "items": self.state.history,
            "latest_symbol": self.state.latest_symbol,
            "latest_view": self.state.latest_view,
        }

    def _payload_for(self, action: str, symbol: str, params: dict[str, Any]) -> dict[str, Any]:
        if action == "status":
            return {
                "system": "ready",
                "symbol": symbol,
                "providers": self.provider_diagnostics,
                "live_trading": False,
            }
        if action == "trade_idea":
            direction = params.get("direction", "call")
            return {
                "symbol": symbol,
                "direction": direction,
                "opinion": "neutral_to_constructive",
                "trigger": "break_and_hold_183_50",
                "invalidation": "below_182_20",
                "target": "184_20_then_185_00",
            }
        if action == "target_move":
            return {
                "symbol": symbol,
                "current_price": params.get("current_price", 183.12),
                "target_price": params.get("target_price", 184.2),
                "choices": ["momentum_push", "range_expansion"],
            }
        if action == "visual_explain":
            return self.dashboard_payload()["visual_explainability"]
        return {"symbol": symbol, "params": params, "note": f"Executed {action}"}
