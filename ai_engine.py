"""Structured AI analysis with a deterministic offline fallback."""

from __future__ import annotations

import json
import os
from typing import Any


def _money(low: float, high: float | None = None) -> str:
    return f"${low:,.2f}" if high is None else f"${low:,.2f} - ${high:,.2f}"


def local_ai_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    signal = payload["decision"]
    fundamentals = payload.get("fundamentals") or {}
    risks = list(signal.get("risks", []))
    if fundamentals.get("revenue_growth") is not None and fundamentals["revenue_growth"] < 0:
        risks.append("Reported revenue growth is negative.")
    invalidation = list(signal.get("exit_alerts", [])) or [
        "A close below the stop zone or 50-day SMA would weaken the setup.",
        "A final score below 45 would trigger an Exit Alert.",
    ]
    return {
        "ticker": payload["ticker"],
        "rating": signal["final_signal"],
        "confidence": round(signal["confidence"]),
        "best_buy_zone": _money(signal["buy_zone_low"], signal["buy_zone_high"]),
        "avoid_zone": f"Below {_money(signal['stop_loss'])} or directly under {_money(signal['resistance'])}",
        "sell_or_exit_zone": _money(signal["sell_zone"]),
        "stop_loss": _money(signal["stop_loss"]),
        "take_profit_1": _money(signal["take_profit_2r"]),
        "take_profit_2": _money(signal["take_profit_3r"]),
        "main_reasons": signal.get("strengths", [])[:5],
        "main_risks": risks[:5],
        "what_would_change_my_mind": invalidation[:5],
        "plain_english_summary": (
            f"{payload['ticker']} is a {signal['final_signal']} with a "
            f"{signal['final_score']:.1f}/100 combined score. Treat the price levels "
            "as educational planning zones, not predictions or trade instructions."
        ),
        "beginner_explanation": (
            "The score combines trend, momentum, volume, risk/reward, fundamentals, "
            "and an optional historical model. Wait for confirmation and treat the stop "
            "as an invalidation level, not a promise."
        ),
        "source": "Local rule-based engine",
    }


def analyze_stock(payload: dict[str, Any], use_openai: bool = True) -> dict[str, Any]:
    """Request strict JSON from OpenAI when configured, otherwise use local rules."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not use_openai or not api_key:
        return local_ai_analysis(payload)
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an educational quant-analysis assistant. Never promise "
                        "profits or issue trade commands. Use only Buy Watch, Sell Watch, "
                        "Exit Alert, Neutral, or Avoid language. Return JSON only with keys: "
                        "ticker, rating, confidence, best_buy_zone, avoid_zone, "
                        "sell_or_exit_zone, stop_loss, take_profit_1, take_profit_2, "
                        "main_reasons, main_risks, what_would_change_my_mind, "
                        "plain_english_summary, beginner_explanation."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
        )
        parsed = json.loads(response.output_text)
        required = {
            "ticker", "rating", "confidence", "best_buy_zone", "avoid_zone",
            "sell_or_exit_zone", "stop_loss", "take_profit_1", "take_profit_2",
            "main_reasons", "main_risks", "what_would_change_my_mind",
            "plain_english_summary", "beginner_explanation",
        }
        if not required.issubset(parsed):
            raise ValueError("AI response omitted required fields.")
        parsed["source"] = "OpenAI structured analysis"
        return parsed
    except Exception as error:
        fallback = local_ai_analysis(payload)
        fallback["source"] += f" (OpenAI unavailable: {error})"
        return fallback
