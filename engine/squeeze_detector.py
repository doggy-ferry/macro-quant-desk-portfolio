"""Score verifiable macro-squeeze inputs without inventing missing market data."""

from __future__ import annotations

import math
from typing import Any


def _finite(value: Any) -> float | None:
    """Reject absent, non-numeric and non-finite inputs (including booleans)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class MacroSqueezeDetector:
    @staticmethod
    def evaluate(market_snapshot: dict) -> dict:
        """Assess four independently sourced signals; missing fields never trigger."""
        snapshot = market_snapshot or {}
        spot = _finite(snapshot.get("spot_price"))
        iv_change = _finite(snapshot.get("iv_change_pct"))
        gamma_level = _finite(snapshot.get("gamma_flip_level"))
        call_flow = _finite(snapshot.get("call_flow_pct"))
        cta_zscore = _finite(snapshot.get("cta_position_zscore"))
        cover_trigger = _finite(snapshot.get("cta_cover_trigger"))
        creation = _finite(snapshot.get("tqqq_net_creation"))

        gamma_available = ((spot is not None and spot > 0 and gamma_level is not None and gamma_level > 0)
                           or (call_flow is not None and 0 <= call_flow <= 1))
        cta_available = (spot is not None and spot > 0 and cta_zscore is not None
                         and cover_trigger is not None and cover_trigger > 0)
        available = {
            "iv_crush": iv_change is not None,
            "gamma_flip": gamma_available,
            "cta_reversal": cta_available,
            "ap_absorption": creation is not None,
        }
        signals = {
            "iv_crush": iv_change is not None and iv_change <= -3.0,
            "gamma_flip": bool(
                (spot is not None and spot > 0 and gamma_level is not None and gamma_level > 0 and spot >= gamma_level)
                or (call_flow is not None and 0 <= call_flow <= 1 and call_flow >= 0.58)
            ),
            "cta_reversal": bool(cta_available and cta_zscore <= -1.2 and spot >= cover_trigger),
            "ap_absorption": creation is not None and creation > 0,
        }
        score = sum(weight for signal, weight in (
            ("iv_crush", 25), ("gamma_flip", 30),
            ("cta_reversal", 25), ("ap_absorption", 20),
        ) if signals[signal])

        if score >= 75:
            status, label, tag_color, alert_needed = (
                "CONFIRMED_SQUEEZE", "Mechanical squeeze confirmed // institutional covering", "#00ff66", True)
        elif score >= 50:
            status, label, tag_color, alert_needed = (
                "EARLY_ABSORPTION", "Early absorption // downside pressure fading", "#00e5ff", False)
        else:
            status, label, tag_color, alert_needed = (
                "REGIME_NORMAL", "No structural squeeze // normal volatility", "#64748b", False)

        def assessment(key: str, positive: str, negative: str) -> str:
            if not available[key]:
                return "Data unavailable"
            return positive if signals[key] else negative

        iv_source = snapshot.get("iv_source") or "ATM IV"
        diagnostic_text = (
            f"Dealer and short-covering squeeze radar: {label} (score: {score}/100) | "
            f"{iv_source} compression: {assessment('iv_crush', 'triggered', 'not triggered')}; "
            f"Gamma structure: {assessment('gamma_flip', 'positive zone / stronger Call flow', 'below threshold')}; "
            f"CTA cover trigger: {assessment('cta_reversal', 'crossed', 'not crossed')}; "
            f"TQQQ primary-market absorption: {assessment('ap_absorption', 'net creation', 'redemption / neutral')}"
        )
        if not all(available.values()):
            diagnostic_text += " (missing inputs are unscored and do not rule out a squeeze)"
        return {
            "score": score, "status": status, "label": label,
            "tag_color": tag_color, "signals": signals,
            "diagnostic_text": diagnostic_text, "alert_needed": alert_needed,
            "available": available,
        }
