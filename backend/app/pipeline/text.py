"""Small formatting helpers shared by the learning loop and the composer."""

from __future__ import annotations

METHOD_LABEL: dict[str, str] = {
    "customer_interviews": "Customer interviews",
    "landing_page_smoke_test": "Landing-page smoke test",
    "letter_of_intent": "Letter-of-intent request",
    "pricing_test": "Pricing test",
    "technical_benchmark": "Technical benchmark against the incumbent",
    "regulatory_expert_call": "Regulatory expert call",
}


def fmt_money(value: float) -> str:
    """€1.2M, €250k, €900."""
    v = abs(value)
    sign = "-" if value < 0 else ""
    if v >= 1e9:
        return f"{sign}€{v / 1e9:.1f}bn"
    if v >= 1e6:
        return f"{sign}€{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{sign}€{v / 1e3:.0f}k"
    return f"{sign}€{v:.0f}"


def fmt_pct(share: float) -> str:
    return f"{share * 100:.0f}%"
