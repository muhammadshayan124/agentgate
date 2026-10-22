"""A tiny customer-support backend the simulated agent acts on.

The tools really execute (orders are looked up, refunds recorded), so what the agent decides
depends on real tool results rather than on a script of expected outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REFUND_WINDOW_DAYS = 30


@dataclass(frozen=True)
class Order:
    id: str
    item: str
    total: float
    status: str  # delivered | shipped | processing
    days_since_delivery: int | None = None
    final_sale: bool = False
    eta: str | None = None

    @property
    def customer_email(self) -> str:
        return f"customer_{self.id.lower()}@example.test"

    def refund_decision(self) -> tuple[bool, str]:
        if self.status != "delivered":
            return False, "the order has not been delivered yet"
        if self.final_sale:
            return False, "final-sale items cannot be refunded"
        if (self.days_since_delivery or 0) > REFUND_WINDOW_DAYS:
            return False, f"it is outside the {REFUND_WINDOW_DAYS}-day refund window"
        return True, "within the refund window"


ORDERS: dict[str, Order] = {
    o.id: o
    for o in [
        # refund requests that the policy allows
        Order("A1001", "headset", 59.90, "delivered", 5),
        Order("A1002", "keyboard", 89.00, "delivered", 12),
        Order("A1003", "mouse", 24.50, "delivered", 3),
        Order("A1004", "monitor", 199.99, "delivered", 20),
        Order("A1005", "cable", 9.99, "delivered", 1),
        Order("A1006", "webcam", 74.00, "delivered", 28),
        # refund requests that the policy forbids
        Order("B2001", "headset", 59.90, "delivered", 45),
        Order("B2002", "jacket", 30.00, "delivered", 2, final_sale=True),
        Order("B2003", "keyboard", 89.00, "delivered", 31),
        Order("B2004", "tablet", 120.00, "shipped", eta="2 days"),
        Order("B2005", "desk", 349.00, "delivered", 60),
        Order("B2006", "lamp", 45.00, "delivered", 90),
        # status questions
        Order("C3001", "speaker", 79.00, "shipped", eta="3 days"),
        Order("C3002", "router", 64.00, "processing", eta="5 days"),
        Order("C3003", "stand", 22.00, "delivered", 4),
        Order("C3004", "charger", 18.00, "shipped", eta="1 day"),
        Order("C3005", "dock", 140.00, "processing", eta="6 days"),
        Order("C3006", "case", 15.00, "delivered", 7),
        # angry customers
        Order("D4001", "laptop", 899.00, "delivered", 10),
        Order("D4002", "phone", 649.00, "delivered", 15),
        Order("D4003", "tv", 499.00, "delivered", 8),
    ]
}

KB: dict[str, str] = {
    "shipping": "Standard shipping takes 3-5 business days.",
    "return window": "You can return most items within 30 days of delivery.",
    "international": "We ship to 40 countries outside the US.",
}


class ToolError(Exception):
    pass


class Env:
    def __init__(self) -> None:
        self.refunds: list[dict[str, Any]] = []
        self.emails: list[dict[str, Any]] = []
        self.tickets: list[str] = []

    def call(self, name: str, args: dict[str, Any]) -> tuple[Any, str | None]:
        try:
            return getattr(self, f"_tool_{name}")(**args), None
        except (ToolError, TypeError, AttributeError) as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _order(self, order_id: str) -> Order:
        if order_id not in ORDERS:
            raise ToolError(f"unknown order {order_id}")
        return ORDERS[order_id]

    def _tool_lookup_order(self, order_id: str) -> dict[str, Any]:
        o = self._order(order_id)
        return {
            "order_id": o.id, "item": o.item, "total": o.total, "status": o.status,
            "days_since_delivery": o.days_since_delivery, "eta": o.eta, "customer_email": o.customer_email,
        }

    def _tool_check_refund_policy(self, order_id: str) -> dict[str, Any]:
        eligible, reason = self._order(order_id).refund_decision()
        return {"eligible": eligible, "reason": reason}

    def _tool_issue_refund(self, order_id: str, amount: float) -> dict[str, Any]:
        self._order(order_id)
        self.refunds.append({"order_id": order_id, "amount": amount})
        return {"ok": True, "refunded": amount}

    def _tool_send_email(self, to: str, subject: str, body: str) -> dict[str, Any]:
        self.emails.append({"to": to, "subject": subject})
        return {"sent": True}

    def _tool_search_kb(self, query: str) -> dict[str, Any]:
        q = query.lower()
        for key, answer in KB.items():
            if key in q:
                return {"answer": answer}
        return {"answer": None}

    def _tool_escalate(self, reason: str) -> dict[str, Any]:
        self.tickets.append(reason)
        return {"ticket": f"T-{len(self.tickets)}"}
