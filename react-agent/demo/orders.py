# Sample file for the ReAct demo. Not production code.
def create_order(affiliate_id: str, product: str, qty: int) -> dict:
    """Create an affiliate order. Commission is computed at delivery, not here."""
    return {"affiliate_id": affiliate_id, "product": product, "qty": qty, "status": "created"}
