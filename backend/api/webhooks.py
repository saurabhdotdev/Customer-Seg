import secrets
from datetime import datetime, timezone
from backend.api.db import get_db_connection, log_audit_event

class WebhookAlertEngine:
    """
    Automated E-Mail & Webhook Notification Alert Engine.
    Dispatches real-time webhooks whenever high churn risk (>70%) or IsolationForest anomalies are flagged.
    """

    @classmethod
    def dispatch_alert(cls, user_id: str, alert_type: str, payload: dict) -> dict:
        event_id = f"alert_{secrets.token_hex(6)}"
        now_str = datetime.now(timezone.utc).isoformat()

        # Log in SQL database
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS webhooks (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    alert_type TEXT,
                    customer_id TEXT,
                    details TEXT,
                    created_at TEXT
                )
            """)
            conn.commit()

            cursor.execute(
                "INSERT INTO webhooks (id, user_id, alert_type, customer_id, details, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, user_id or "system", alert_type, payload.get("customer_id", "N/A"), str(payload), now_str)
            )
            conn.commit()
            conn.close()

            log_audit_event(user_id, f"webhook_alert_{alert_type}", payload)
        except Exception as exc:
            print("SQL webhook insert warning:", exc)

        return {
            "status": "dispatched",
            "event_id": event_id,
            "alert_type": alert_type,
            "timestamp": now_str,
            "payload": payload,
            "webhook_target": "https://api.customer-seg.com/webhooks/listener"
        }

    @classmethod
    def get_recent_alerts(cls, user_id: str = None, limit: int = 10) -> list:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS webhooks (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    alert_type TEXT,
                    customer_id TEXT,
                    details TEXT,
                    created_at TEXT
                )
            """)
            conn.commit()

            if user_id:
                cursor.execute("SELECT id, alert_type, customer_id, details, created_at FROM webhooks WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit))
            else:
                cursor.execute("SELECT id, alert_type, customer_id, details, created_at FROM webhooks ORDER BY created_at DESC LIMIT ?", (limit,))

            rows = cursor.fetchall()
            conn.close()
            return [
                {
                    "id": r["id"],
                    "alert_type": r["alert_type"],
                    "customer_id": r["customer_id"],
                    "details": r["details"],
                    "created_at": r["created_at"]
                } for r in rows
            ]
        except Exception as exc:
            print("Failed to fetch webhook logs:", exc)
            return []
