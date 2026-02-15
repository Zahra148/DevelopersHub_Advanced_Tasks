import json
from datetime import datetime

LOG_FILE = "feedback_log.jsonl"

def save_feedback(message, rating):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "message": message,
        "rating": rating
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
