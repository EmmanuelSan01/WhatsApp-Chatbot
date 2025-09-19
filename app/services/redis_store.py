import redis
import os
from app.config import Config

class RedisConversationStore:
    def __init__(self):
        self.client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            password=Config.REDIS_PASSWORD,
            decode_responses=True
        )

    def get_conversation(self, user_id: str) -> str:
        return self.client.get(f"chat:{user_id}") or ""

    def set_conversation(self, user_id: str, conversation: str, expire_seconds: int = 86400):
        self.client.setex(f"chat:{user_id}", expire_seconds, conversation)

    def clear_conversation(self, user_id: str):
        self.client.delete(f"chat:{user_id}")
