import redis
import os
from app.config import Config

class RedisCache:
    def __init__(self):
        self.client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            password=Config.REDIS_PASSWORD,
            decode_responses=True
        )

    def get(self, key: str):
        return self.client.get(key)

    def set(self, key: str, value, expire_seconds: int = 3600):
        self.client.setex(key, expire_seconds, value)

    def delete(self, key: str):
        self.client.delete(key)
