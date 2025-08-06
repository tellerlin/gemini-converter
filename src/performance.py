"""
Performance optimization utilities for Gemini Claude Adapter
"""

import time
import json
import hashlib
import asyncio
from typing import Dict, Any, Optional
from cachetools import TTLCache
import logging
from contextlib import asynccontextmanager
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class ResponseCache:
    def __init__(self, enabled: bool, max_size: int, ttl: int, key_prefix: str):
        self.enabled = enabled
        self.max_size = max_size
        self.ttl = ttl
        self.key_prefix = key_prefix
        self.cache = TTLCache(maxsize=self.max_size, ttl=self.ttl)
        self.frequent_cache = TTLCache(maxsize=self.max_size // 4, ttl=self.ttl * 2)
        self.hit_count = 0
        self.miss_count = 0
        self.lock = asyncio.Lock()
    
    def _generate_cache_key(self, request_dict: Dict[str, Any]) -> str:
        cache_data = {
            "model": request_dict.get("model", ""),
            "messages": request_dict.get("messages", []),
            "temperature": request_dict.get("temperature", 1.0),
            "max_tokens": request_dict.get("max_tokens", None),
            "tools": request_dict.get("tools", None)
        }
        json_str = json.dumps(cache_data, sort_keys=True)
        hash_key = hashlib.md5(json_str.encode()).hexdigest()
        return f"{self.key_prefix}:{hash_key}"
    
    def _should_cache(self, request_dict: Dict[str, Any]) -> bool:
        if request_dict.get("stream", False): return False
        if request_dict.get("tools"): return False
        if request_dict.get("temperature", 0.7) > 1.5: return False
        return True
    
    async def get(self, request_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.enabled or not self._should_cache(request_dict): return None
        async with self.lock:
            cache_key = self._generate_cache_key(request_dict)
            cached_data = self.frequent_cache.get(cache_key)
            if cached_data:
                self.hit_count += 1
                return cached_data
            cached_data = self.cache.get(cache_key)
            if cached_data:
                self.hit_count += 1
                self.frequent_cache[cache_key] = cached_data
                return cached_data
            self.miss_count += 1
            return None
    
    async def set(self, request_dict: Dict[str, Any], response_data: Dict[str, Any]) -> None:
        if not self.enabled or not self._should_cache(request_dict): return
        async with self.lock:
            cache_key = self._generate_cache_key(request_dict)
            self.cache[cache_key] = response_data
    
    def get_stats(self) -> Dict[str, Any]:
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        return {
            "enabled": self.enabled,
            "total_requests": total_requests,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 2),
            "cache_size": len(self.cache),
            "frequent_cache_size": len(self.frequent_cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
        }
    
    def clear(self) -> None:
        self.cache.clear()
        self.frequent_cache.clear()
        self.hit_count = 0
        self.miss_count = 0

class PerformanceMonitor:
    def __init__(self):
        self.request_times = deque(maxlen=1000)
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.lock = asyncio.Lock()
    
    async def record_request(self, endpoint: str, duration: float, success: bool = True):
        async with self.lock:
            self.request_times.append(duration)
            self.request_counts[endpoint] += 1
            if not success: self.error_counts[endpoint] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        if not self.request_times:
            return {"total_requests": 0, "avg_response_time": 0, "p95_response_time": 0, "p99_response_time": 0, "endpoint_stats": {}}
        sorted_times = sorted(self.request_times)
        total_requests = len(sorted_times)
        p95_index = int(total_requests * 0.95)
        p99_index = int(total_requests * 0.99)
        p95_time = sorted_times[p95_index] if p95_index < total_requests else sorted_times[-1]
        p99_time = sorted_times[p99_index] if p99_index < total_requests else sorted_times[-1]
        endpoint_stats = {
            endpoint: {
                "request_count": self.request_counts[endpoint],
                "error_count": self.error_counts[endpoint],
                "error_rate": (self.error_counts[endpoint] / self.request_counts[endpoint] * 100) if self.request_counts[endpoint] > 0 else 0
            } for endpoint in self.request_counts
        }
        return {
            "total_requests": total_requests,
            "avg_response_time": round(sum(sorted_times) / total_requests, 3),
            "p95_response_time": round(p95_time, 3),
            "p99_response_time": round(p99_time, 3),
            "endpoint_stats": endpoint_stats
        }

# Initialize variables
response_cache: Optional[ResponseCache] = None
performance_monitor: PerformanceMonitor = PerformanceMonitor()

def initialize_performance_modules(cache_enabled: bool, cache_max_size: int, cache_ttl: int, cache_key_prefix: str):
    global response_cache
    response_cache = ResponseCache(enabled=cache_enabled, max_size=cache_max_size, ttl=cache_ttl, key_prefix=cache_key_prefix)
    logger.info("Performance modules initialized.")

@asynccontextmanager
async def monitor_performance(endpoint: str):
    start_time = time.time()
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        duration = time.time() - start_time
        await performance_monitor.record_request(endpoint, duration, success)

def get_performance_stats() -> Dict[str, Any]:
    cache_stats = response_cache.get_stats() if response_cache else {}
    return {
        "cache_stats": cache_stats,
        "request_stats": performance_monitor.get_performance_stats()
    }
