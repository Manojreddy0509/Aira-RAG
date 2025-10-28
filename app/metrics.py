from functools import wraps
import time
from typing import Callable

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQUEST_COUNTER = Counter("requests_total", "Total requests", ["endpoint"])
LATENCY_HIST = Histogram("request_latency_seconds", "Request latency (s)", ["endpoint"], buckets=(0.01,0.05,0.1,0.2,0.5,1,2,5,10))


def track_latency(endpoint_name: str):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                LATENCY_HIST.labels(endpoint=endpoint_name).observe(time.perf_counter() - start)
        return wrapper
    return decorator


def setup_metrics(app: FastAPI):
    @app.get("/metrics")
    def metrics():
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)
