"""
Gunicorn configuration for production deployment
Usage: gunicorn -c gunicorn_config.py main:app
"""

import multiprocessing
import os

# Server socket
bind = os.getenv("BIND", "0.0.0.0:8000")
backlog = 2048

# Worker processes
workers = int(os.getenv("WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 10000
max_requests_jitter = 500

# Timeouts
timeout = 120
graceful_timeout = 30
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "leaf-disease-api"

# SSL (set these if using HTTPS)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"
# ssl_version = "TLSv1_2"

# Preload app to optimize memory
preload_app = True

# Server mechanics
daemon = False
umask = 0
user = None
group = None
tmp_upload_dir = None

# Hook functions
def on_starting(server):
    """Hook called when Gunicorn arbiter starts."""
    print(f"[Gunicorn] Starting {workers} workers...")

def when_ready(server):
    """Hook called after Gunicorn worker processes are spawned."""
    print(f"[Gunicorn] Gunicorn server is ready. Spawning workers...")

def on_exit(server):
    """Hook called when Gunicorn arbiter exits."""
    print("[Gunicorn] Gunicorn arbiter exited")

def post_worker_init(worker):
    """Hook called after worker initialization."""
    print(f"[Gunicorn] Worker spawned (pid: {worker.pid})")
