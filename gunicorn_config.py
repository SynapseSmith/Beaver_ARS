# Gunicorn Configuration for Beaver ARS
import multiprocessing
import os

# Server Socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker Processes
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 120
keepalive = 5

# Threads
threads = int(os.getenv("GUNICORN_THREADS", 2))

# Process Naming
proc_name = "beaver-ars"

# Server Mechanics
daemon = False
pidfile = "/tmp/gunicorn.pid"
user = None
group = None
tmp_upload_dir = None

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
def on_starting(server):
    print("Starting Beaver ARS server...")

def on_reload(server):
    print("Reloading Beaver ARS server...")

def worker_int(worker):
    print(f"Worker {worker.pid} received INT signal")

def worker_abort(worker):
    print(f"Worker {worker.pid} aborted")

# Preload application for faster worker spawning
preload_app = True

# Server hooks
def post_fork(server, worker):
    server.log.info(f"Worker spawned (pid: {worker.pid})")

def pre_fork(server, worker):
    pass

def pre_exec(server):
    server.log.info("Forked child, re-executing.")

def when_ready(server):
    server.log.info("Server is ready. Spawning workers")

def worker_exit(server, worker):
    server.log.info(f"Worker {worker.pid} exited")
