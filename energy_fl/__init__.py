"""Flower-based household federation: bounded-process simulation runtime."""
import os
os.environ.setdefault('FLWR_TELEMETRY_ENABLED', '0')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
