# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/data_parallel_intro.html
import os

# Set this true to run model on CPU only
USE_CPU_ONLY = True


flags = os.environ.get("XLA_FLAGS", "")
if USE_CPU_ONLY:
    flags += " --xla_force_host_platform_device_count=8" # Simulate 8 CPU devices

