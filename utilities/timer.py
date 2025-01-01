import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{name} took {elapsed_time:.3f} seconds")