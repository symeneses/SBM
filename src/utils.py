import functools
import time
import tracemalloc


def monitor(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tracemalloc.start()
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        elapsed_time = toc - tic
        return value, {"elapsed_time": elapsed_time,
                       "current_memory": current_memory,
                       "peak_memory": peak_memory}
    return wrapper_timer
