import time

def time_usage(func):
    def wrapper(*args, **kwargs):
        beg_ts = time.time()
        result = func(*args, **kwargs)
        end_ts = time.time()
        print("[INFO] elapsed time: %f" % (end_ts - beg_ts))
        return result
    return wrapper

def mapVal(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# @time_usage
# from memory_profiler import profile
# @profile