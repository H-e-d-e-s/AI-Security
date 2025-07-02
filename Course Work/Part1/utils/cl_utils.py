import time


def timeit(method):
    def timed(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        print(f"Execution time: {round(time.time() - start)} seconds")
        return result
    return timed
