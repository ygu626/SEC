import multiprocessing as mp
from multiprocessing import set_start_method


# function executed in a new process
def func(i, j):
    return i*j


# protect entry point
if __name__ == '__main__':
    # set the start method to fork
    set_start_method('fork')

    with mp.Pool() as pool:
        f = pool.starmap(func, [(i, j) for i in range(0, 3)
                                       for j in range(2, 5)])

print(f)