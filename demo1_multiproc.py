from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from time import sleep


def x(p):
    a, b = tuple(p)
    print(111111)
    sleep(3)
    print(222222)
    return a * b


if __name__ == '__main__':
    freeze_support()
    with ProcessPoolExecutor(5) as executor:
        for a in executor.map(x, [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]):
            print(a)
