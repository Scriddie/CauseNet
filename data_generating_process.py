import numpy as np
import pandas as pd
import time

def a_b_c():
    """
    A -> B -> C
    (value, time)
    """
    a = np.random.randint(low=0, high=2)
    a_time = 1

    if a:
        b = 1
    else:
        b = 0
    b_time = 2

    if b:
        c = 1
    else: 
        c= 0
    c_time = 3

    return (a, a_time), (b, b_time), (c, c_time)


if __name__ == "__main__":
    a = []
    b = []
    c = []
    for i in range(100):
        i, j, k = a_b_c()
        a.append(i)
        b.append(j)
        c.append(k)
    df = pd.DataFrame({
        "a": a,
        "b": b,
        "c": c
    })
