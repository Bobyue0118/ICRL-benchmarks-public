import numpy as np


def a(**kwargs):
    print(kwargs)

def step(a,b=2):
    return a+b

if __name__ == '__main__':
    print(step(2,b=2))

