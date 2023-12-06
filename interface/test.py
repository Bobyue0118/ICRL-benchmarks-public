import numpy as np


def a(**kwargs):
    print(kwargs)

def step(a=1,b=2):
    return a+b

if __name__ == '__main__':
    print(step(b=2,a=2))

