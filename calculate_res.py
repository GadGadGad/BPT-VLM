import numpy as np

while True:
    a = np.float32(input())
    b = np.float32(input())

    print(f"Avg: {(a + b)/2:.2f}\nStd: {np.std([a, b]):.2f}")