import numpy as np
import random

# create an array of random ints
a=np.array([random.randint(-2048, 2047) for _ in range(2048)])

# convert them to complex pairings (first convert to floats)
b=(a/2047).astype(np.float32).view(np.complex64)

# deinterleave
c = [b[0::2], b[1::2]]

print(b[0:4])
print(c[0][0:2])