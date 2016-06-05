from __future__ import absolute_import, division, print_function, unicode_literals
from fastExtensionspy import openmmArraySum
import numpy as np 
from mirnylib.numutils import openmpSum 

a = np.random.random(100000000)
print (a.sum())
print (openmmArraySum(a))
print (openmpSum(a))

for i in range(100):
    openmmArraySum(a)


