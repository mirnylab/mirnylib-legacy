from fastExtensionspy import openmmArraySum
import numpy as np 
from mirnylib.numutils import openmpSum 

a = np.random.random(100000000)
print a.sum()
print openmmArraySum(a)
print openmpSum(a)

for i in xrange(100):
    openmmArraySum(a)


