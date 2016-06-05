from __future__ import absolute_import, division, print_function, unicode_literals
from mirnylib.genome import Genome

a = Genome("/home/magus/HiC2011/data/hg19", readChrms = ["#","X"])
a.setResolution(200000)
t = a.parseAnyWigFile("/home/magus/HiC2011/conservation/vertebrates.wig")
print (t)
