import numpy
import warnings
import mirnylib.systemutils
na = numpy.array 
import  scipy.weave,scipy.sparse.linalg, scipy.stats 
from scipy import weave 
import scipy.linalg
from math import cos,log,sin,sqrt 
#-----------------------------
"Mathematical & programming utilities first" 
#-----------------------------

"Numpy-related (programming) utilities"

def generalizedDtype(inObject):
    """"returns generalized dtype of an object
    upscales precision to the system-specific precision (int16 -> int; int32 -> int) 
    Accepts all dtype-compatible objects.
    Upscales bool to int.   
     
    Bool -> int64
    int -> int64
    float -> float64
    complex -> complex     
    
    """
    if type(inObject) == type:
        inObject = numpy.dtype(inObject)    
    if issubclass( type(inObject) , numpy.ndarray):
        inObject  = inObject.dtype                     
    if numpy.issubdtype(inObject,numpy.complex) == True: return numpy.complex
    if numpy.issubdtype(inObject,numpy.float) == True: return numpy.float
    if numpy.issubdtype(inObject,numpy.int) == True: return numpy.int        
    if numpy.issubdtype(inObject,numpy.bool) == True: return numpy.int
    
    raise ValueError("Data  type not known")


def isInteger(inputData):
    return numpy.mod(inputData,1).mean() < 0.00001

def openmpSum(in_array):
    """
    Performs fast sum of an array using openmm  
    """
    a = numpy.asarray(in_array)
    b = numpy.array([1.])
    N = int(numpy.prod(a.shape))
    N  #PyDev warning remover 
    code = r"""     
    int i=0; 
    double sum = 0;     
    omp_set_num_threads(4);
    #pragma omp parallel for      \  
      default(shared) private(i)  \
      reduction(+:sum)                 
        for (i=0; i<N; i++)
              sum += a[i];
    b[0] = sum;
    """    
    weave.inline(code, ['a','N','b'], 
                     extra_compile_args=['-march=native  -O3  -fopenmp ' ],
                     support_code = r"""
    #include <stdio.h>
    #include <omp.h>
    #include <math.h>""",
     libraries=['gomp'])
    return b[0]

    
"Manipulation with numpy arrays"


def fasterBooleanIndexing(array,indexes,output = None,outLen = None, bounds = True):
    """
    A faster way of writing "output = a[my_boolean_array]"
    Is approximately 30-80 % faster, but requires pre-allocation of the output array.  
    
    .. warning :: this function relies on the fact that you know the length of the output array, 
    and either supply it as an output array, or provide outLen. 
    If not supplied, it will be estimated, and function will be as slow as numpy indexing
    
    Parameters
    ----------
    array : numpy.array compatible
        array to be indexed
    indexes : numpy.array compatible
        array if indexes of the same length as array
    output : numpy.array compatible or None, optional 
        output array. If not provided, please provide outLen parameter
        Note that if you create it, use numpy.empty, not numpy.zeros! 
    outLen : int, optional 
        length of output array that must be equal to indexes.sum() 
    bounds : bool, optional 
        check "on the fly" if length of output is indeed equal to indexes.sum() 
        
    """
    array = numpy.asarray(array)
    indexes = numpy.asarray(indexes)    
    N = len(array)
    N  #Eclipse warning removal
    if output == None: 
        if outLen == None: 
            outLen = indexes.sum() 
        output = numpy.empty(outLen, dtype = array.dtype)
    else:
        output = numpy.asarray(output)  
    M = len(output)
    
    returnFlag = numpy.array([0])
    assert indexes.dtype == numpy.bool
    assert len(indexes) == len(array) 
    if bounds == True:     
        code = """
        int j = 0,i=0; 
        for (i = 0;i<N;i++)
        {
        if (indexes[i] == true)
            {
            if (j == M) 
               {
               returnFlag[0] = -1;
               break;
               }
            output[j] = array[i];
            j++;        
            }    
        }
        if (returnFlag[0] != -1) returnFlag[0] = j;
        """
        weave.inline(code,["array","indexes","output","M","N","returnFlag"])
        if returnFlag[0] == -1:
            raise ValueError("Array out of bounds")
        if returnFlag[0] != M:
            raise ValueError("Out array is too big")
    else:      
        code2 = """
        int j = 0,i=0; 
        for (i = 0;i<N;i++)
        {
        if (indexes[i] == true)
            {
            output[j] = array[i];
            j++;        
            }    
        }        
        """
        weave.inline(code2,["array","indexes","output","M","N"])
    return output        
    



    

def rank(x):
    "Returns rank of an array"
    tmp = numpy.argsort(x)
    return na(numpy.arange(len(x)),float).take(tmp.argsort())

def trunc(x,low=0.005,high = 0.005):
    "Truncates top 'low' fraction and top 'high' fraction of an array "    
    lowValue, highValue = numpy.percentile(x,[low*100.,(1-high)*100.])     
    return numpy.clip(x,a_min = lowValue, a_max = highValue)

trunk = mirnylib.systemutils.deprecate(trunc,"trunk")

def externalMergeSort(inDataset, tempDataset,chunkSize = 300000000):
    """
    An in-place merge sort to work with persistent numpy-type arrays, in particular, h5py datasets. 
    
    Accepts two persistent arrays: inDataset, an array to be sorted, and tempDataset - temporary working copy. 
    Returns the result in inDataset. 
    It is a two-pass algorithm, meaning that the entire dataset will be read and written twice to the HDD. Which is not that bad :) 
    """
    
    if (len(inDataset) < 10000) or (len(inDataset) < chunkSize):
        data = numpy.array(inDataset)
        inDataset[:] = numpy.sort(data)
        print "Sorted using default sort"
        return  
    elif chunkSize < 5 * sqrt(len(inDataset)):        
        warnings.warn("Chunk size should be big enough... you provide %d , upscaled to %d" % (chunkSize,5 * int(sqrt(len(inDataset)))))
        chunkSize = 5 * int(sqrt(len(inDataset)))        

    N = len(inDataset) 
    bins = range(0,N,chunkSize) + [N]
    bins = zip(bins[:-1],bins[1:])   #"big chunks" of chunkSize each
    print "Sorting using %d chunks" % (len(bins),) 
    
    #Initial pre-sorting inDataset - tempDataset
    for start,stop in bins: 
        chunk = inDataset[start:stop]
        chunk.sort()        
        tempDataset[start:stop] = chunk
            
    #Determining smaller merge chunk sizes and positions
    #each chunk is separated into chunkSize/numChunks smaller chunks
    #Smaller chunks are called "merge chunks"
    #Further they will be merged one by one   
    M = len(bins) 
    mergeChunkSize = chunkSize / M
    chunkLocations = []
    for start,stop in bins:
        chunkBins = range(start,stop,mergeChunkSize) + [stop]  
        chunkBins = zip(chunkBins[:-1],chunkBins[1:])
        chunkLocations.append(chunkBins[::-1])   #first chunk is last, as we use pop() later
    outputPosition = 0  #location in the output file, inDataset now 
    
    currentChunks = []   #a set of smaller merge-chunks for each big chunk. 
    for chunk in chunkLocations:
        start,end = chunk.pop()
        currentChunks.append(tempDataset[start:end])
    maxes = [i.max() for i in currentChunks]  #A set of maximum values of working chunks    
    positions = [0 for _ in currentChunks]    #A set of current positions in working chunks

    while True:        
        chInd = numpy.argmin(maxes)  #An index of a chunk that has minimum maximum value right now
          
        #We can't merge beyound this value, we need to update this chunk before doing this
                                  
        limits = [numpy.searchsorted(chunk,maxes[chInd], side = "right") for chunk in currentChunks]
        #Up to where can we merge each chunk now? Find it using searchsorted
         
        currentMerge  = numpy.concatenate([chunk[st:ed] for chunk,st,ed in zip(currentChunks,positions,limits)])
        #Create a current array to merge
         
        positions = limits #we've merged up to here 
        currentMerge.sort()  #Poor man's merge 
        inDataset[outputPosition:outputPosition + len(currentMerge)] = currentMerge  #write sorted merge output here.
        outputPosition += len(currentMerge)
        
        if len(chunkLocations[chInd]) > 0:  #If we still have merge-chunks in that chunk 
            start,end = chunkLocations[chInd].pop() #pull out a new chunk
            chunk = tempDataset[start:end] #update all the values
            maxes[chInd] = chunk.max() 
            currentChunks[chInd] = chunk
            positions[chInd] = 0
        else: #If we don't have any merge-chunks in this chunk 
            currentChunks.pop(chInd)   #Remove all entrees about this chunk 
            positions.pop(chInd)
            chunkLocations.pop(chInd)
            maxes.pop(chInd)

        if len(currentChunks) == 0:   #If we don't have chunks
            assert outputPosition == N   #happily exit
            return



def _testExternalSort():    
    a = numpy.random.random(150000000)
    from mirnylib.h5dict import h5dict
    t = h5dict() 
    t["1"] = a
    t["2"] = a
      
    print "starting sort"
    import time
    tt = time.time() 
    externalMergeSort(t.get_dataset("1"), t.get_dataset("2"),chunkSize = 30000000)
    print  "time to sort", time.time() - tt
    tt = time.time()  
    dif =  t.get_dataset("1")[::1000000] - numpy.sort(a)[::1000000]
    print "time to do regular sort:", time.time() - tt 
    assert dif.sum() == 0
    print "Test finished successfully!"


def uniqueIndex(data):
    """Returns a binary index of unique elements in an array data.
    This method is very memory efficient, much more than numpy.unique! 
    It grabs only 9 extra bytes per record :) 
    """ 
    
    
    args = numpy.argsort(data)
    index = numpy.zeros(len(data),bool)
    myr = range(0,len(data),len(data)/50+1) + [len(data)]
    for i in xrange(len(myr)-1):
        start = myr[i]
        end = myr[i+1]
        dataslice = data.take(args[start:end],axis = 0)        
        ind = dataslice[:-1] != dataslice[1:]
        index[args[start:end-1]] = ind 
        if end != len(data):
            if data[args[end-1]] != data[args[end]]:
                index[args[end-1]] = True
        else: 
            index[args[end-1]] = True
     
    return index


    
def chunkedUnique(data,chunksize = 5000000,return_index = False):
    "Performs unique of a long, but repetitive array. Set chunksize 5-10 times longer than the number of unique elements"
    
    if len(data) < chunksize * 2: return numpy.unique(data,return_index = return_index)
        
    mytype  = data.dtype  
    bins = range(0,len(data),chunksize) + [len(data)]
    bins = zip(bins[:-1],bins[1:])
    current = numpy.zeros(0,dtype = mytype)    
    if return_index == True: currentIndex = numpy.zeros(0,dtype = int )     
    for start,end in bins:
        chunk = data[start:end]
        if return_index == True:
            chunkUnique,chunkIndex = numpy.unique(chunk, return_index = True)
            chunkIndex += start
            current,extraIndex = numpy.unique(numpy.concatenate([current,chunkUnique]),return_index = True)
            currentIndex = numpy.concatenate([currentIndex,chunkIndex])[extraIndex]
        else: 
            chunkUnique= numpy.unique(chunk)
            current = numpy.unique(numpy.concatenate([current,chunkUnique]))
            
            
    if return_index == True: 
        return current,currentIndex
    else:
        return current 
        

def trimZeros(x):
    "trims leading and trailing zeros of a 1D/2D array"
    if len(x.shape) == 1:
        nz = numpy.nonzero(x)[0]
        return x[nz.min():nz.max()+1]         
    ax1 = numpy.nonzero(numpy.sum(x,axis = 0))[0]    
    ax2 = numpy.nonzero(numpy.sum(x,axis = 1))[0]
    return x[ax1.min():ax1.max()+1, ax2.min() : ax2.max()+1 ]
    
def zoomOut(x,shape):
    "rescales an array preserving the structure and total sum"
    M1 = shape[0]
    M2 = shape[1]
    N1 = x.shape[0]
    N2 = x.shape[1]
    if (N1 < M1) or (N2 < M2):
        d1 = M1/N1 + 1
        d2 = M2/N2 + 1
        
        newx = numpy.zeros((d1*N1,d2*N2))        
        for i in xrange(d1):
            for j in xrange(d2):
                newx[i::d1,j::d2] = x/(1. * d1*d2)
        x = newx
        N1,N2 = N1*d1,N2*d2   #array is bigger now
    
    shift1 = N1/float(M1) + 0.000000001
    shift2 = N2/float(M2) + 0.000000001
    x = numpy.asarray(x,dtype = float)
    tempres = numpy.zeros((M1,N2),float)    
    for i in xrange(N1):
        beg = (i/shift1)
        end = ((i+1)/shift1)
        if int(beg) == int(end): 
            tempres[beg,:] += x[i,:]
        else:
            tempres[beg,:] += x[i,:] * (int(end) - beg) / (end - beg)
            tempres[beg+1,:] += x[i,:] * (end - int(end)) / (end - beg)
    res = numpy.zeros((M1,M2),float)
    for i in xrange(N2):
        beg = (i/shift2)
        end = ((i+1)/shift2)
        if int(beg) == int(end): 
            res[:,beg] += tempres[:,i]
        else:
            res[:,beg] += tempres[:,i] * (int(end) - beg) / (end - beg)
            res[:,beg+1] += tempres[:,i] * (end - int(end)) / (end - beg)
    return res


smartZoomOut = mirnylib.systemutils.deprecate(zoomOut,"smartZoomOut")  #backwards compatibility

def coarsegrain(array,size,extendEdge = False):
    "coarsegrains array by summing values in sizeXsize squares; truncates the unused squares"
    array = numpy.asarray(array, dtype = generalizedDtype(array.dtype) )
    
    
    if extendEdge == False:  
        if len(array.shape) == 2:
            N = len(array) - len(array) % size 
            array = array[:N,:N]
            a = numpy.zeros((N/size,N/size),float)
            for i in xrange(size):
                for j in xrange(size):
                    a += array[i::size,j::size]
            return a
        if len(array.shape) == 1:
            array = array[:(len(array) / size) * size]
            narray = numpy.zeros(len(array)/size,float)
            for i in xrange(size):
                narray += array[i::size]
            return narray
    else: 
        N = len(array) 
        if len(array.shape) == 2:
            resultSize = numpy.ceil(float(N) / size )            
            a = numpy.zeros((resultSize,resultSize),float)
            for i in xrange(size):
                for j in xrange(size):
                    add = array[i::size,j::size] 
                    a[:add.shape[0],:add.shape[1]] += add 
            return a
        if len(array.shape) == 1:
            resultSize = numpy.ceil(float(N) / size )            
            a = numpy.zeros((resultSize),float)
            for i in xrange(size):                
                add = array[i::size] 
                a[:len(add)] += add
            return a 



def partialCorrelation(x,y,z,corr = lambda x,y:scipy.stats.spearmanr(x,y)[0] ):
    xy,xz,yz = corr(x,y),corr(x,z),corr(y,z)
    return (xy - xz*yz) / (sqrt(1 - xz**2) * sqrt(1 - yz**2))

"Array indexing-related utilities, written in numpy/c++"
    
def arraySearch(array,tosearch):
    "returns location of tosearch in array; -->> assumes that elements exist!!! <--- " 
    inds = numpy.argsort(array)
    arSorted = array.take(inds,axis= 0)
    newinds = numpy.searchsorted(arSorted[:-1],tosearch)    
    return inds[newinds]

def _arrayInArray(array,filterarray):    
    "Actual implementation of arrayInArray"
    mask = numpy.zeros(len(array),'bool')       
    args = numpy.argsort(array)  
    arsort = array.take(args,axis = 0)    
    diffs = numpy.r_[0,numpy.nonzero(numpy.diff(arsort) )[0]+1,len(arsort)]  #places where sorted values of an array are changing
    values = arsort.take(diffs[:-1])  #values at that places    
    allinds = numpy.searchsorted(values[:-1],filterarray)   
    exist = values.take(allinds) == filterarray                 #check that value in filterarray exists in array    
    N = len(allinds)    
    N,args,exist  #Eclipse warning remover 
    code = r"""
    #line 54 "numutils"
    using namespace std;
    for (int i = 0; i < N; i++)
    {    
        if (exist[i] == 0) continue;
        for (int j=diffs[allinds[i]];j<diffs[allinds[i]+1];j++)
        {
            mask[args[j]] = true;
        }
    } 
    """
    weave.inline(code, ['allinds', 'diffs' , 'mask' ,'args','N','exist'], 
                 extra_compile_args=['-march=native -malign-double -O3'],
                 support_code = r"#include <math.h>" )
    return mask

def arrayInArray(array,filterarray,chunkSize = "auto"):
    """gives you boolean array of indices of elements in array that are contained in filterarray
    a faster version of  [(i in filterarray) for i in array]
    In fact, it's faster than numpy's buildin function, that is written in pure numpy but uses 2 argsorts instead of one
    
    This method is additionaly optimized for large array and short filterarray!\n    
    Actual implementation is in the _arrayInArray"""           #sorted array                    
    array = numpy.asarray(array)
    filterarray = numpy.unique(filterarray)
    if chunkSize == "auto":
        chunkSize = max(4 * len(filterarray),50000)
    if len(array) < 2.5 * chunkSize: 
        return _arrayInArray(array, filterarray)
    
    mask = numpy.zeros(len(array),'bool')
    N = len(array)
    chunks =  range(0,N,chunkSize) + [N]
    bins = zip(chunks[:-1],chunks[1:])
    for start,end in bins:
        mask[start:end] = _arrayInArray(array[start:end], filterarray)
    return mask 
        
def _testArayInArray():
    a = numpy.random.randint(0,1000000,10000000)
    b = numpy.random.randint(0,1000000,500000)
    arrayInArray(a, b)
    import time 
    tt = time.time() 
    res1 = arrayInArray(a, b)
    print "optimized way: ", time.time() - tt
    tt = time.time()
    res2 = _arrayInArray(a,b) 
    print "standard way: ", time.time() - tt
    tt = time.time()
    res3 = numpy.in1d(a, b)
    print "numpy way: ", time.time() - tt
    assert (res1 != res2).sum() == 0
    assert (res1 != res3).sum() == 0
    print "arrayInArray test finished successfully "
    

    
            
def _arraySumByArray(array,filterarray,weightarray):
    "faster [sum(weightarray [array == i]) for i in filterarray]"
    array = numpy.asarray(array,dtype = float)
    weightarray = numpy.asarray(weightarray,dtype = float)    
    if len(array) != len(weightarray): raise ValueError    
    args = numpy.argsort(array)
    arsort = array.take(args)
    diffs = numpy.r_[0,numpy.nonzero(numpy.diff(arsort) > 0.5)[0]+1,len(arsort)]
    values = arsort.take(diffs[:-1])
    allinds = numpy.searchsorted(values[:-1],filterarray)
    exist = values.take(allinds) == filterarray
    N = len(allinds)
    args,exist,N #Eclipse warning removal 
    ret = numpy.zeros(len(allinds),float)    
    code = """
    #line 277 "numutils"
    using namespace std;
    for (int i = 0; i < N; i++)
    {
        if (exist[i] == 0) continue; 
        for (int j=diffs[allinds[i]];j<diffs[allinds[i]+1];j++)
        {
            ret[i] += weightarray[args[j]];
        }
    } 
    """
    support = """
    #include <math.h>  
    """
    weave.inline(code, ['allinds', 'diffs' , 'args' , 'ret','N','weightarray','exist'], extra_compile_args=['-march=native -malign-double -O3'],support_code =support )
    return ret


def arraySumByArray(array,filterarray,meanarray):
    """faster [sum(array == i) for i in filterarray]
    Current method is a wrapper that optimizes this method for speed and memory efficiency.
    """
    if (len(array) / len(filterarray) > 4) and (len(array) > 20000000):
        M = len(array)/len(filterarray) + 1
        chunkSize = min(len(filterarray) * M,10000000)        
        bins = range(0,len(array),chunkSize) + [len(array)]
        toreturn = numpy.zeros(len(filterarray),float)
        for i in xrange(len(bins)- 1):
            toreturn += _arraySumByArray(array[bins[i]:bins[i+1]], filterarray,meanarray[bins[i]:bins[i+1]])
        return toreturn
    else:
        return _arraySumByArray(array, filterarray)
    

def _testArraySumByArray():    
    a = numpy.random.randint(0,1000000,30000000)
    b = numpy.random.randint(0,1000000,500000)
    c = numpy.random.random(30000000)
    
    r1 = _arraySumByArray(a, b, c)
    r2 = arraySumByArray(a, b, c)
    
    dif =  (r1 - r2).sum()
    print "Difference is {0}, should be less than 1e-10".format(dif)
    assert dif < 1e-10
                
#_testArraySumByArray()


def _sumByArray(array,filterarray, dtype = "int64"):
    "actual implementation of sumByArray"            
    arsort = numpy.sort(array)    
    diffs = numpy.r_[0,numpy.nonzero(numpy.diff(arsort) > 0.5)[0]+1,len(arsort)]
    if dtype != None: diffs = numpy.array(diffs, dtype = dtype)
    values = arsort.take(diffs[:-1])
    del arsort        
    allinds = numpy.searchsorted(values[:-1],filterarray)
    notexist = values.take(allinds) != filterarray    
    del values
    c = diffs.take(allinds + 1) - diffs[allinds]
    c[notexist] = 0 
    return c


def sumByArray(array,filterarray,dtype = "int64"):
    """faster [sum(array == i) for i in filterarray]
    Current method is a wrapper that optimizes this method for speed and memory efficiency.
    """
    if (len(array) / len(filterarray) > 2) and (len(array) > 20000000):
        M = len(array)/len(filterarray) + 1
        chunkSize = min(len(filterarray) * M,10000000)        
        bins = range(0,len(array),chunkSize) + [len(array)]
        toreturn = numpy.zeros(len(filterarray),array.dtype)
        for i in xrange(len(bins)- 1):
            toreturn += _sumByArray(array[bins[i]:bins[i+1]], filterarray, dtype)
        return toreturn
    else:
        return _sumByArray(array, filterarray, dtype)
                

"Mathematical utilities"

def corr2d(x):
    "FFT-based 2D correlation"
    x = numpy.array(x)
    t = numpy.fft.fft2(x)
    return numpy.real(numpy.fft.ifft2(t*numpy.conjugate(t)))

def logbins(a, b, pace, N_in=0):
    "create log-spaced bins"
    a = int(a)
    b = int(b) 
    beg = log(a)
    end = log(b - 1)
    pace = log(pace)
    N = int((end - beg) / pace)
     
    if N > 0.8 * (b-a): 
        return numpy.arange(a,b+1)
    
    if N_in != 0: N = N_in  
    pace = (end - beg) / N
    mas = numpy.arange(beg, end + 0.000000001, pace)
    ret = numpy.exp(mas)
    ret = numpy.array([int(i) for i in ret])
    ret[-1] = b 
    for i in xrange(len(ret) - 1):
        if ret[i + 1] <= ret[i]:
            ret[i + 1: - 1] += 1
    return [int(i) for i in ret]



def rescale(data):
    "rescales array to zero mean unit variance"
    data = numpy.asarray(data,dtype = float)
    return (data - data.mean())/ sqrt(data.var())
    
def autocorr(x):
    "autocorrelation function"
    x = rescale(x)
    result = numpy.correlate(x, x, mode='full')
    return result[result.size/2:]

def rotationMatrix(theta):
    "Calculates 3D rotation matrix based on angles"
    tx,ty,tz = theta    
    Rx = numpy.array([[1,0,0], [0, cos(tx), -sin(tx)], [0, sin(tx), cos(tx)]])
    Ry = numpy.array([[cos(ty), 0, -sin(ty)], [0, 1, 0], [sin(ty), 0, cos(ty)]])
    Rz = numpy.array([[cos(tz), -sin(tz), 0], [sin(tz), cos(tz), 0], [0,0,1]])    
    return numpy.dot(Rx, numpy.dot(Ry, Rz))

def random_on_sphere(r=1):
    while True:        
        a = numpy.random.random(2);
        x1 = 2*a[0]-1
        x2 = 2*a[1]-1
        if x1**2 + x2**2 >1: continue
        t = sqrt(1 - x1**2 - x2**2)
        x = r*2*x1* t
        y = r*2* x2 * t
        z = r*(1 - 2*(x1**2 + x2**2))
        return (x,y,z)

def random_in_sphere(r=1):
    while True:
        a = numpy.random.random(3)*2-1
        if numpy.sum(a**2) < 1:
            return r*a
randomInSphere = random_in_sphere
randomOnSphere = random_on_sphere




        
          
         


#---------------------------------------------------------
"Iterative correction, PCA and other Hi-C related things"
#---------------------------------------------------------

def maskPCA(A,mask):
    "attempts to perform PCA-like analysis of an array with a masked part"    
    from numpy import linalg
    mask = numpy.array(mask, int)
    bmask = mask == 0
    A[bmask] = 0  
    sums = numpy.sum(A,axis = 0)
    means = sums / (1. * numpy.sum(mask, axis = 0))     
    M = (A - means).T
    M[bmask]  = 0
    covs = numpy.zeros((len(M),len(M)),float)
    for i in xrange(len(M)):
        myvector = M[i]
        mymask = mask[i]
        allmask = mask * mymask[None,:]
        tocov = myvector[None,:] * M 
        covsums = tocov.sum(axis = 1)
        masksums = allmask.sum(axis = 1)
        covs[i] = covsums/masksums
    [latent,coeff] = linalg.eig(covs)
    print latent[:4]
    return coeff


def PCA(A, numPCs = 6):
    """performs PCA analysis, and returns 6 best principal components
    result[0] is the first PC, etc"""    
    A = numpy.array(A,float)
    M = (A-numpy.mean(A.T,axis=1)).T 
    covM = numpy.dot(M,M.T)
    [latent,coeff] =  scipy.sparse.linalg.eigsh(covM,numPCs)
    print latent
    return (numpy.transpose(coeff[:,::-1]),latent[::-1])


def EIG(A,numPCs = 3):
    """Performs mean-centered engenvector expansion
    result[0] is the first EV, etc.; 
    by default returns 3 EV 
    """
    A = numpy.array(A,float)    
    M = (A - numpy.mean(A)) # subtract the mean (along columns)
    if (M -M.T).var() < numpy.var(M[::10,::10]) * 0.000001:
        [latent,coeff] = scipy.sparse.linalg.eigsh(M,numPCs)
        print "mode autodetect: hermitian"   #matrix is hermitian
    else: 
        [latent,coeff] = scipy.sparse.linalg.eigs(M,numPCs)
        print "mode autodetect : non-hermitian"   #Matrix is normal
    alatent = numpy.argsort(numpy.abs(latent)) 
    print "eigenvalues are:", latent[alatent]
    coeff = coeff[:,alatent]     
    return (numpy.transpose(coeff[:,::-1]),latent[alatent][::-1])



def project(data,vector):
    "project data on a single vector"
    dot = numpy.sum((data*vector[:,None]),axis = 0)     
    den = (vector * vector).sum()
    return vector[:,None] * (dot / den)[None,:]


def projectOnEigenvalues(data,N=1):
    "projects symmetric data on the first N eigenvalues"
    #TODO: rewrite properly for both symmetric and non-symmetric case 
    meanOfData = numpy.mean(data)
    mdata = data - meanOfData
    symData = 0.5*(mdata + mdata.T)    
    values,vectors = scipy.linalg.eig(symData)
    ndata = 0
    for i in xrange(N):
        ndata += values[i] * vectors[:,i][:,None] * vectors[:,i][None,:]
    return ndata + meanOfData 
    

def correct(y):
    "Correct non-symmetric or symmetirc data once"
    x = numpy.array(y,float)        
    s = numpy.sum(x,axis = 1)
    s /= numpy.mean(s[s!=0])    
    s[s==0] = 1     
    s2 = numpy.sum(x,axis = 0)        
    s2 /= numpy.mean(s2[s2!=0])
    s2[s2==0] = 1
    return x / (s2[None,:] * s[:,None])

def correctInPlace(x):
    "works for non-symmetric and symmetric data"            
    s = numpy.sum(x,axis = 1)
    s /= numpy.mean(s[s!=0])    
    s[s==0] = 1     
    s2 = numpy.sum(x,axis = 0)        
    s2 /= numpy.mean(s2[s2!=0])
    s2[s2==0] = 1    
    x /= (s2[None,:] * s[:,None])


def ultracorrectSymmetricWithVector(x,v = None,M=50,diag = -1):
    """Main method for correcting DS and SS read data. Possibly excludes diagonal.
    By default does iterative correction, but can perform an M-time correction"""    
    totalBias = numpy.ones(len(x),float)
    code = """
    #line 288 "numutils" 
    using namespace std;
    for (int i = 0; i < N; i++)    
    {    
        for (int j = 0; j<N; j++)
        {
        x[N * i + j] = x [ N * i + j] / (s[i] * s[j]);
        }
    } 
    """
    if v == None: v = numpy.zeros(len(x),float)  #single-sided reads
    x = numpy.array(x,float,order = 'C')
    v = numpy.array(v,float,order = "C")
    N = len(x)
    N #Eclipse warning remover     
    for _ in xrange(M):         
        s0 = numpy.sum(x,axis = 1)         
        mask = [s0 == 0]            
        v[mask] = 0   #no SS reads if there are no DS reads here        
        nv = v / (totalBias * (totalBias[mask==False]).mean())
        s = s0 + nv
        for dd in xrange(diag + 1):   #excluding the diagonal 
            if dd == 0:
                s -= numpy.diagonal(x)
            else:
                dia = numpy.diagonal(x,dd)
                #print dia
                s[dd:] -= dia
                s[:-dd] -= dia 
        s /= numpy.mean(s[s0!=0])
        s[s0==0] = 1
        totalBias *= s
        scipy.weave.inline(code, ['x','s','N'], extra_compile_args=['-march=native -malign-double -O3'])  #performing a correction
    corr = totalBias[s0!=0].mean()  #mean correction factor
    x *= corr * corr #renormalizing everything
    totalBias /= corr
    return x,v/totalBias 



def ultracorrectSymmetricByMask(x,mask,M = 50):
    """performs iterative correction excluding some regions of a heatmap from consideration.
    These regions are still corrected, but don't contribute to the sums 
    """
    code = """
    #line 333 "numutils.py"
    using namespace std;    
    for (int i=0;i<N;i++)
    {
        for (int j = 0;j<N;j++)
        {
        if (mask[N * i + j] == 1)
            {
            sums[i] += x[N * i + j];
            counts[i] += 1;             
            }
        }
    }
    float ss = 0;
    float count = 0; 
    for (int i = 0;i<N;i++)
    {    
        if ((counts[i] > 0) && (sums[i] > 0))
        {
            sums[i] = sums[i] / counts[i];
            ss+= sums[i];             
            count+= 1;           
        }
        else
        {
            sums[i] = 1; 
        }
    }
    for (int i = 0;i<N;i++)
    {
        sums[i] /= (ss/count);
    }
    for (int i = 0; i < N; i++)    
    {    
        for (int j = 0; j<N; j++)
        {
        x[N * i + j] = x [ N * i + j] / (sums[i] * sums[j]);
        }
    } 
    """
    support = """
    #include <math.h>  
    """    
    x = numpy.asarray(x,float,order = 'C')
    N = len(x)
    allsums = numpy.zeros(N,float)
    for i in xrange(M):
        sums = numpy.zeros(N,float)
        counts = numpy.zeros(N,float)
        i,counts  #Eclipse warning removal     
        scipy.weave.inline(code, ['x','N','sums','counts','mask'], extra_compile_args=['-march=native -malign-double -O3'],support_code =support )
        allsums *= sums
    return x,allsums


def ultracorrect(x,M=20):
    "just iterative correction of symmetric matrix"
    x = numpy.array(x,float)
    print numpy.mean(x),
    newx = numpy.array(x)
    for _ in xrange(M):         
        correctInPlace(newx)
    print numpy.mean(newx),
    newx /= (1. * numpy.mean(newx)/numpy.mean(x))
    print numpy.mean(newx)
    return newx

def correctBias(y):
    "performs single correction and returns data + bias"
    x = numpy.asarray(y,dtype=float)        
    s = numpy.sum(x,axis = 1)
    s /= numpy.mean(s[s!=0])    
    s[s==0] = 1     
    return x / (s[None,:] * s[:,None]),s

 

def ultracorrectBiasReturn(x,M=20):
    "performs iterative correction and returns bias"
    x = numpy.array(x,float)
    print numpy.mean(x),
    newx = numpy.array(x)
    ball = numpy.ones(len(x),float)
    for _ in xrange(M):
        newx,b = correctBias(newx)
        ball *= b
    print numpy.mean(newx),
    newx /= (1. * numpy.mean(newx)/numpy.mean(x))
    print numpy.mean(newx)
    return newx,ball

def create_regions(a):
    "creates array of start/stop positions of continuous nonzero regions of array a"    
    a = numpy.array(a,int)
    a = numpy.concatenate([numpy.array([0],int),a,numpy.array([0],int)])
    a1 = numpy.nonzero(a[1:] * (1-a[:-1]))[0]
    a2 = numpy.nonzero(a[:-1] * (1-a[1:]))[0]    
    return numpy.transpose(numpy.array([a1,a2]))
    
def observedOverExpected(matrix):
    "Calculates observedOverExpected of any contact map"
    data = numpy.asarray(matrix, dtype = float, order = "C")
    N = data.shape[0]
    bins = logbins(1,N,1.2)
    bins = [(0,1)] + [(bins[i],bins[i+1]) for i in xrange(len(bins)-1)]
    bins = numpy.array(bins,order = "C")
    M = len(bins)
    M #Eclipse warning remover
    code = r"""
    #line 50 "binary_search.py"
    using namespace std;
    for (int bin = 0; bin < M; bin++)
    {
        int start = bins[2 * bin];
        int end = bins[2 * bin + 1];
        
        double ss = 0 ;
        int count   = 0 ;  
        for (int offset = start; offset < end; offset ++)
        {
            for (int j = 0; j < N - offset; j++)
            {
                ss += data[(offset + j) * N + j];
                count += 1;                            
            }
        }
        double meanss = ss / count;
        printf("%lf\n",meanss); 
        for (int offset = start; offset < end; offset ++)
        {
            for (int j = 0; j < N - offset; j++)
            {
                if (meanss !=  0)
                {
                    data[(offset + j) * N + j] /= meanss;                                                             
                    if (offset > 0) {data[(offset + j)  + j*N] /= meanss;}
                }
            }
        }
    }
    
    """
    support = """
    #include <math.h>  
    """
    weave.inline(code, ['data', 'bins' , 'N' ,'M'], extra_compile_args=['-march=native -malign-double -O3'],support_code =support )
    return data


def _test():
    _testArayInArray()
    _testExternalSort()
    _testArraySumByArray