import os, glob, re

import numpy 

import Bio.SeqIO, Bio.SeqUtils, Bio.Restriction
Bio.Restriction  #To shut up Eclipse warning
import joblib 

import numutils

class Genome():
    '''A Genome object contains the cached properties of a genome.

    Glossary
    --------

    positional identifier : a number that can be decomposed into a tuple 
        (chromosome index, base pair).

    chromosome string label : the name of a chromosome. 
        Examples: 'X', 'Y', '1', '22'.

    chromosome index : a zero-based numeric index of a chromosome.
        For numbered chromosomes it is int(label) - 1, unless some of the 
        chromosomes are absent. The chromosomes 'X', 'Y', 'M' are indexed 
        after, the rest is indexed in alphabetical order.

    concatenated genome : a genome with all chromosomes merged together
        into one sequence.

    binned genome : a genome splitted into bins `resolution` bp each.

    binned concatenated genome : a genome with chromosomes binned and merged.
        GOTCHA: since the genome is binned FIRST and merged after that, the 
        number of bins may be greater than (sum of lengths / resolution).
        The reason for this behavior is that the last bins in the chromosomes
        are usually shorter than `resolution`, but still count as a full bin.

    Attributes
    ----------

    chrmCount : int
        the total number of chromosomes.

    chrmLabels : list of str
        a list of chromosomal IDs sorted in ascending index order.

    fastaNames : list of str
        FASTA files for sorted in ascending index order of respective
        chromosomes.

    genomePath : str
        The path to the folder with the genome.

    name : str
        The string identifier of the genome, the name of the last folder in
        the path.

    label2idx : dict
        a dictionary for conversion between string chromosome labels and
        zero-based indices.

    idx2label : dict
        a dictionary for conversion between zero-based indices and
        string chromosome labels.

    seqs : list of str
        a list of chromosome sequences. Loads on demand.

    chrmLens : list of int
        The lengths of chromosomes.
       
    maxChrmLen : int
        The length of the longest chromosome.

    cntrStarts : array of int
        The start positions of the centromeres.

    cntrMids : array of int
        The middle positions of the centromeres.

    cntrEnds : array of int
        The end positions of the centromeres.

    -------------------------------------------------------------------------------

    The following attributes are calculated after setResolution() is called:

    Attributes
    ----------

    resolution : int
        The size of a bin for the binned values.

    chrmLensBin : array of int
        The lengths of chromosomes in bins.

    chrmStartsBinCont : array of int
        The positions of the first bins of the chromosomes in the 
        concatenated genome.
        
    chrmEndsBinCont : array of int
        The positions of the last plus one bins of the chromosomes in the 
        concatenated genome.

    chrmIdxBinCont : array of int
        The index of a chromosome in each bin of the concatenated genome.

    posBinCont : array of int
        The index of the first base pair in a bin in the concatenated 
        genome.

    cntrMidsBinCont : array of int
        The position of the middle bin of a centromere in the concatenated
        genome.

    GCBin : list of arrays of float
        % of GC content of bins in individual chromosomes.

    -------------------------------------------------------------------------------
        
    The following attributes are calculated after setEnzyme() is called:

    Attributes
    ----------

    rsites : list of arrays of int
        The indices of the first base pairs of restriction fragments 
        in individual chromosomes.

    rfragMids : list of arrays of int
        The indices of the middle base pairs of restriction fragments
        in individual chromosomes.
        
    rsiteIds : array of int
        The position identifiers of the first base pairs of restriction 
        fragments.

    rsiteMidIds : array of int
        The position identifiers of the middle base pairs of restriction 
        fragments.

    rsiteChrms : array of int 
        The indices of chromosomes for restriction sites in corresponding 
        positions of rsiteIds and rsiteMidIds.

    '''

    def _memoize(self, func_name):
        '''Local version of joblib memoization.
        The key difference is that _memoize() takes into account only the 
        relevant attributes of a Genome object (folder, name, gapFile, 
        chrmFileTemplate) and ignores the rest.

        The drawback is that _memoize() doesn't check for the changes in the
        code of the function!'''
        if not hasattr(self, '_mymem'):
            self._mymem = joblib.Memory(cachedir = self.genomePath)

        def run_func(genomePath, readChrms, gapFile, chrmFileTemplate,
                     func_name, *args, **kwargs):
            return getattr(self, func_name)(*args, **kwargs)

        mem_func = self._mymem.cache(run_func)

        def memoized_func(*args, **kwargs):
            return mem_func(
                self.genomePath, self.readChrms, self.gapFile,
                self.chrmFileTemplate, func_name, *args, **kwargs)

        return memoized_func

    def clearCache(self):
        '''Delete the cached data in the genome folder.'''
        if hasattr(self, '_mymem'):
            self._mymem.clear()

    def __init__(self, genomePath, gapFile = 'gap.txt', chrmFileTemplate = 'chr%s.fa',
                 readChrms = ['#', 'X', 'Y', 'M']):
        '''Load a FASTA genome and calculate its properties.
       
        Parameters
        ----------

        genomePath : str
            The path to the folder with the FASTA files.

        gapFile : str
            The path to the gap file relative to genomePath.

        chrmFileTemplate : str
            The template of the FASTA file names.

        readChrms : list of str
            The list with the string labels of chromosomes to read from the 
            genome folder. '#' stands for chromosomes with numerical labels 
            (e.g. 1-22 for human).
        '''
        # Set the main attributes of the class.
        self.genomePath = os.path.abspath(genomePath)
        self.readChrms = set(readChrms)

        self.folderName = os.path.split(self.genomePath)[-1]

        self.gapFile = gapFile

        self.chrmFileTemplate = chrmFileTemplate

        # Scan the folder and obtain the list of chromosomes.
        self._scanGenomeFolder()

        # Get the lengths of the chromosomes.
        self.chrmLens = self.getChrmLen()   
        self.maxChrmLen = max(self.chrmLens)  
        self.fragIDmult = self.maxChrmLen + 1000   #to be used when calculating fragment IDs for HiC  

        # Parse a gap file and mark the centromere positions.
        self._parseGapFile()  

    def _scanGenomeFolder(self):
        self.fastaNames = [os.path.join(self.genomePath, i)
            for i in glob.glob(os.path.join(
                self.genomePath, self.chrmFileTemplate % ('*',)))]

        if len(self.fastaNames) == 0: 
            raise('No Genome files found')

        # Read chromosome IDs.
        self.chrmLabels = []
        for i in self.fastaNames: 
            chrm = re.search(self.chrmFileTemplate % ('(.*)',), i).group(1)
            if ((chrm.isdigit() and '#' in self.readChrms)
                or chrm in self.readChrms):
                self.chrmLabels.append(chrm)
    
        # Convert IDs to indices:
        # A. Convert numerical IDs.
        num_ids = [i for i in self.chrmLabels if i.isdigit()]
        # Sort IDs naturally, i.e. place '2' before '10'.
        num_ids.sort(key=lambda x: int(re.findall(r'\d+$', x)[0]))
        
        self.chrmCount = len(num_ids)
        self.label2idx = dict([(num_ids[i], int(i)) for i in xrange(len(num_ids))])
        self.idx2label = dict([(int(i), num_ids[i]) for i in xrange(len(num_ids))])

        # B. Convert non-numerical IDs. Give the priority to XYM over the rest.
        nonnum_ids = [i for i in self.chrmLabels if not i.isdigit()]
        for i in ['M', 'Y', 'X']:
            if i in nonnum_ids:
                nonnum_ids.pop(nonnum_ids.index(i))
                nonnum_ids.insert(0, i)

        for i in nonnum_ids:
            self.label2idx[i] = self.chrmCount
            self.idx2label[self.chrmCount] = i
            self.chrmCount += 1

        # Sort fastaNames and self.chrmLabels according to the indices:
        self.chrmLabels = zip(*sorted(self.idx2label.items(),
                                      key=lambda x: x[0]))[1]
        self.fastaNames = [
            os.path.join(self.genomePath, self.chrmFileTemplate % i)
            for i in self.chrmLabels]

    def getChrmLen(self):
        # At the first call redirects itself to a memoized private function.
        self.getChrmLen = self._memoize('_getChrmLen')
        return self.getChrmLen()

    def _getChrmLen(self):
        return numpy.array([len(self.seqs[i]) 
                            for i in xrange(0, self.chrmCount)])     
    @property 
    def seqs(self):
        if not hasattr(self, "_seqs"): 
            self._seqs = []
            for i in xrange(self.chrmCount):
                self._seqs.append(Bio.SeqIO.read(open(self.fastaNames[i]), 
                                                 'fasta'))
        return self._seqs

    def _parseGapFile(self):
        """Parse a .gap file to determine centromere positions.
        """
        try: 
            gapFile = open(os.path.join(self.genomePath, self.gapFile)
                           ).readlines()
        except IOError: 
            print "Gap file not found! \n Please provide a link to a gapfile or put a file genome_name.gap in a genome directory"
            exit() 

        self.cntrStarts = -1 * numpy.ones(self.chrmCount,int)
        self.cntrEnds = -1 * numpy.zeros(self.chrmCount,int)

        for line in gapFile:
            splitline = line.split()
            if splitline[7] == 'centromere':
                chrm_str = splitline[1][3:]
                if chrm_str in self.label2idx:
                    chrm_idx = self.label2idx[chrm_str]
                    self.cntrStarts[chrm_idx] = int(splitline[2])
                    self.cntrEnds[chrm_idx] = int(splitline[3])

        self.cntrMids = (self.cntrStarts + self.cntrEnds) / 2
        lowarms = numpy.array(self.cntrStarts)
        higharms = numpy.array(self.chrmLens) - numpy.array(self.cntrEnds)
        self.maxChrmArm = max(lowarms.max(), higharms.max())
            
    def setResolution(self, resolution):
        self.resolution = int(resolution)

        # Bin chromosomes.
        self.chrmLensBin = self.chrmLens / self.resolution + 1 
        self.chrmStartsBinCont = numpy.r_[0, numpy.cumsum(self.chrmLensBin)[:-1]]
        self.chrmEndsBinCont = numpy.cumsum(self.chrmLensBin)
        self.numBins = self.chrmEndsBinCont[-1]

        self.chrmIdxBinCont = numpy.zeros(self.numBins, int)
        for i in xrange(self.chrmCount):
            self.chrmIdxBinCont[
                self.chrmStartsBinCont[i]:self.chrmEndsBinCont[i]] = i

        self.posBinCont = numpy.zeros(self.numBins, int)        
        for i in xrange(self.chrmCount):
            self.posBinCont[
                self.chrmStartsBinCont[i]:self.chrmEndsBinCont[i]] = (
                    self.resolution
                    * numpy.arange(- self.chrmStartsBinCont[i] 
                                   + self.chrmEndsBinCont[i]))

        # Bin centromeres.
        self.cntrMidsBinCont = (self.chrmStartsBinCont
                                + self.cntrMids / self.resolution)

        # Bin GC content.
        self.GCBin = self.getGCBin(self.resolution)

    def splitByChrms(self, inArray):
        return [inArray[self.chrmStartsBinCont[i]:self.chrmEndsBinCont[i]]
                for i in xrange(self.chrmCount)]
                    
    def getGC(self, chrmIdx, start, end):
        "Calculate the GC content of a region."
        seq = self.seqs[chrmIdx][start:end]
        return Bio.SeqUtils.GC(seq.seq)

    def getGCBin(self, resolution):
        # At the first call the function rewrites itself with a memoized 
        # private function.
        self.getGCBin = self._memoize('_getGCBin')
        return self.getGCBin(resolution)
    
    def _getGCBin(self, resolution):
        GCBin = []
        for chrm in xrange(self.chrmCount):
            chrmSizeBin = int(self.chrmLens[chrm] // resolution) + 1
            GCBin.append(numpy.ones(chrmSizeBin, dtype=numpy.float))
            for j in xrange(chrmSizeBin):
                GCBin[chrm][j] = self.getGC(
                    chrm, j * int(resolution), (j + 1) * int(resolution))
        return GCBin

    def getRsites(self, enzymeName):
        # At the first call redirects itself to a memoized private function.
        self.getRsites = self._memoize('_getRsites')
        return self.getRsites(enzymeName)

    def _getRsites(self, enzymeName):
        '''Returns: tuple(rsites, rfrags) 
        Finds restriction sites and mids of rfrags for a given enzyme
        
        '''
        
        #Memorized function
        enzymeSearchFunc = eval('Bio.Restriction.%s.search' % enzymeName)
        rsites = []
        rfragMids = []
        for i in xrange(self.chrmCount):
            rsites.append(numpy.r_[
                0, numpy.array(enzymeSearchFunc(self.seqs[i].seq)) + 1, 
                len(self.seqs[i].seq)])
            rfragMids.append((rsites[i][:-1] + rsites[i][1:]) / 2)

        # Remove the first trivial restriction site (0)
        # to equalize the number of points in rsites and rfragMids.
        for i in xrange(len(rsites)):
            rsites[i] = rsites[i][1:]

        return rsites, rfragMids
    
    def setEnzyme(self, enzymeName):
        """Calculates rsite/rfrag positions and IDs for a given enzyme name 
        and memoizes them"""

        self.rsites, self.rfragMids = self.getRsites(enzymeName)

        self.rsiteIds = numpy.concatenate(
            [self.rsites[chrm] + chrm * self.fragIDmult 
             for chrm in xrange(self.chrmCount)])

        self.rfragMidIds = numpy.concatenate(
            [self.rfragMids[chrm] + chrm * self.fragIDmult 
             for chrm in xrange(self.chrmCount)])

        self.rsiteChrms = numpy.concatenate(
            [numpy.ones(len(self.rsites[chrm]), int) * chrm 
             for chrm in xrange(self.chrmCount)])

        assert (len(self.rsiteIds) == len(self.rfragMidIds))
        
    def getFragmentDistance(self, fragments1, fragments2, enzymeName):
        "returns distance between fragments in... fragments. (neighbors = 1, etc. )"
        if not hasattr(self,"rfragMidIds"): 
            self.setEnzyme(enzymeName)

        frag1ind = numpy.searchsorted(self.rfragMids, fragments1)
        frag2ind = numpy.searchsorted(self.rfragMids, fragments2)
        distance = numpy.abs(frag1ind - frag2ind)

        del frag1ind,frag2ind
        ch1 = fragments1 / self.fragIDmult
        ch2 = fragments2 / self.fragIDmult
        distance[ch1 != ch2] = 1000000
        return distance
    
    def getPairsLessThanDistance(self,fragments1,fragments2,cutoffDistance,enzymeName):
        "returns all possible pairs (fragment1,fragment2) with fragment distance less-or-equal than cutoff"
        if not hasattr(self,"rsiteIDs"): self._calculateRsiteIDs(enzymeName)
        f1ID = numpy.searchsorted(self.rfragMidIds,fragments1) - 1
        f2ID = numpy.searchsorted(self.rfragMidIds,fragments2) - 1    
        fragment2Candidates = numpy.concatenate(
            [f1ID + i for i in (range(-cutoffDistance,0) + range(1,cutoffDistance+1))])        
        fragment1Candidates = numpy.concatenate(
            [f1ID for i in (range(-cutoffDistance,0) + range(1,cutoffDistance+1))])        
        mask = numutils.arrayInArray(fragment2Candidates, f2ID) 
        
        fragment2Real = fragment2Candidates[mask]
        fragment1Real = fragment1Candidates[mask]
        return  (self.rfragIDs[fragment1Real],self.rfragIDs[fragment2Real])
        
        
