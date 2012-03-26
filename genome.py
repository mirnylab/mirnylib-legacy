import os, glob, tempfile, re

import numpy 

import Bio.SeqIO, Bio.SeqUtils, Bio.Restriction
import joblib 

import numutils

class Genome():
    '''A Genome object contains the cached properties of a genome.

    Glossary
    --------

    id : str
        a string label of a chromosome. Examples: 'X', 'Y', '1', '22'.

    idx : int
        a zero-based index of a chromosome. 
        For numbered chromosomes it is (id - 1).
        The 'X', 'Y' and 'M' chromosomes are assigned an index after the 
        numbered chromosomes, i.e. for human their indices are 22, 23 and 24.
        The rest of the chromosomes are indexed after XYM in alphabetical order.

    Attributes
    ----------
    ids : list of str
        a list of chromosomal IDs sorted in ascending index order.

    fasta_names : a list of FASTA files sorted in ascending index order.


    '''
    def _memoize(self, func_name):
        '''Local version of joblib memoization.
        The key difference is that _memoize() takes into account only the 
        relevant attributes of a Genome object (folder, name, gapfile, 
        chr_file_template) and ignores the rest.

        The drawback is that _memoize() doesn't check for the changes in the
        code of the function!'''
        if not hasattr(self, '_mymem'):
            self._mymem = joblib.Memory(cachedir = self.genomeFolder)

        def run_func(genomeFolder, genomeName, gapfile, chr_file_template,
                     func_name, *args, **kwargs):
            return getattr(self, func_name)(*args, **kwargs)

        mem_func = self._mymem.cache(run_func)

        def memoized_func(*args, **kwargs):
            return mem_func(
                self.genomeFolder, self.type, self.gapfile, self.chr_file_template,
                func_name, *args, **kwargs)

        return memoized_func

    def clear_cache(self):
        if hasattr(self, '_mymem'):
            self._mymem.clear()

    def __init__(self, genomeFolder, genomeName = None, gapfile = None,
                 chr_file_template = 'chr%s.fa'):
        """Loads a genome and calculates certain genomic parameters"""        
        # Set the main attributes of the class.
        self.genomeFolder = genomeFolder                 

        if genomeName == None: 
            folderName = os.path.split(genomeFolder)[1]
            self.type = folderName
            print "genome name inferred from genome folder name : %s" % self.type
        else: 
            self.type = genomeName  

        if gapfile == None:            
            self.gapfile = os.path.join(self.genomeFolder, "%s.gap" % self.type )            
        else:
            self.gapfile = gapfile

        self.chr_file_template = chr_file_template

        # Reading the basic properties of the chromosomes.
        self._read_fasta_names()
        self.chromosomes = self.loadChromosomeLength()   #loading cached chromosome length
        self.chromosomeLength = self.chromosomes
        self.maxChromLen = max(self.chromosomes)  
        self.fragIDmult = self.maxChromLen + 1000   #to be used when calculating fragment IDs for HiC  
        self._parseGapfile()  # Parsing gap file and mark the centromere positions.

    def _read_fasta_names(self):
        self.fasta_names = [os.path.join(self.genomeFolder, i)
            for i in glob.glob(os.path.join(
                self.genomeFolder, self.chr_file_template % ('*',)))]

        if len(self.fasta_names) == 0: 
            raise('No Genome files found')

        # Read chromosome IDs.
        self.chrm_ids = []
        for i in self.fasta_names: 
            chrm = re.search(self.chr_file_template % ('(.*)',), i).group(1)
            self.chrm_ids.append(chrm)
    
        # Convert IDs to indices:
        # A. Convert numerical IDs.
        num_ids = [i for i in self.chrm_ids if i.isdigit()]
        # Sort IDs naturally, i.e. place '2' before '10'.
        num_ids.sort(key=lambda x: int(re.findall(r'\d+$', x)[0]))
        
        self.chromosomeCount = len(num_ids)
        self.id_to_idx = dict([(num_ids[i], int(i)) for i in xrange(len(num_ids))])
        self.idx_to_id = dict([(int(i), num_ids[i]) for i in xrange(len(num_ids))])

        # B. Convert non-numerical IDs. Give the priority to XYM over the rest.
        nonnum_ids = [i for i in self.chrm_ids if not i.isdigit()]
        for i in ['M', 'Y', 'X']:
            if i in nonnum_ids:
                nonnum_ids.pop(nonnum_ids.index(i))
                nonnum_ids.insert(0, i)

        for i in nonnum_ids:
            self.id_to_idx[i] = self.chromosomeCount
            self.idx_to_id[self.chromosomeCount] = i
            self.chromosomeCount += 1

        # Sort fasta_names and self.chrm_ids according to the indices:
        self.chrm_ids = zip(*sorted(self.idx_to_id.items(), key=lambda x: x[0]))[1]
        self.fasta_names = [
            os.path.join(self.genomeFolder, self.chr_file_template % i)
            for i in self.chrm_ids]

    def loadChromosomeLength(self):
        # At the first call redirects itself to a memoized private function.
        self.loadChromosomeLength = self._memoize('_loadChromosomeLength')
        return self.loadChromosomeLength()

    def _loadChromosomeLength(self):
        self._loadSequence()
        return numpy.array([len(self.genome[i]) 
                            for i in xrange(0, self.chromosomeCount)])     
    
    def _loadSequence(self):
        '''Load genomic sequence if it has not been loaded before.'''
        if hasattr(self,"genome"): 
            return 
        self.genome = {}
        for i in xrange(self.chromosomeCount):
            self.genome[i] = Bio.SeqIO.read(open(self.fasta_names[i]), 'fasta')
        
    def getSequence(self, chromosome, start, end):
        if not hasattr(self, "genome"):
            self._loadSequence()        
        return self.genome[chromosome][start:end]

    def _parseGapfile(self):
        """Parse a .gap file to determine centromere positions.
        """
        try: 
            gapfile = open(os.path.join(self.genomeFolder, self.gapfile)
                           ).readlines()
        except IOError: 
            print "Gap file not found! \n Please provide a link to a gapfile or put a file genome_name.gap in a genome directory"
            exit() 

        self.centromereStarts = -1 * numpy.ones(self.chromosomeCount,int)
        self.centromereEnds = -1 * numpy.zeros(self.chromosomeCount,int)

        for line in gapfile:
            splitline = line.split()
            if splitline[7] == 'centromere':
                chr_id = splitline[1][3:]
                if chr_id in self.id_to_idx:
                    chr_idx = self.id_to_idx[chr_id]
                    self.centromereStarts[chr_idx] = int(splitline[2])
                    self.centromereEnds[chr_idx] = int(splitline[3])

        self.centromeres = (self.centromereStarts + self.centromereEnds) / 2
        lowarms = numpy.array(self.centromereStarts)
        higharms = numpy.array(self.chromosomes) - numpy.array(self.centromereEnds)
        self.maximumChromosomeArm = max(lowarms.max(), higharms.max())
        self.maximumChromosome = max(self.chromosomes)          
            
    def createMapping(self, resolution):
        self.resolution = resolution
        self.chromosomeSizes = self.chromosomes / self.resolution + 1 
        self.chromosomeStarts = numpy.r_[0, numpy.cumsum(self.chromosomeSizes)[:-1]]
        self.centromerePositions = (self.chromosomeStarts 
                                    + self.centromeres / self.resolution)
        self.chromosomeEnds = numpy.cumsum(self.chromosomeSizes)
        self.N = self.chromosomeEnds[-1]
        self.chromosomeIndex = numpy.zeros(self.N,int)
        self.positionIndex = numpy.zeros(self.N,int)        
        for i in xrange(self.chromosomeCount):
            self.chromosomeIndex[self.chromosomeStarts[i]:self.chromosomeEnds[i]] = i
            self.positionIndex[self.chromosomeStarts[i]:self.chromosomeEnds[i]] = (
                numpy.arange(-self.chromosomeStarts[i]+self.chromosomeEnds[i])
                * self.resolution)
                    
    def getGC(self,chromosome,start,end):
        "Calculate the GC content of a region."
        seq = self.getSequence(chromosome,start,end)
        return Bio.SeqUtils.GC(seq.seq)

    def getBinnedGCContent(self, resolution):
        # At the first call redirects itself to a memoized private function.
        self.getBinnedGCContent = self._memoize('_getBinnedGCContent')
        return self.getBinnedGCContent(resolution)
    
    def _getBinnedGCContent(self, resolution):
        binnedGC = []
        for chromNum in xrange(self.chromosomeCount):
            binnedGC.append([])
            for j in xrange(self.chromosomes[chromNum] / resolution + 1):
                binnedGC[chromNum].append(
                    self.getGC(chromNum+1, j*resolution, (j + 1) * resolution))
                print "Chrom:", chromNum, "bin:", j
        return binnedGC

    def getRsitesRfrags(self, enzymeName):
        # At the first call redirects itself to a memoized private function.
        self.getRsitesRfrags = self._memoize('_getRsitesRfrags')
        return self.getRsitesRfrags(enzymeName)

    def _getRsitesRfrags(self, enzymeName):
        """returns: tuple(rsiteMap, rfragMap) 
        Finds restriction sites and mids of rfrags for a given enzyme
        Note that there is one extra rsite at beginning and end of chromosome
        Note that there are more rsites than rfrags (by 1)"""
        
        #Memorized function
        self._loadSequence()
        enzymeSearchFunc = eval('Bio.Restriction.%s.search' % enzymeName)
        rsiteMap = {}
        rfragMap = {}        
        for i in xrange(self.chromosomeCount):
            rsites = numpy.r_[
                0, numpy.array(enzymeSearchFunc(self.genome[i].seq)) + 1, 
                len(self.genome[i].seq)] 
            rfrags = (rsites[:-1] + rsites[1:]) / 2
            rsiteMap[i] = rsites
            rfragMap[i] = rfrags          
        return rsiteMap, rfragMap 
    
    def _calculateRsiteIDs(self, enzymeName):
        """Calculates rsite/rfrag positions and IDs for a given enzyme name 
        and memoizes them"""
        rsiteMap, rfragMap = self.getRsitesRfrags(enzymeName)
        # Now truncating one "fake" rsite at the end of each chr, 
        # so that number of rsites matches number of rfrags.
        for i in rsiteMap:
            rsiteMap[i] = rsiteMap[i][:-1]

        self.rsiteMap = rsiteMap 
        self.rfragMap = rfragMap         

        rsiteIDs = [self.rsiteMap[chrom] + chrm * self.fragIDmult 
                    for chrm in xrange(self.chromosomeCount)]        
        self.rsiteIDs = numpy.concatenate(rsiteIDs)

        rsiteChroms = [numpy.ones(len(self.rsiteMap[chrom]), int) * chrom 
                       for chrm in xrange(self.chromosomeCount)]
        self.rsiteChroms = numpy.concatenate(rsiteChroms)

        rfragIDs = [self.rfragMap[chrom] + chrom * self.fragIDmult 
                    for chrom in xrange(self.chromosomeCount)]
        self.rfragIDs = numpy.concatenate(rfragIDs)

        assert (len(self.rsiteIDs) == len(self.rfragIDs))
        
    def getFragmentDistance(self,fragments1,fragments2,enzymeName):
        "returns distance between fragments in... fragments. (neighbors = 1, etc. )"
        if not hasattr(self,"rsiteIDs"): 
            self._calculateRsiteIDs(enzymeName)

        frag1ind = numpy.searchsorted(self.rsiteIDs,fragments1)
        frag2ind = numpy.searchsorted(self.rsiteIDs,fragments2)
        distance = numpy.abs(frag1ind - frag2ind)

        del frag1ind,frag2ind
        ch1 = fragments1 / self.fragIDmult
        ch2 = fragments2 / self.fragIDmult
        distance[ch1 != ch2] = 1000000
        return distance
    
    def getPairsLessThanDistance(self,fragments1,fragments2,cutoffDistance,enzymeName):
        "returns all possible pairs (fragment1,fragment2) with fragment distance less-or-equal than cutoff"
        if not hasattr(self,"rsiteIDs"): self._calculateRsiteIDs(enzymeName)
        f1ID = numpy.searchsorted(self.rsiteIDs,fragments1) - 1
        f2ID = numpy.searchsorted(self.rsiteIDs,fragments2) - 1    
        fragment2Candidates = numpy.concatenate(
            [f1ID + i for i in (range(-cutoffDistance,0) + range(1,cutoffDistance+1))])        
        fragment1Candidates = numpy.concatenate(
            [f1ID for i in (range(-cutoffDistance,0) + range(1,cutoffDistance+1))])        
        mask = numutils.arrayInArray(fragment2Candidates, f2ID) 
        
        fragment2Real = fragment2Candidates[mask]
        fragment1Real = fragment1Candidates[mask]
        return  (self.rfragIDs[fragment1Real],self.rfragIDs[fragment2Real])
        
        
