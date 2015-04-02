#include <math.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
using namespace std; 


void readWigFileCpp(char* filename, double* data, int chromCount,
                            bool useX, bool useY, bool useM,
                            int Xnum, int Ynum, int Mnum, int Mkb, int resolution)
    {
        using namespace std;
        int chrom=1;
        bool skip = false;
        int pos;
        int step;
        char line[100];
        char chromNum[10];
        FILE *myfile;

        myfile = fopen(filename,"r");

        int breakflag = 0;

        while (fgets(line, 100, myfile) != NULL)
        {

          if (line[0] == 'f')
              {

              if (breakflag == 1) break;
              sscanf(line,"fixedStep chrom=chr%s start=%d step=%d",
              &chromNum,&pos,&step);

              skip = false;

              if (strcmp(chromNum ,"X") == 0)
               { chrom = Xnum; if (useX == false) skip = true;}
              else if (strcmp(chromNum ,"Y") == 0)
               { chrom = Ynum; if (useY == false) skip = true;}
              else if (strcmp(chromNum ,"M") == 0)
               { chrom = Mnum; if (useM == false) skip = true;}
              else {chrom = atoi(chromNum);}               
              printf("Chromosome %d \n",chrom);               
              if ((chrom == 0) || (chrom > chromCount)) skip = true;
              if (skip == true) printf("Skipping chromosome %s\n", chromNum);

              continue;
              }
            if (skip == false)
            {
              double t;
              sscanf(line,"%lf",&t);
              int ind = Mkb * (chrom - 1) + pos / resolution;
              if (ind >= Mkb * chromCount)
              {
                printf("Wrong index %d exceeds array size %d",ind,Mkb*chromCount);
              }
              else
              {                            
                  data[Mkb * (chrom - 1) + pos / resolution] += t;
              }
              pos+= step;
            }
        }
    return; 
    }
