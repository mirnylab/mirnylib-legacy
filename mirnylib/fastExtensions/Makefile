all:
	cython --cplus fastExtensionspy.pyx  
	g++  -g -march=native -Ofast -fpic -c   fastExtensions.cpp -o fastExtensions.o  -march=native  -fopenmp
	g++  -g -march=native -Ofast -fpic -c   fastExtensionspy.cpp -o fastExtensionspy.o `python-config --includes`  -fopenmp
	g++  -g -march=native -Ofast -shared  -o fastExtensionspy.so  fastExtensionspy.o fastExtensions.o `python-config --libs`  -lgomp

debug:
	cython --cplus fastExtensionspy.pyx  
	g++  -g -O0  --debug -fpic -c   fastExtensions.cpp -o fastExtensions.o  -march=native 
	g++  -g -O0 --debug -fpic -c   fastExtensionspy.cpp -o fastExtensionspy.o `python-config --includes` 
	g++  -g -O0 --debug -shared  -o fastExtensionspy.so  fastExtensionspy.o fastExtensions.o `python-config --libs` 

clean:
	rm -f fastExtensionspy.cpp *.o *.so

