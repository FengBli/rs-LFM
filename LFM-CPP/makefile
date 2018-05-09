LFM:  LFM-run.o LFM.o
	g++ -std=c++11  LFM-run.o  LFM.o -o LFM
LFM-run.o:  LFM-run.cpp  LFM.h
	g++ -std=c++11 -c LFM-run.cpp
LFM.o:  LFM.cpp
	g++ -std=c++11 -c LFM.cpp
clean:
	rm -f   *.o
