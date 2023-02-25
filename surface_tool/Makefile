

SUBDIRS=$(shell ls -l | grep ^d | awk '{print $$9}')
compiler=nvcc
COMMON_COMMAND=-arch=sm_60 -rdc=true -lcudadevrt -lcufft --use_fast_math -O4 -std=c++11 -lgomp
BIN_NAME=surface_tool

install: Run_Surface.o Surface.o common.o 
	$(compiler) -o $(BIN_NAME)  $^ $(COMMON_COMMAND)


all: Run_Surface.o Surface.o common.o
	

Run_Surface.o: Run_Surface.cu
	$(compiler) -o $@ -c $< $(COMMON_COMMAND)

Surface.o: Surface.cu Surface.cuh
	$(compiler) -o $@ -c $< $(COMMON_COMMAND)


common.o: common.cu common.cuh
	$(compiler) -o $@ -c $< $(COMMON_COMMAND)



clean:
	rm -f *.o
	rm -f $(foreach i, $(SUBDIRS), $(i)/*.o)
