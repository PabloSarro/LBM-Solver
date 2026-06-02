CXX 	 := nvcc

HDF5_INC := $(if $(strip $(HDF5_ROOT)),-I$(HDF5_ROOT)/include)
HDF5_LIB := $(if $(strip $(HDF5_ROOT)),-L$(HDF5_ROOT)/lib)
CXXFLAGS := -O3 -arch=sm_70 -std=c++14 $(HDF5_INC)
LDFLAGS  := $(CXXFLAGS) $(HDF5_LIB) -lhdf5 -lm

OBJS = main_CUDA.o lbm_CUDA.o output_CUDA.o

all: lbm_CUDA

lbm_CUDA: $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.cu
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f lbm_CUDA *.o

distclean: clean
	rm -rf lbm_CUDA *.o out/ probe.csv *.gif lbm_MPI_29*

.PHONY: all clean distclean
