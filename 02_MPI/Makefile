# TODO[Davor]: confirm SCITAS module names. Suggested:
#   module load gcc hdf5
# HDF5_ROOT must point at an HDF5 install that provides include/ and lib/.

CXX      := mpicxx
LD       := $(CXX)

HDF5_INC := $(if $(strip $(HDF5_ROOT)),-I$(HDF5_ROOT)/include)
HDF5_LIB := $(if $(strip $(HDF5_ROOT)),-L$(HDF5_ROOT)/lib)
CXXFLAGS += -O3 -g -Wall -Wextra -Werror -pedantic -std=c++11 $(HDF5_INC)
LDFLAGS  += $(CXXFLAGS) $(HDF5_LIB) -lhdf5 -lm

OBJS = main_MPI.o lbm_MPI.o output_MPI.o

all: lbm_MPI

lbm_MPI: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f lbm_MPI *.o *~

distclean: clean
	rm -rf out/ probe.csv *.gif

.PHONY: all clean distclean
