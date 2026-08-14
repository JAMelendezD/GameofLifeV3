# Build the CUDA kernel + C API into a shared library Python can load.
# RTX 3060 Laptop  -> sm_86

NVCC ?= nvcc
ARCH ?= -arch=sm_86
NVCC_FLAGS = -O3 -std=c++17 --shared -Xcompiler -fPIC

lib/libgameoflife.so: src/gameoflife.cu include/gameoflife.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -Iinclude src/gameoflife.cu -o $@

CXX ?= g++
CXX_FLAGS = -O3 -std=c++17 -fPIC -fopenmp

lib/libgameoflife_cpu.so: src/gameoflife_cpu.cpp include/gameoflife.h
	$(CXX) $(CXX_FLAGS) -shared -Iinclude src/gameoflife_cpu.cpp -o $@

all: lib/libgameoflife.so lib/libgameoflife_cpu.so

clean:
	rm -f lib/libgameoflife.so lib/libgameoflife_cpu.so

.PHONY: all clean
