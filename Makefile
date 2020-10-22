# Edit the CUDA_PATH variable for your system
CUDA_PATH?=/usr/local/cuda-9.0


#Set compilation flags
CXX=g++
CFLAGS=-Wall -fPIC -shared -ldl


all: lib_cuda_intercept.so 

lib_cuda_intercept.so: cuda_intercept.cpp
	$(CXX) -I$(CUDA_PATH)/include $(CFLAGS) -o lib_cuda_intercept.so cuda_intercept.cpp

clean:
	-rm lib_cuda_intercept.so
