# CUDA-Runtime-API-calls-interception
Shared library for intercepting CUDA Runtime API calls. This was part of my Bachelor thesis: A Study on the Computational Exploitation of Remote Virtualized Graphics Cards (https://bit.ly/37tIG0D)


Prerequisites:
-GNU/Linux for compilation
-Set CUDA_PATH variable in the Makefile to the correct directory 
 where cuda is installed.
 
Tested on:
gcc-6,gcc-7
 
 
How to compile:
$ make

To remove:
$ make clean


How to run:
$ LD_PRELOAD=/full_path_to_thecuda_intercept_directory/cuda_intercept/lib_cuda_intercept.so ./full_path_to_the_directory_of_the_CUDA_Program/your_cuda_program.cu

e.g.
LD_PRELOAD=/home/Desktop/cuda_intercept/lib_cuda_intercept.so /home/NVIDIA_CUDA-9.0_Samples/6_Advanced/transpose/transpose
