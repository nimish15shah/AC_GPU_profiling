#include <stdio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cooperative_groups.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <assert.h>
#include <string.h>

#include "./files/bnetflix_psdd_gpu_cuda_4.cu"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 */
__global__ void
main_ac(float *A, int nIter) { 
  ac(A, nIter); 
}

int 
main(int argc, char **argv)
{
    // nIter 
    int nIter = getCmdLineArgumentInt(argc, (const char **)argv, "nIter");
    
    size_t size_a= sizeof(float)* SIZE_OF_IN;

    // Allocate the device input vector A
    float *d_A = NULL;
    gpuErrchk(cudaMalloc((void **)&d_A, size_a));


    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    gpuErrchk(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid= BLOCKS_PER_GRID;
    struct timeval t1, t2;
    gettimeofday(&t1, 0);

    main_ac<<<blocksPerGrid, threadsPerBlock>>>(d_A, nIter);

    // FInish execution of kernel
    cudaDeviceSynchronize();

    gettimeofday(&t2, 0);
    
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    
    printf("Time of kernel:  %3.4f ms \n", time);

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    gpuErrchk(cudaGetLastError());

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    gpuErrchk(cudaMemcpy(h_A, d_A, size_a, cudaMemcpyDeviceToHost));
    
    for (int i=0; i< 4; i++) {
      printf("%d , %f | ", i, h_A[i]);
    }

    gpuErrchk(cudaFree(d_A));

    printf("Done!\n");
    return 0;
}
