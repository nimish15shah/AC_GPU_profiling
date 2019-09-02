
#include <stdio.h>
#include <stdlib.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cooperative_groups.h>

//#include <helper_cuda.h>

__global__ void
ac(float *A, const int *B, const int *C, const int *op_sel, int n_inputs, int n_arith, int iter) {
  int i= blockDim.x * blockIdx.x + threadIdx.x;
  
  int a_off= i*(n_inputs + n_arith) + n_inputs;
  int idx_off= i*(n_inputs + n_arith);

  for (int k=0; k<iter; k++) {
    for (int j=0; j <n_arith; j++ ) {
      
      if (op_sel[j] == 0) 
        A[a_off + j] = A[idx_off + B[j]] + A[idx_off + C[j]];
      else
        A[a_off + j] = A[idx_off + B[j]] * A[idx_off + C[j]];
    }
  }
}

int 
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    
    int n_inputs= 32;
    int n_arith= 256;
    int batch_size= 2048;
    int iter= 4096;
    
    int n_tot= n_inputs + n_arith; 
    size_t size= batch_size * (n_tot) * sizeof(float);
    size_t size_idx= n_arith * sizeof(int);

    float *h_A= (float *)malloc(size);
    int *h_B= (int *)malloc(size_idx);
    int *h_C= (int *)malloc(size_idx);
    int *h_op_sel= (int *) malloc(size_idx);
    
    // Initialize the host input vectors
    for (int i = 0; i < n_arith; ++i)
    {
        h_B[i] = rand() % (n_inputs + i); 
        h_C[i] = rand() % (n_inputs + i);  
        h_op_sel[i]= rand() % 2;
    }
    
    for (int i= 0; i < n_inputs; ++i) {
      for (int b =0; b< batch_size; ++b) {
        h_A[b* n_tot + i]= float(rand());
      }
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size_idx);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size_idx);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *d_op_sel = NULL;
    err = cudaMalloc((void **)&d_op_sel, size_idx);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_B, h_B, size_idx, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_C, h_C, size_idx, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_op_sel, h_op_sel, size_idx, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 64;
    int blocksPerGrid= (batch_size + threadsPerBlock -1)/ threadsPerBlock;
    struct timeval t1, t2;

    // Perform Warmup
    ac<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_op_sel, n_inputs, n_arith, iter);
    // FInish execution of kernel
    cudaDeviceSynchronize();
    
    gettimeofday(&t1, 0);

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    ac<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_op_sel, n_inputs, n_arith, iter);

    // FInish execution of kernel
    cudaDeviceSynchronize();
    gettimeofday(&t2, 0);
    
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    
    printf("Time of kernel:  %3.4f ms \n", time);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Throughput: %.3f Gops/sec\n", (((1.0*batch_size*iter*n_arith))/time)/10E6);

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    //for (int i=0; i<numElements; i++) {
    for (int i=0; i<8; i++) {
      printf("%d , %f\n", i, h_A[i]);
    }

    err = cudaFree(d_A);
    err = cudaFree(d_B);
    err = cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done!\n");
    return 0;
}
