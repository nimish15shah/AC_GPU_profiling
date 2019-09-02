
#include <stdio.h>
#include <stdlib.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cooperative_groups.h>

//#include <helper_cuda.h>
#define N_INPUTS 32
#define N_ARITH 74

__global__ void
ac(float *A, const int *B, const int *C, const int *op_sel, int n_inputs, const int n_arith, int thresh, int iter) {
  int i= blockDim.x * blockIdx.x + threadIdx.x;
  
  int idx_off= i*n_inputs;
  
  float val_31, val_32, val_33, val_34, val_35, val_36, val_37, val_38, val_39, val_40, val_41, val_42, val_43, val_44, val_45, val_46, val_47, val_48, val_49, val_50, val_51, val_52, val_53, val_54, val_55, val_56, val_57, val_58, val_59, val_60, val_61, val_62, val_63, val_64, val_65, val_66, val_67, val_68, val_69, val_70, val_71, val_72, val_73, val_74, val_75, val_76, val_77, val_78, val_79, val_80, val_81, val_82, val_83, val_84, val_85, val_86, val_87, val_88, val_89, val_90, val_91, val_92, val_93, val_94, val_95, val_96, val_97, val_98, val_99, val_100, val_101, val_102, val_103, val_104;
  
  float *val= &A[idx_off];

  for (int k=0; k<iter; k++) {
     val_31 =  A[idx_off + 5] *  val[29];
     val_32 =  val[6] *  val[30];
     val_33 =  val[3] *  val[29];
     val_34 =  val[4] *  val[30];
     val_35 =  val_31 +  val_32;
     val_36 =  val_33 +  val_34;
     val_37 =  val[0] *  val[15];
     val_38 =  val[1] *  val[16];
     val_39 =  val[3] *  val_37;
     val_40 =  val[4] *  val_37;
     val_41 =  val[0] *  val_38;
     val_42 =  val[1] *  val_38;
     val_43 =  val_39 +  val_41;
     val_44 =  val_40 +  val_42;
     val_45 =  val[12] *  val[19];
     val_46 =  val[13] *  val[19];
     val_47 =  val[11] *  val[20];
     val_48 =  val[14] *  val[20];
     val_49 =  val[10] *  val[19];
     val_50 =  val[11] *  val[19];
     val_51 =  val[9] *  val[20];
     val_52 =  val[12] *  val[20];
     val_53 =  val_45 +  val_47;
     val_54 =  val_46 +  val_48;
     val_55 =  val_49 +  val_51;
     val_56 =  val_50 +  val_52;
     val_57 =  val[27] *  val_43;
     val_58 =  val[28] *  val_44;
     val_59 =  val_57 +  val_58;
     val_60 =  val[2] *  val[25];
     val_61 =  val[2] *  val[26];
     val_62 =  val[7] *  val_60;
     val_63 =  val[9] *  val_61;
     val_64 =  val[8] *  val_60;
     val_65 =  val[10] *  val_61;
     val_66 =  val[11] *  val_62;
     val_67 =  val[0] *  val_63;
     val_68 =  val[12] *  val_62;
     val_69 =  val[1] *  val_63;
     val_70 =  val[11] *  val_64;
     val_71 =  val[0] *  val_65;
     val_72 =  val[12] *  val_64;
     val_73 =  val[1] *  val_65;
     val_74 =  val_66 +  val_67;
     val_75 =  val_68 +  val_69;
     val_76 =  val_70 +  val_71;
     val_77 =  val_72 +  val_73;
     val_78 =  val[21] *  val_35;
     val_79 =  val[22] *  val_36;
     val_80 =  val_53 *  val_78;
     val_81 =  val_54 *  val_79;
     val_82 =  val_55 *  val_78;
     val_83 =  val_56 *  val_79;
     val_84 =  val_59 *  val_80;
     val_85 =  val_57 *  val_80;
     val_86 =  val_58 *  val_81;
     val_87 =  val_59 *  val_82;
     val_88 =  val_57 *  val_82;
     val_89 =  val_58 *  val_83;
     val_90 =  val_85 +  val_86;
     val_91 =  val_88 +  val_89;
     val_92 =  val[17] *  val_74;
     val_93 =  val[17] *  val_75;
     val_94 =  val[18] *  val_76;
     val_95 =  val[18] *  val_77;
     val_96 =  val_84 *  val_92;
     val_97 =  val_90 *  val_93;
     val_98 =  val_87 *  val_94;
     val_99 =  val_91 *  val_95;
     val_100 =  val_96 +  val_98;
     val_101 =  val_97 +  val_99;
     val_102 =  val[23] *  val_100;
     val_103 =  val[24] *  val_101;
     val_104 =  val_102 +  val_103;
     A[i*n_inputs+5] += val_104;
  }
  A[i*n_inputs]= val_104;
}

int 
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    
    const int n_inputs= N_INPUTS;
    const int n_arith= N_ARITH;
    const int batch_size= 128;
    const int iter=  1;
    const int thresh= n_arith/3;

    size_t size= batch_size * (n_inputs) * sizeof(float);
    size_t size_idx= n_arith * sizeof(int);

    float *h_A= (float *)malloc(size);
    int *h_B= (int *)malloc(size_idx);
    int *h_C= (int *)malloc(size_idx);
    int *h_op_sel= (int *) malloc(size_idx);
    
    // Initialize the host input vectors
    for (int i = 0; i < n_arith; ++i)
    {
        if (i < thresh) {
          h_B[i] = rand() % (n_inputs); 
          h_C[i] = rand() % (n_inputs);  
        }
        else{
          h_B[i] = rand() % (i); 
          h_C[i] = rand() % (i);  
        }
        h_op_sel[i]= rand() % 2;
    }
    
    for (int i= 0; i < n_inputs; ++i) {
      for (int b =0; b< batch_size; ++b) {
        //h_A[b* n_inputs + i]= float(rand());
        h_A[b* n_inputs + i]= 0.5;
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
    int threadsPerBlock = 32;
    int blocksPerGrid= (batch_size + threadsPerBlock -1)/ threadsPerBlock;
    struct timeval t1, t2;

    // Perform Warmup
    ac<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_op_sel, n_inputs, n_arith, thresh, iter);
    // FInish execution of kernel
    cudaDeviceSynchronize();
    
    gettimeofday(&t1, 0);

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    ac<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_op_sel, n_inputs, n_arith, thresh, iter);

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
    for (int i=0; i<32; i++) {
      printf("%d : %f,", i, h_A[i*n_inputs]);
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
