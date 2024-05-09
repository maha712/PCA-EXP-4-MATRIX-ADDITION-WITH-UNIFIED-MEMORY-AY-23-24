# PCA-EXP-4-MATRIX-ADDITION-WITH-UNIFIED-MEMORY AY 23-24

ENTER YOUR NAME:Mahalakshmi k

ENTER YOUR REGISTER NO:212222240057

EX. NO: 4

  Refer to the program sumMatrixGPUManaged.cu. Would removing the memsets below affect performance? If you can, check performance with nvprof or nvvp.</h3>

## AIM:

To perform Matrix addition with unified memory and check its performance with nvprof.

## EQUIPMENTS REQUIRED:

Hardware – PCs with NVIDIA GPU & CUDA NVCC

Google Colab with NVCC Compiler

## PROCEDURE:

1.	Setup Device and Properties
Initialize the CUDA device and get device properties.

3.	Set Matrix Size: Define the size of the matrix based on the command-line argument or default value.
Allocate Host Memory

5.	Allocate memory on the host for matrices A, B, hostRef, and gpuRef using cudaMallocManaged.
  
7.	Initialize Data on Host

8.	Generate random floating-point data for matrices A and B using the initialData function.

9.	Measure the time taken for initialization.

10.	Compute Matrix Sum on Host: Compute the matrix sum on the host using sumMatrixOnHost.

11.	Measure the time taken for matrix addition on the host.

12.	Invoke Kernel

13.	Define grid and block dimensions for the CUDA kernel launch.

14.	Warm-up the kernel with a dummy launch for unified memory page migration.

15.	Measure GPU Execution Time

16.	Launch the CUDA kernel to compute the matrix sum on the GPU.

17.	Measure the execution time on the GPU using cudaDeviceSynchronize and timing functions.

18.	Check for Kernel Errors

19.	Check for any errors that occurred during the kernel launch.

20.	Verify Results

21.	Compare the results obtained from the GPU computation with the results from the host to ensure correctness.

22.	Free Allocated Memory

23.	Free memory allocated on the device using cudaFree.

24.	Reset Device and Exit

25.	Reset the device using cudaDeviceReset and return from the main function.

## PROGRAM:

WITH MEMSET:

%%cuda

#include <stdio.h>

#include <cuda_runtime.h>

#include <cuda.h>

#include <sys/time.h>

#ifndef _COMMON_H

#define _COMMON_H

#define CHECK(call)                                                            \

{                                                                              \
  
    const cudaError_t error = call;                                            \
   
    if (error != cudaSuccess)                                                  \
   
    {                                                                          \
       
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
      
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
        
                cudaGetErrorString(error));                                    \
       
        exit(1);                                                               \
  
    }                                                                          \

}

#define CHECK_CUBLAS(call)                                                     \

{                                                                              \
  
    cublasStatus_t err;                                                        \
   
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
   
    {                                                                          \
       
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
         
                __LINE__);                                                     \
       
        exit(1);                                                               \
   
    }                                                                          \

}

#define CHECK_CURAND(call)                                                     \
{                                                                              \
    curandStatus_t err;                                                        \
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUFFT(call)                                                      \

{                                                                              \
   
    cufftResult err;                                                           \
    
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
    
    {                                                                          \
       
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
          
                __LINE__);                                                     \
       
        exit(1);                                                               \
  
    }                                                                          \

}

#define CHECK_CUSPARSE(call)                                                   \

{                                                                              \
   
    cusparseStatus_t err;                                                      \
   
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
   
    {                                                                          \
       
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
       
        cudaError_t cuda_err = cudaGetLastError();                             \
        
        if (cuda_err != cudaSuccess)                                           \
       
        {                                                                      \
           
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                  
                    cudaGetErrorString(cuda_err));                             \
       
        }                                                                      \
        
        exit(1);                                                               \
  
    }                                                                          \
}


inline double seconds()

{
   
    struct timeval tp;
    
    struct timezone tzp;
    
    int i = gettimeofday(&tp, &tzp);
   
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


#endif // _COMMON_H

void initialData(float *ip, const int size)

{
   
    int i;

    for (i = 0; i < size; i++)
   
    {
       
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
   
    }
        
        return;
}


void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)

{

    float *ia = A;
   
    float *ib = B;
    
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
   
    {
       
        for (int ix = 0; ix < nx; ix++)
       
        {
           
            ic[ix] = ia[ix] + ib[ix];
       
        }

        ia += nx;
       
        ib += nx;
       
        ic += nx;
    }
     
     return;
}

void checkResult(float *hostRef, float *gpuRef, const int N)

{

    double epsilon = 1.0E-8;
   
    bool match = 1;

    for (int i = 0; i < N; i++)
    
    {
        
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
       
        {
            
            match = 0;
            
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
           
            break;
       
        }
    
    }

    if (!match)
    {
        printf("Arrays do not match.\n\n");
    }
}

// grid 2D block 2D

__global__ void sumMatrixGPU(float *MatA, float *MatB, float *MatC, int nx,
                            
                             int ny)

{

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
       
        MatC[idx] = MatA[idx] + MatB[idx];

}

int main(int argc, char **argv)

{
   
    printf("%s Starting ", argv[0]);

   // set up device
   
    int dev = 0;
   
    cudaDeviceProp deviceProp;
   
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
   
    printf("using Device %d: %s\n", dev, deviceProp.name);
    
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
   
    int nx, ny;
   
    int ishift = 12;

    if  (argc > 1) ishift = atoi(argv[1]);

    nx = ny = 1 << ishift;

    int nxy = nx * ny;
    
    int nBytes = nxy * sizeof(float);
   
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    
    float *A, *B, *hostRef, *gpuRef;
    
    CHECK(cudaMallocManaged((void **)&A, nBytes));
    
    CHECK(cudaMallocManaged((void **)&B, nBytes));
    
    CHECK(cudaMallocManaged((void **)&gpuRef,  nBytes);  );
    
    CHECK(cudaMallocManaged((void **)&hostRef, nBytes););

    // initialize data at host side
    
    double iStart = seconds();
    
    initialData(A, nxy);
    
    initialData(B, nxy);
   
    double iElaps = seconds() - iStart;
    
    printf("initialization: \t %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
   
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    
    iStart = seconds();
    
    sumMatrixOnHost(A, B, hostRef, nx, ny);
    
    iElaps = seconds() - iStart;
   
    printf("sumMatrix on host:\t %f sec\n", iElaps);

    // invoke kernel at host side
    
    int dimx = 32;
    
    int dimy = 32;
   
    dim3 block(dimx, dimy);
    
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // warm-up kernel, with unified memory all pages will migrate from host to
    
    // device
   
    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, 1, 1);

    // after warm-up, time with unified memory
    
    iStart = seconds();

    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, nx, ny);

    CHECK(cudaDeviceSynchronize());
    
    iElaps = seconds() - iStart;
    
    printf("sumMatrix on gpu :\t %f sec <<<(%d,%d), (%d,%d)>>> \n", iElaps,
           
            grid.x, grid.y, block.x, block.y);

    // check kernel error
   
    CHECK(cudaGetLastError());

    // check device results
   
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    
    CHECK(cudaFree(A));
    
    CHECK(cudaFree(B));
    
    CHECK(cudaFree(hostRef));
    
    CHECK(cudaFree(gpuRef));

    // reset device
    
    CHECK(cudaDeviceReset());

    return (0);
}

WITHOUT MEMSET:

%%cuda

#include <stdio.h>

#include <cuda_runtime.h>

#include <cuda.h>

#include <sys/time.h>

#ifndef _COMMON_H

#define _COMMON_H

#define CHECK(call)                                                            \

{                                                                              \
    const cudaError_t error = call;                                            \
    
    if (error != cudaSuccess)                                                  \
   
    {                                                                          \
        
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                
                cudaGetErrorString(error));                                    \
      
        exit(1);                                                               \
   
    }                                                                          \

}

#define CHECK_CUBLAS(call)                                                     \

{                                                                              \
    
    cublasStatus_t err;                                                        \
   
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
   
    {                                                                          \
       
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
               
                __LINE__);                                                     \
       
        exit(1);                                                               \
   
    }                                                                          \

}

#define CHECK_CURAND(call)                                                     \

{                                                                              \
    
    curandStatus_t err;                                                        \
    
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
   
    {                                                                          \
        
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
               
                __LINE__);                                                     \
       
        exit(1);                                                               \
   
    }                                                                          \

}

#define CHECK_CUFFT(call)                                                      \

{                                                                              \
   
    cufftResult err;                                                           \
    
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
   
    {                                                                          \
        
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
               
                __LINE__);                                                     \
       
        exit(1);                                                               \
  
    }                                                                          \

}

#define CHECK_CUSPARSE(call)                                                   \

{                                                                              \
   
    cusparseStatus_t err;                                                      \
    
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    
    {                                                                          \
        
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        
        cudaError_t cuda_err = cudaGetLastError();                             \
       
        if (cuda_err != cudaSuccess)                                           \
        
        {                                                                      \
           
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                
                    cudaGetErrorString(cuda_err));                             \
       
        }                                                                      \
       
        exit(1);                                                               \
   
    }                                                                          \

}

inline double seconds()

{
   
    struct timeval tp;
   
    struct timezone tzp;
    
    int i = gettimeofday(&tp, &tzp);
    
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);

}

#endif // _COMMON_H

void initialData(float *ip, const int size)

{
    int i;

    for (i = 0; i < size; i++)
   
    {
        
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    
    }

    return;

}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)

{
    float *ia = A;
    
    float *ib = B;
    
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    
    {
        
        for (int ix = 0; ix < nx; ix++)
        
        {
            
            ic[ix] = ia[ix] + ib[ix];
       
        }

        ia += nx;
        
        ib += nx;
       
        ic += nx;
   
    }

    return;

}

void checkResult(float *hostRef, float *gpuRef, const int N)

{
    
    double epsilon = 1.0E-8;
    
    bool match = 1;

    for (int i = 0; i < N; i++)
    
    {
        
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
       
        {
           
            match = 0;
           
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            
            break;
       
        }
    
    }

    if (!match)
   
    {
        
        printf("Arrays do not match.\n\n");
   
    }

}

// grid 2D block 2D

__global__ void sumMatrixGPU(float *MatA, float *MatB, float *MatC, int nx,
                           
                             int ny)

{

   unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  
   unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  
   unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
       
        MatC[idx] = MatA[idx] + MatB[idx];

}

int main(int argc, char **argv)

{
    printf("%s Starting ", argv[0]);

    // set up device
    
    int dev = 0;
    
    cudaDeviceProp deviceProp;
   
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    
    printf("using Device %d: %s\n", dev, deviceProp.name);
    
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    
    int nx, ny;
    
    int ishift = 12;

    if  (argc > 1) ishift = atoi(argv[1]);

    nx = ny = 1 << ishift;

    int nxy = nx * ny;
    
    int nBytes = nxy * sizeof(float);
    
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    
    float *A, *B, *hostRef, *gpuRef;
    
    CHECK(cudaMallocManaged((void **)&A, nBytes));
    
    CHECK(cudaMallocManaged((void **)&B, nBytes));
    
    CHECK(cudaMallocManaged((void **)&gpuRef,  nBytes);  );
    
    CHECK(cudaMallocManaged((void **)&hostRef, nBytes););

    // initialize data at host side
    
    double iStart = seconds();
    
    initialData(A, nxy);
    
    initialData(B, nxy);
    
    double iElaps = seconds() - iStart;
    
    printf("initialization: \t %f sec\n", iElaps);

    //memset(hostRef, 0, nBytes);
    
    //memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
   
    iStart = seconds();
    
    sumMatrixOnHost(A, B, hostRef, nx, ny);
    
    iElaps = seconds() - iStart;
    
    printf("sumMatrix on host:\t %f sec\n", iElaps);

    // invoke kernel at host side
    
    int dimx = 32;
    
    int dimy = 32;
    
    dim3 block(dimx, dimy);
    
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // warm-up kernel, with unified memory all pages will migrate from host to
    
    // device
    
    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, 1, 1);

    // after warm-up, time with unified memory
    
    iStart = seconds();

    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, nx, ny);

    CHECK(cudaDeviceSynchronize());
   
    iElaps = seconds() - iStart;
    
    printf("sumMatrix on gpu :\t %f sec <<<(%d,%d), (%d,%d)>>> \n", iElaps,
            
            grid.x, grid.y, block.x, block.y);

    // check kernel error
    
    CHECK(cudaGetLastError());

    // check device results
   
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    
    CHECK(cudaFree(A));
    
    CHECK(cudaFree(B));
    
    CHECK(cudaFree(hostRef));
    
    CHECK(cudaFree(gpuRef));

    // reset device
    
    CHECK(cudaDeviceReset());

    return (0);

}


## OUTPUT:

WITH MEMSET
https://private-user-images.githubusercontent.com/119091638/322711427-b860b1ee-7d2a-4086-8abc-1228a29d587b.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTUyMzI4ODcsIm5iZiI6MTcxNTIzMjU4NywicGF0aCI6Ii8xMTkwOTE2MzgvMzIyNzExNDI3LWI4NjBiMWVlLTdkMmEtNDA4Ni04YWJjLTEyMjhhMjlkNTg3Yi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNTA5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDUwOVQwNTI5NDdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yMjNlNmI4ODc5NzcxYmE1NjlhYTNjZWQ4YWZhM2RmOGY4OGNhNTM2ODM1ZTM4NTM5MjE5OTEyMDk2MzcyMjI4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.IEgbqWfuqIuNEXycJwvyTDGkY5cm2wcaT_TWuu0BJ08

WITHOUT MEMSET

https://private-user-images.githubusercontent.com/119091638/322711481-495b24b3-67c4-488b-b4b8-7371a60e5eaa.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTUyMzI4ODcsIm5iZiI6MTcxNTIzMjU4NywicGF0aCI6Ii8xMTkwOTE2MzgvMzIyNzExNDgxLTQ5NWIyNGIzLTY3YzQtNDg4Yi1iNGI4LTczNzFhNjBlNWVhYS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNTA5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDUwOVQwNTI5NDdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05NzRhZWI5YTQ0NGFlMTc2MTFlMzIzNzdlNTllYzAxNWJhMWQ3Y2M1NWNlYTI5ZTY1OTRhOThiZmNhZGZiZWVmJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.7l_fKPcj7jBMKs4GXUQd0aQ1anglPTsdN6OiC4ytrzs
## RESULT:
Thus the program has been executed by using unified memory. It is observed that removing memset function has given less time by a difference of 0.006344 sec.
