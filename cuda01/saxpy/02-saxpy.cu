#include <stdio.h>

#define N 2048 * 2048 // Number of elements in each vector

/*
 * Optimize this already-accelerated codebase. Work iteratively,
 * and use nsys to support your work.
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 200,000 ns.
 *
 * Some bugs have been placed in this codebase for your edification.
 */
__global__
void initWith(int num, int *a)
{

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    a[i] = num;
  }
}

__global__ void saxpy(int * a, int * b, int * c)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < N; i += stride)
        c[i] = 2 * a[i] + b[i];
}

int main()
{
    int deviceId;
    int numberOfSMs;
    cudaError_t addArraysErr;
    cudaError_t asyncErr;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    int *a, *b, *c;

    int size = N * sizeof (int); // The total number of bytes per vector

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);
    cudaMemPrefetchAsync(a, size, deviceId);
    cudaMemPrefetchAsync(b, size, deviceId);
    cudaMemPrefetchAsync(c, size, deviceId);
    int threadsPerBlock = 128;
    int numberOfBlocks = 32 * numberOfSMs;
    initWith<<<numberOfBlocks, threadsPerBlock>>>(2, a);
    initWith<<<numberOfBlocks, threadsPerBlock>>>(1, b);
    initWith<<<numberOfBlocks, threadsPerBlock>>>(0, c);

    saxpy <<< numberOfBlocks, threadsPerBlock >>> ( a, b, c );
    addArraysErr = cudaGetLastError();
    if(addArraysErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addArraysErr));

    asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));
  
    cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);
    // Print out the first and last 5 values of c for a quality check
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");
   

    cudaFree( a ); cudaFree( b ); cudaFree( c );
}
