#include <stdio.h>
#include <iostream>

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int init = threadIdx.x + blockIdx.x * blockDim.x;
  int step = gridDim.x * blockDim.x;
  for(int i = init; i < N; i += step)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  const int N = 2<<25;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;
  cudaMallocManaged(&a ,size);
  cudaMallocManaged(&b ,size);
  cudaMallocManaged(&c ,size);
  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);
  // size_t threads_per_block = 256;
  // size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
  addVectorsInto<<<1, 10>>>(c, a, b, N);
  cudaDeviceSynchronize();
  checkElementsAre(7, c, N);
  return 0;

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
