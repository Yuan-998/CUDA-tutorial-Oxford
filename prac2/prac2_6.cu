
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

#include <helper_cuda.h>

__constant__ float a, b, c;

__global__ void aver(float *z, float *res) {
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   float tmp = 0;

   for (int i = 0; i < 100; i++) {
      tmp += a * z[tid] * z[tid] + b * z[tid] + c;
      tid += blockIdx.x * blockDim.x;
   }
   res[threadIdx.x] = tmp/100;
}

int main() {
   int n_thread = 256;
   int N = n_thread * 100;
   float *d_z, *h_res, *d_res;
   float h_a, h_b, h_c;

   h_res = (float *)malloc(sizeof(float) * n_thread);

   checkCudaErrors(cudaMalloc((void **)&d_z, sizeof(float)*N));
   checkCudaErrors(cudaMalloc((void**)&d_res, sizeof(float)*n_thread));

   h_a = 1.0f;
   h_b = 2.0f;
   h_c = 3.0f;

   checkCudaErrors(cudaMemcpyToSymbol(a, &h_a, sizeof(h_a)));
   checkCudaErrors(cudaMemcpyToSymbol(b, &h_b, sizeof(h_b)));
   checkCudaErrors(cudaMemcpyToSymbol(c, &h_c, sizeof(h_c)));

   curandGenerator_t gen;
   checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
   checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );

   checkCudaErrors( curandGenerateNormal(gen, d_z, N, 0.0f, 1.0f) );

   cudaEvent_t start, end;
   float milli;

   cudaEventCreate(&start);
   cudaEventCreate(&end);

   cudaEventRecord(start);
   aver<<<1, n_thread>>>(d_z, d_res);
   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&milli, start, end);

   checkCudaErrors( cudaMemcpy(h_res, d_res, sizeof(float)*n_thread,
                   cudaMemcpyDeviceToHost) );

   float sum = 0.0f;
   for (int i = 0; i < n_thread; i++) {
      sum += h_res[i];
   }
   printf("average val == a + c: %f == %f + %f\ntime comsumed: %f\n", sum/n_thread, h_a, h_c, milli);

   checkCudaErrors( curandDestroyGenerator(gen) );

   free(h_res);
   checkCudaErrors(cudaFree(d_z));
   checkCudaErrors(cudaFree(d_res));

   cudaDeviceReset();

   return 0;
}