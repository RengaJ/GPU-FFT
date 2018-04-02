#include "Complex.h"
#include "math_constants.h"

__device__ gpuFFT::Complex getTwiddleFactor(unsigned int k, unsigned int N)
{
  float imaginary = exp(-2 * CUDART_PI_F * k * N);
  
  return gpuFFT::Complex(0, imaginary);
}

int main(int argc, char* argv[])
{
  return EXIT_SUCCESS;
}