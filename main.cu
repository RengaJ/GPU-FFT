#include "Complex.h"
#include "Filereader.h"
#include "math_constants.h"

#include <fstream>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/// ---------------   DEVICE FUNCTIONS   --------------- ///

/// @brief Retrieve the complex twiddle factor, given the current item number
///        and the total number of items in the Fourier Transform
///
/// @param [in] k    The current DFT index
/// @param [in] N    The number of DFT elements
///
/// @return A complex number that contains the twiddle factor (in the imaginary component)
__device__ gpuFFT::Complex getTwiddleFactor(unsigned int k, unsigned int N)
{
  float imaginary = exp(-2 * CUDART_PI_F * k * N);
  
  return gpuFFT::Complex(0, imaginary);
}

/// @brief Retrieve the complex number used in the Discrete Fourier Transform. This has the form
///    -(2pi / N)kn
///   e             , which can further simplified into the following:
///
///   cos((-2*pi*kn)/N) + isin((-2*pi*kn)/N)
///
///   This value is used as the multiplicative factor in the Discrete Fourier Transform equation
__device__ gpuFFT::Complex dft_omega(const unsigned int k, const unsigned int n, const unsigned int N)
{
  float reciprocal = 1.0f / N;
  float theta      = (-k * n * 2 * CUDART_PI_F * reciprocal);
  
  return gpuFFT::Complex(cosf(theta), sinf(theta));
}

__device__ gpuFFT::Complex idft_omega(const unsigned int k, const unsigned int n, const unsigned int N)
{
  float reciprocal = 1.0f / N;
  float theta      = (k * n * 2 * CUDART_PI_F * reciprocal);
  
  return gpuFFT::Complex(cosf(theta), sinf(theta));
}

/// @brief Compute the Discrete Fourier Transform for a single value
///
///
///  The Discrete Fourier Transform has the following general equation form:
///
///          N - 1
///   X   = [SIGMA] f  [omega_n]^-kn
///    k     n = 0   n
///
///
__global__ void dft(gpuFFT::Complex* input, gpuFFT::Complex* output, unsigned int size)
{
  unsigned int k = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  gpuFFT::Complex result(0, 0);
  for (unsigned int n = 0; n < size; ++n)
  {
    result += (input[k] * dft_omega(k, n, size));
  }
  
  output[k] = result;
}

__global__ void idft(gpuFFT::Complex* input, gpuFFT::Complex* output, unsigned int size)
{
  unsigned int n = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  gpuFFT::Complex result(0, 0);
  for (unsigned int k = 0; k < size; ++k)
  {
    result += (input[k] * idft_omega(k, n, size));
  }
  
  output[n] = result;
}

__device__ unsigned int reverse(unsigned int k, unsigned int n)
{
  unsigned int reversed = 0;
  unsigned int original = k;
  unsigned int maxIters = (unsigned int) roundf(log2f(n) + 0.5);
  for (unsigned int i = 0; i < maxIters; ++i)
  {
    reversed = (reversed << 1) + ((original >> 1) & 1);
  }
  
  return reversed;
}

__device__ void bitReverseCopy(gpuFFT::Complex* a, gpuFFT::Complex* A, unsigned int n)
{
  for (unsigned int k = 0; k < n; ++k)
  {
    A[reverse(k, n)] = a[k];
  }
}

__global__ void FFT(gpuFFT::Complex* input, gpuFFT::Complex* output, const unsigned int size)
{
  // bitReverseCopy(input, A, size);
  /*unsigned int maxIters = (unsigned int) roundf(log2f(n) + 0.5);
  for (unsigned int s = 1; s < maxIters; ++s)
  {
    unsigned int m = powf(2, s);
    gpuFFT::Complex twiddle = getTwiddleFactor(m, size);
    for (unsigned int k = 0; k < size; k += m)
    {
      
    }
  }*/
}

/// ---------------   GLOBAL FUNCTIONS   --------------- ///

/// ----------------   HOST FUNCTIONS   ---------------- ///

/// @brief Show how the program is used when it is being used improperly
__host__ void show_program_usage()
{
  std::cout << "Improper program use detected. Please see the following for instructions:"                        << std::endl;
  std::cout << "\tgpuFFT.exe <input_file> <output_file_location>"                                                 << std::endl;
  std::cout << ""                                                                                                 << std::endl;
  std::cout << "input_file           - The input file containg the data on which to perform the FFT (must exist)" << std::endl;
  std::cout << "                       * Must have the .dat extension * "                                         << std::endl;
  std::cout << "output_file_location - The location where the output.dat file will be created (must exist)"       << std::endl;
  std::cout << ""                                                                                                 << std::endl;
  std::cout << "Please verify your inputs and try again."                                                         << std::endl;
}

int main(int argc, char* argv[])
{
  if (argc != 3)
  {
    show_program_usage();
    
    return EXIT_FAILURE;
  }
  
  // The first argument should be the input file.
  // Check to see if it exists
  gpuFFT::Filereader inputFileReader(argv[1]);
  if (!inputFileReader.exists())
  {
    show_program_usage();
    
    return EXIT_FAILURE;
  }
  
  
  
  return EXIT_SUCCESS;
}