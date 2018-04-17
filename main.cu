#include "Filereader.h"       // Includes Complex, vector and string
#include "math_constants.h"

#include <fstream>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda.h"

#define COMPLEX_SIZE sizeof(gpuFFT::Complex)
#define BLOCK_SIZE 4096

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
/*  float imaginary = exp(-2 * CUDART_PI_F * k * N);
  
  return gpuFFT::Complex(0, imaginary); */
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

  gpuFFT::Complex result;
  result.real = __cosf(theta);
  result.imag = __sinf(theta);
  
  return result;
}

__device__ gpuFFT::Complex idft_omega(const unsigned int k, const unsigned int n, const unsigned int N)
{
  float reciprocal = 1.0f / N;
  float theta      = (k * n * 2 * CUDART_PI_F * reciprocal);

  gpuFFT::Complex result;
  result.real = cosf(theta);
  result.imag = sinf(theta);
  
  return result;
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
__global__ void dft(gpuFFT::Complex* const input, gpuFFT::Complex* output, unsigned int size)
{
  __shared__ float input_data[BLOCK_SIZE];
  
  const unsigned int k = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  input_data[(2 * k)]     = input[k].real;
  input_data[(2 * k) + 1] = input[k].imag;
  
  __syncthreads();
  
  gpuFFT::Complex result;
  gpuFFT::Complex omega;
  for (unsigned int n = 0; n < size; ++n)
  {
	  omega       = dft_omega(k, n, size);
	  result.real = result.real + ((input[n].real * omega.real) - (input[n].imag * omega.imag));
	  result.imag = result.imag + ((input[n].real * omega.imag) + (input[n].imag * omega.real));
  }
  
  output[k] = result;
}


/// @brief Compute the Inverse Discrete Fourier Transform for a single value
///
///
/// The Inverse Discrete Fourier Transform has the following general equation form:
///
///       1  N - 1
///  f  = -*[SIGMA] X  [omega_n]^kn
///   n   N  k = 0   k
///
///
__global__ void idft(gpuFFT::Complex* input, gpuFFT::Complex* output, unsigned int size)
{
  /*const unsigned int n = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  gpuFFT::Complex result(0, 0);
  float reciprocal = 1 / size;
  for (unsigned int k = 0; k < size; ++k)
  {
    result += (input[k] * idft_omega(k, n, size));
  }
  
  output[n] = result * reciprocal; */
}

__device__ unsigned int reverse(unsigned int k, unsigned int n)
{
  /*unsigned int reversed = 0;
  unsigned int original = k;
  unsigned int maxIters = (unsigned int) roundf(log2f(n) + 0.5);
  for (unsigned int i = 0; i < maxIters; ++i)
  {
    reversed = (reversed << 1) + ((original >> 1) & 1);
  }
  */
  return 0;
}

__device__ void bitReverseCopy(gpuFFT::Complex* a, gpuFFT::Complex* A, unsigned int n)
{
  /*for (unsigned int k = 0; k < n; ++k)
  {
    A[reverse(k, n)] = a[k];
  }*/
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

// ====================================================
//                   EXECUTE DFT
// ====================================================

void execute_dft(std::vector<gpuFFT::Complex>& data)
{
  unsigned int dataSize   = (unsigned int)data.size();
  unsigned int deviceSize = dataSize * sizeof(gpuFFT::Complex); 
  
  // Allocate the host data (output)
  gpuFFT::Complex* hostData = (gpuFFT::Complex*)malloc(deviceSize);
  for (unsigned int i = 0; i < dataSize; ++i)
  {
    hostData[i] = data[i];
  }
  
  // Allocate the device data
  gpuFFT::Complex* deviceData;
  gpuFFT::Complex* deviceOutput;
  cudaMalloc((void**)&deviceData,   deviceSize);
  cudaMalloc((void**)&deviceOutput, deviceSize);
  
  // Copy the data from the host to the device
  cudaMemcpy(deviceData, hostData, deviceSize, cudaMemcpyHostToDevice);
  
  // Invoke the DFT
  // dft<<<1, 4, deviceSize>>>(deviceData, deviceOutput, dataSize);
  dft <<<1, dataSize>>> (deviceData, deviceOutput, dataSize);
  
  // Copy the data from the device to the host
  cudaMemcpy(hostData, deviceOutput, deviceSize, cudaMemcpyDeviceToHost);
  
  // Print the results
  for (unsigned int i = 0; i < dataSize; ++i)
  {
	  std::cout << hostData[i].real;
	  if (hostData[i].imag >= 0.0f)
	  {
		  std::cout << "+";
	  }
	  std::cout << hostData[i].imag << std::endl;
  }

  // Perform cleanup
  cudaFree(&deviceData);
  cudaFree(&deviceOutput);
  free(hostData);
}

// ====================================================
//                       MAIN
// ====================================================

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
  
  std::vector<gpuFFT::Complex> inputData;
  
  // Read the data in from the file
  inputFileReader.readFile(inputData);
  
  for (unsigned int i = 0; i < inputData.size(); ++i)
  {
	  std::cout << inputData[i].real;
	  if (inputData[i].imag >= 0.0f)
	  {
		  std::cout << "+";
	  }
	  std::cout << inputData[i].imag << std::endl;
  }
  
  std::cout << "Performing DFT" << std::endl;
  
  execute_dft(inputData);
  
  return EXIT_SUCCESS;
}