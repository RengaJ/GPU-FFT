#include "Filereader.h"       // Includes Complex, vector and string

#include "CPUDFT.h"
#include "math_constants.h"

#include <fstream>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda.h"

#define FLOAT_SIZE sizeof(float)
#define BLOCK_DIM 1024

/// ---------------   DEVICE FUNCTIONS   --------------- ///
__device__ float2 dft_omega(const unsigned int k, const unsigned int n, const unsigned int N)
{
	float reciprocal = 1.0f / N;
	int exponent     = -k * n;
	float theta      = 2.0f * exponent * CUDART_PI_F * reciprocal;

	float2 omega;
	omega.x = cosf(theta);
	omega.y = sinf(theta);

	return omega;
}

__device__ float2 idft_omega(const unsigned int k, const unsigned int n, const unsigned int N)
{
  float reciprocal = 1.0f / N;
  int exponent     = k * n;
  float theta      = 2.0f * exponent * CUDART_PI_F * reciprocal;
  
  float2 omega;
  omega.x = cosf(theta);
  omega.y = sinf(theta);
  
  return omega;
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
__global__ void dft(float* realData,  float* imagData,
                    float* realInput, float* imagInput,
                    unsigned int N)
{
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x; // Row
  const unsigned int n = threadIdx.x;
  
  float2 omega = dft_omega(blockIdx.x, n, N);

  realData[k] = (realInput[n] * omega.x) - (imagInput[n] * omega.y);
  imagData[k] = (realInput[n] * omega.y) + (imagInput[n] * omega.x);
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
__global__ void idft(float* realData,  float* imagData,
	                 float* realInput, float* imagInput,
	                 unsigned int N)
{
	const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int k = threadIdx.x;

	float2 omega = idft_omega(k, blockIdx.x, N);

	realData[n] = (realInput[k] * omega.x) - (imagInput[k] * omega.y);
	imagData[n] = (realInput[k] * omega.y) + (imagInput[k] * omega.x);
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

void execute_dft(std::vector<float>& realParts, std::vector<float>& imagParts)
{
  unsigned int dataSize   = (unsigned int)realParts.size();
  unsigned int deviceSize = dataSize * sizeof(float); 
  
  // Allocate the host data
  float* realResiduals;
  float* imagResiduals;

  float* hostRealData = (float*)malloc(deviceSize * dataSize);
  float* hostImagData = (float*)malloc(deviceSize * dataSize);

  for (unsigned int i = 0; i < dataSize * dataSize; ++i)
  {
	  hostRealData[i] = 0.0f;
	  hostImagData[i] = 0.0f;
  }

  float* deviceRealData;
  float* deviceImagData;
  
  // Allocate the device data
  cudaMalloc(&realResiduals, deviceSize * dataSize);
  cudaMalloc(&imagResiduals, deviceSize * dataSize);
  
  cudaMalloc(&deviceRealData, deviceSize);
  cudaMalloc(&deviceImagData, deviceSize);

  cudaMemset(realResiduals, 0, deviceSize * dataSize);
  cudaMemset(imagResiduals, 0, deviceSize * dataSize);

  // Copy the data from the host to the device
  cudaMemcpy(deviceRealData, &realParts[0], deviceSize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceImagData, &imagParts[0], deviceSize, cudaMemcpyHostToDevice);
  
  // Invoke the DFT
  dft <<< dataSize, dataSize >>> (realResiduals, imagResiduals, deviceRealData, deviceImagData, dataSize);
  
  cudaDeviceSynchronize();
  
  // Copy the data from the device to the host
  cudaMemcpy(hostRealData, realResiduals, deviceSize * dataSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostImagData, imagResiduals, deviceSize * dataSize, cudaMemcpyDeviceToHost);
  
  std::vector<float> realResult(dataSize);
  std::vector<float> imagResult(dataSize);
  
  for (unsigned int i = 0; i < dataSize; ++i)
  {
    realResult[i] = 0.0f;
    imagResult[i] = 0.0f;
    for (unsigned int j = 0; j < dataSize; ++j)
    {
		int index = (i * dataSize) + j;
		float realResidual = hostRealData[index];
		float imagResidual = hostImagData[index];

		realResult[i] += realResidual;
		imagResult[i] += imagResidual;
    }
  }
  
  for (unsigned int i = 0; i < realResult.size(); ++i)
  {
    std::cout << realResult[i];
    
	  if (imagResult[i] >= 0.0f)
	  {
		  std::cout << "+";
	  }
	  std::cout << imagResult[i] << "i" << std::endl;
  }

  // Copy the results back into the device real and imaginary parts
  cudaMemcpy(deviceRealData, &realResult[0], deviceSize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceImagData, &imagResult[0], deviceSize, cudaMemcpyHostToDevice);

  // Invoke the IDFT
  idft << < dataSize, dataSize >> > (realResiduals, imagResiduals, deviceRealData, deviceImagData, dataSize);

  // Copy the data from the device to the host
  cudaMemcpy(hostRealData, realResiduals, deviceSize * dataSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostImagData, imagResiduals, deviceSize * dataSize, cudaMemcpyDeviceToHost);

  realResult.clear();
  imagResult.clear();

  realResult.resize(dataSize, 0.0f);
  imagResult.resize(dataSize, 0.0f);

  for (unsigned int i = 0; i < dataSize; ++i)
  {
	  realResult[i] = 0.0f;
	  imagResult[i] = 0.0f;
	  for (unsigned int j = 0; j < dataSize; ++j)
	  {
		  int index = (i * dataSize) + j;
		  float realResidual = hostRealData[index];
		  float imagResidual = hostImagData[index];

		  realResult[i] += realResidual / dataSize;
		  imagResult[i] += imagResidual / dataSize;
	  }
  }

  for (unsigned int i = 0; i < realResult.size(); ++i)
  {
	  std::cout << realResult[i];

	  if (imagResult[i] >= 0.0f)
	  {
		  std::cout << "+";
	  }
	  std::cout << imagResult[i] << "i" << std::endl;
  }
  
  // Print the results
  // Perform cleanup
  cudaFree(realResiduals);
  cudaFree(imagResiduals);
  cudaFree(deviceRealData);
  cudaFree(deviceImagData);
 
  free(hostRealData);
  free(hostImagData);
}

// ====================================================
//                       MAIN
// ====================================================

int main(int argc, char* argv[])
{
  if (argc != 2)
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
  
  std::vector<float> realParts;
  std::vector<float> imagParts;
  
  // Read the data in from the file
  inputFileReader.readFile(realParts, imagParts);
  
  for (unsigned int i = 0; i < realParts.size(); ++i)
  {
    std::cout << realParts[i];
    
	  if (imagParts[i] >= 0.0f)
	  {
		  std::cout << "+";
	  }
	  std::cout << imagParts[i] << "i" << std::endl;
  }
  
  execute_dft(realParts, imagParts);

  gpuFFT::CPUDFT cpuDFT(realParts, imagParts);
  std::vector<float> transformedReal;
  std::vector<float> transformedImag;

  cpuDFT.dft(transformedReal, transformedImag);

  for (unsigned int i = 0; i < transformedReal.size(); ++i)
  {
	  std::cout << "CPU DFT [" << i << "] = (" << transformedReal[i] << ", " << transformedImag[i] << ")" << std::endl;
  }

  gpuFFT::CPUDFT cpuIDFT(transformedReal, transformedImag);
  cpuIDFT.idft(transformedReal, transformedImag);

  for (unsigned int i = 0; i < transformedReal.size(); ++i)
  {
	  std::cout << "CPU IDFT [" << i << "] = (" << transformedReal[i] << ", " << transformedImag[i] << ")" << std::endl;
  }
  
  return EXIT_SUCCESS;
}