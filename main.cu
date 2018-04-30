#include "Filereader.h"
#include "Filewriter.h"

#include "CommandLineParser.h"

#include "CPUDFT.h"
#include "math_constants.h"

#include <fstream>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda.h"

#define FLOAT_SIZE sizeof(float)
#define BLOCK_DIM 1024

#define DFT_MULTIPLIER 1
#define IDFT_MULTIPLIER -1

/// //////////////////////////////////////////////////// ///
/// ---------------   DEVICE FUNCTIONS   --------------- ///
/// //////////////////////////////////////////////////// ///

/// @brief Compute the omega value used in the Fourier Transforms
///
/// The standard form of the omega value is the following:
///                    (2*pi / N)
///    [omega]    =   e
///           n
///
/// Which can be re-written as the following complex number:
///
///    [omega]    = cos(2*pi / N) + isin(2*pi / N)
///           n
///
/// When raised to the kn (or -kn) power (as is found in the actual
/// algorithm), the value now becomes the following:
///
///    [omega]    = cos(2*k*n*pi / N) + isin(2*k*n*pi / N)
///           n
///
/// @param [in] k           The current frequency space value index
/// @param [in] n           The current time space value index
/// @param [in] N           The total number of elements in the data-set
/// @param [in] multiplier  A multiplier value used to indicate if DFT (-1) or IDFT (1) mode is active
///
/// @return A two-element floating-point value containing the real (x) and imaginary (y) parts of the omega value
__device__ float2 compute_omega(const unsigned int k, const unsigned n, const unsigned N, const int multiplier)
{
	float reciprocal = 1.0f / N;
	int exponent     = multiplier * k * n;
	float theta      = 2.0f * exponent * CUDART_PI_F * reciprocal;

	float2 omega;
	omega.x = cosf(theta);
	omega.y = sinf(theta);

	return omega;
}

/// //////////////////////////////////////////////////// ///
/// ---------------   GLOBAL FUNCTIONS   --------------- ///
/// //////////////////////////////////////////////////// ///

/// @brief Compute the Discrete Fourier Transform for a single value
///
///
///  The Discrete Fourier Transform has the following general equation form:
///
///          N - 1
///   X   = [SIGMA] f  [omega_n]^-kn
///    k     n = 0   n
///
/// @param [out] realData   The real result of the current (k,n) transform residual
/// @param [out] imagData   The imaginary result of the current (k,n) transform residual
/// @param [ in] realInput  The set of real input data
/// @param [ in] imagInput  The set of imaginary input data
/// @param [ in] N          The total number of elements in the input data
__global__ void dft(float* realData,  float* imagData,
                    float* realInput, float* imagInput,
                    unsigned int N)
{
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int n = threadIdx.x;
  
  float2 omega = compute_omega(blockIdx.x, n, N, DFT_MULTIPLIER);

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
/// @param [out] realData   The real result of the current (k,n) transform residual
/// @param [out] imagData   The imaginary result of the current (k,n) transform residual
/// @param [ in] realInput  The set of real input data
/// @param [ in] imagInput  The set of imaginary input data
/// @param [ in] N          The total number of elements in the input data
__global__ void idft(float* realData,  float* imagData,
	                 float* realInput, float* imagInput,
	                 unsigned int N)
{
	const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int k = threadIdx.x;

	float2 omega = compute_omega(k, blockIdx.x, N, IDFT_MULTIPLIER);

	realData[n] = (realInput[k] * omega.x) - (imagInput[k] * omega.y);
	imagData[n] = (realInput[k] * omega.y) + (imagInput[k] * omega.x);
}

/// //////////////////////////////////////////////////// ///
/// ----------------   HOST FUNCTIONS   ---------------- ///
/// //////////////////////////////////////////////////// ///

/// @brief Show how the program is used when it is being used improperly
__host__ void show_program_usage()
{
  std::cout << "Improper program use detected. Please see the following for instructions:"                        << std::endl;
  std::cout << "\tgpuFFT.exe <input_file> [MODE]"                                                                 << std::endl;
  std::cout << "------------------------------------------------------------------------------------------------" << std::endl;
  std::cout << "input_file           - The input file containg the data on which to perform the FFT (must exist)" << std::endl;
  std::cout << "                       * Must have the .dat extension * "                                         << std::endl;
  std::cout << "[MODE]               - The preferred mode of operation. Select one of the following:"             << std::endl;
  std::cout << "                       --gpu-dft    -- Execute the GPU Discrete Fourier Transform"                << std::endl;
  std::cout << "                                       Will output results in gpu_dft_output.dat here"            << std::endl;
  std::cout << "                       --gpu-idft   -- Executes the GPU Inverse Discrete Fourier Transform"       << std::endl;
  std::cout << "                                       Will output results in gpu_idft_output.dat here"           << std::endl;
  std::cout << "                       --cpu-dft    -- Executes the CPU Discrete Fourier Transform"               << std::endl;
  std::cout << "                                       Will output results in cpu_dft_output.dat here"            << std::endl;
  std::cout << "                       --cpu-idft   -- Executes the CPU Inverse Discrete Fourier Transform"       << std::endl;
  std::cout << "                                       Will output results in cpu_idft_output.dat here"           << std::endl;
  std::cout << "                       --compare    -- Executes the all Fourier Transform options and displays"   << std::endl;
  std::cout << "                                       execution times. Provides all output in previous options"  << std::endl;
  std::cout << "------------------------------------------------------------------------------------------------" << std::endl;
  std::cout << "Please verify your inputs and try again."                                                         << std::endl;
}

/// //////////////////////////////////////////////////// ///
/// --------------   UTILITY FUNCTIONS   --------------- ///
/// //////////////////////////////////////////////////// ///

/// @brief Allocate an array of float data on the host-side. This will
///        also ensure the data is initialized to zero prior to being
///        used.
///
///        *** NOTE *** This array needs to be freed with the freeHostMemory
///                     function. It will not be freed automatically.
///
/// @param [in] numElements    The number of elements for which to allocate.
///
/// @return The allocated array of float data.
float* allocateHostData(unsigned int numElements)
{
  float* input = (float*)calloc(numElements, FLOAT_SIZE);

  return input;
}

void allocateDeviceMemory(float** input, unsigned int size)
{
  cudaMalloc((void**)input, size);
  cudaMemset(*input, 0, size);
}

void freeDeviceMemory(float* input)
{
  cudaFree(input);
}



/// @brief Free an allocated float array.
///
/// @param [in] input   The float array to de-allocate (free)
void freeHostMemory(float* input)
{
  if (input != NULL)
  {
    free(input);
  }
}

// ====================================================
//                   EXECUTE DFT
// ====================================================
void execute_gpu_fourier_transform(
  std::vector<float>& realParts,
  std::vector<float>& imagParts,
  gpuFFT::ModeFlag modeFlag)
{
  unsigned int dataSize   = (unsigned int)realParts.size();
  unsigned int deviceSize = dataSize * FLOAT_SIZE;

  std::string filename =
    (modeFlag == gpuFFT::GPU_DFT_ONLY ? "gpu_dft_output.dat" : "gpu_idft_output.dat");
  
  // Allocate the host data
  float* hostRealData = allocateHostData(dataSize * dataSize);
  float* hostImagData = allocateHostData(dataSize * dataSize);

  // Allocate the device data
  float* deviceRealData;
  float* deviceImagData;
  float* realResiduals;
  float* imagResiduals;

  allocateDeviceMemory(&deviceRealData, deviceSize);
  allocateDeviceMemory(&deviceImagData, deviceSize);
  allocateDeviceMemory(&realResiduals,  deviceSize * dataSize);
  allocateDeviceMemory(&imagResiduals,  deviceSize * dataSize);

  // Create CUDA event data
  cudaEvent_t fourierTransformStartEvent;
  cudaEvent_t fourierTransformEndEvent;

  cudaEventCreate(&fourierTransformStartEvent);
  cudaEventCreate(&fourierTransformEndEvent);

  // Copy the data from the host to the device
  cudaMemcpy(deviceRealData, &realParts[0], deviceSize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceImagData, &imagParts[0], deviceSize, cudaMemcpyHostToDevice);
  
  // Start event recording
  cudaEventRecord(fourierTransformStartEvent);

  if (modeFlag == gpuFFT::GPU_DFT_ONLY)
  {
    // Invoke the DFT
    dft <<< dataSize, dataSize >>> (realResiduals, imagResiduals, deviceRealData, deviceImagData, dataSize);
  }
  else if (modeFlag == gpuFFT::GPU_IDFT_ONLY)
  {
    // Invoke the IDFT
    idft <<< dataSize, dataSize >>> (realResiduals, imagResiduals, deviceRealData, deviceImagData, dataSize);
  }
  else
  {
    // Perform clean up
    freeDeviceMemory(deviceRealData);
    freeDeviceMemory(deviceImagData);
    freeDeviceMemory(realResiduals );
    freeDeviceMemory(imagResiduals );
    freeHostMemory(hostRealData);
    freeHostMemory(hostImagData);

    // Throw an error
    throw std::runtime_error("*** GPU FOURIER *** INVALID MODE FLAG");
  }

  cudaEventRecord(fourierTransformEndEvent);
  cudaEventSynchronize(fourierTransformEndEvent);
  
  cudaDeviceSynchronize();
  
  // Copy the data from the device to the host
  cudaMemcpy(hostRealData, realResiduals, deviceSize * dataSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostImagData, imagResiduals, deviceSize * dataSize, cudaMemcpyDeviceToHost);
  
  // INSTEAD OF USING STL VECTOR, CONSIDER USING THRUST VECTOR AND USING THE
  // REDUCE OPERTIONS
  std::vector<float> realResult(dataSize);
  std::vector<float> imagResult(dataSize);

  float reciprocal =
    (modeFlag == gpuFFT::GPU_DFT_ONLY ? 1.0f : 1.0f / dataSize);
  
  for (unsigned int i = 0; i < dataSize; ++i)
  {
    realResult[i] = 0.0f;
    imagResult[i] = 0.0f;
    for (unsigned int j = 0; j < dataSize; ++j)
    {
		  int index = (i * dataSize) + j;
		  realResult[i] += hostRealData[index] * reciprocal;
		  imagResult[i] += hostImagData[index] * reciprocal;
    }
  }
  
  // Write the results to a file
  gpuFFT::Filewriter gpuFourierWriter(filename);
  gpuFourierWriter.write(realResult, imagResult);

  // Free the device-memory
  freeDeviceMemory(deviceRealData);
  freeDeviceMemory(deviceImagData);
  freeDeviceMemory(realResiduals );
  freeDeviceMemory(imagResiduals );
 
  // Free the host-memory
  freeHostMemory(hostRealData);
  freeHostMemory(hostImagData);
}

/// @brief  Executes the CPU-based Discrete Fourier Transform
///         (and inverse Discrete Fourier Transform)
///
/// @param[in] real    The real parts of the input
/// @param[in] imag    The imaginary parts of the input

// TODO: ADD OUTPUTS FOR TIMINGS (DFT and IDFT)
void execute_cpu_fourier_transform(
  std::vector<float>& real,
  std::vector<float>& imag,
  gpuFFT::ModeFlag modeFlag)
{
	// Create the CPU-based DFT object
	gpuFFT::CPUDFT cpuFourier(real, imag);

	// Create vectors to contain the transformed data
	std::vector<float> transformedReal;
	std::vector<float> transformedImag;

  std::string filename;

  // Check the operating mode
  if (modeFlag == gpuFFT::CPU_DFT_ONLY)
  {
	  // Perform the Discrete Fourier Transform
	  // (note that the DFT operation does not affect the original input)
	  cpuFourier.dft(transformedReal, transformedImag);

    filename = "cpu_dft_output.dat";
  }

  else if (modeFlag == gpuFFT::CPU_IDFT_ONLY)
  {
	  // Perform the Inverse Discrete Fourier Transform
	  // (note that the IDFT operation does not affect the original input)
	  cpuFourier.idft(transformedReal, transformedImag);

    filename = "cpu_idft_output.dat";
  }

  // If the operating mode is invalid, throw a runtime error
  else
  {
    throw std::runtime_error("*** CPU FOURIER *** INVALID MODE FLAG");
  }

	// Write out the DFT results to a file
	gpuFFT::Filewriter cpuFourierWriter(filename);
	cpuFourierWriter.write(transformedReal, transformedImag);
}

// ====================================================
//                       MAIN
// ====================================================

int main(int argc, char* argv[])
{
  // Parse the inputs using the CommandLineParser
  gpuFFT::InputData parsedInput = gpuFFT::CommandLineParser::ParseInput(argc, argv);

  if (!parsedInput.valid)
  {
    show_program_usage();
    return EXIT_FAILURE;
  }

  // Check to see if the provided .dat file actually exists
  gpuFFT::Filereader inputFileReader(parsedInput.inputFile);
  if (!inputFileReader.exists())
  {
    show_program_usage();
    
    return EXIT_FAILURE;
  }
  
  // Provide the vectors for storing the real and imaginary portions
  // of the input data
  std::vector<float> realParts;
  std::vector<float> imagParts;
  
  // Read the data in from the file
  inputFileReader.readFile(realParts, imagParts);
  
  // ===========================================
  // Execute the GPU DFT
  // ===========================================
  if (parsedInput.mode & gpuFFT::ModeFlag::GPU_DFT_ONLY)
  {
    execute_gpu_fourier_transform(
      realParts, imagParts, gpuFFT::ModeFlag::GPU_DFT_ONLY);
  }

  // ===========================================
  // Execute the GPU IDFT
  // ===========================================
  if (parsedInput.mode & gpuFFT::GPU_IDFT_ONLY)
  {
    execute_gpu_fourier_transform(
      realParts, imagParts, gpuFFT::GPU_IDFT_ONLY);
  }

  // ===========================================
  // Execute the CPU DFT
  // ===========================================
  if (parsedInput.mode & gpuFFT::CPU_DFT_ONLY)
  {
    execute_cpu_fourier_transform(
      realParts, imagParts, gpuFFT::CPU_DFT_ONLY);
  }

  // ===========================================
  // Execute the CPU IDFT
  // ===========================================
  if (parsedInput.mode & gpuFFT::CPU_IDFT_ONLY)
  {
    execute_cpu_fourier_transform(
      realParts, imagParts, gpuFFT::CPU_IDFT_ONLY);
  }

  return EXIT_SUCCESS;
}