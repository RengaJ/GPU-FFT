#include "Complex.h"
#include "math_constants.h"

#include <fstream>
#include <iostream>

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
  std::ifstream inputFile(argv[1]);
  if (!inputFile)
  {
    show_program_usage();
    
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}