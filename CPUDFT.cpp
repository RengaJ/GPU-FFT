#include "CPUDFT.h"
#include <exception>

#define _USE_MATH_DEFINES
#include <math.h>

namespace gpuFFT
{
  /// @brief CPU-DFT Constructor
  ///
  /// @param [in] realInput  The vector containing the real parts of the input data
  /// @param [in] imagInput  The vector containing the imaginary parts of the input data
  ///
  /// @exception If the realInput and imagInput vectors are not the same size, a std::runtime_error
  ///            will be thrown and the program will terminate
  CPUDFT::CPUDFT(std::vector<float> realInput, std::vector<float> imagInput)
  {
    _realParts = std::vector<float>(realInput.begin(), realInput.end());
    _imagParts = std::vector<float>(imagInput.begin(), imagInput.end());

    if (_realParts.size() != _imagParts.size())
    {
      throw std::runtime_error("Real and Imaginary Parts are not the same size");
    }
  }

  /// @brief CPU-DFT Destructor
  CPUDFT::~CPUDFT()
  {
    // There is nothing to do here
  }

  /// @brief Compute the Discrete Fourier Transform
  ///
  /// @param [out] realTransformed    The vector containing the real parts of the DFT results
  /// @param [out] imagTransformed    The vector containing the imaginary parts of the DFT results
  void CPUDFT::dft(std::vector<float>& realTransformed, std::vector<float>& imagTransformed)
  {
    // Ensure the outputs are clear
    realTransformed.clear();
    imagTransformed.clear();

    // Discrete Fourier Transform:

    //         N - 1            -kn
    //   X  =  sigma (f  * omega   )
    //    k    n = 0   n        n

    // Grab the size of the data
    size_t N = _realParts.size();

    // Inflate the output to the correct sizes
    realTransformed.resize(N, 0.0f);
    imagTransformed.resize(N, 0.0f);

    // Iterate through all k elements
    for (size_t k = 0; k < N; ++k)
    {
      // Iterate through all n elements
      for (size_t n = 0; n < N; ++n)
      {
        // Compute Omega_n:
        // Omega_n = cos( (-2kn[PI]) / N) + isin( (-2kn[PI]) / N)
        float theta = -2.0f * k * n * (float)M_PI / N;
        float omega_real = cos(theta);
        float omega_imag = sin(theta);

        // Compute the real and imaginary parts
        //
        // (a + bi) * (c + di) = (ac - bd) + i(ad + bc)
        //
        realTransformed[k] += ((_realParts[n] * omega_real) - (_imagParts[n] * omega_imag));
        imagTransformed[k] += ((_realParts[n] * omega_imag) + (_imagParts[n] * omega_real));
      }
    }
  }

  /// @brief Compute the Inverse Discrete Fourier Transform
  ///
  /// @param [out] realTransformed    The vector containing the real parts of the IDFT results
  /// @param [out] imagTransformed    The vector containing the imaginary parts of the IDFT results
  void CPUDFT::idft(std::vector<float>& realTransformed, std::vector<float>& imagTransformed)
  {
    // Ensure the outputs are clear
    realTransformed.clear();
    imagTransformed.clear();

    // Inverse Discrete Fourier Transform:

    //       1   N - 1            kn
    //  f  = - * sigma (X  * omega  )
    //   n   N   k = 0   k        n

    // Grab the size of the data
    size_t N = _realParts.size();

    // Inflate the output to the correct size
    realTransformed.resize(N, 0.0f);
    imagTransformed.resize(N, 0.0f);

    // Iterate through all n elements
    for (size_t n = 0; n < N; ++n)
    {
      // Iterate through all k elements
      for (size_t k = 0; k < N; ++k)
      {
        // Compute Omega_n:
        // Omega_n = cos( (2kn[PI]) / N) + isin( (2kn[PI]) / N)
        float theta = 2.0f * k * n * (float)M_PI / N;
        float omega_real = cos(theta);
        float omega_imag = sin(theta);

        // Compute the real and imaginary parts
        //
        // (a + bi) * (c + di) = (ac - bd) + i(ad + bc)
        //
        realTransformed[n] += ((_realParts[k] * omega_real) - (_imagParts[k] * omega_imag));
        imagTransformed[n] += ((_realParts[k] * omega_imag) + (_imagParts[k] * omega_real));
      }

      // Divide the results of the real and imaginary by N
      realTransformed[n] = realTransformed[n] / N;
      imagTransformed[n] = imagTransformed[n] / N;
    }
  }
}