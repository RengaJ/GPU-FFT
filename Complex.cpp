#include "Complex.h"

namespace gpuFFT
{
  /// @brief Default constructor for a Complex number
  Complex::Complex()
  {
    real = 0.0f;
    imag = 0.0f;
  }
  
  /// @brief Full constructor for a Complex number
  ///
  /// @param [in] _real The real part of the complex number
  /// @param [in] _imag The imaginary part of the complex number
  Complex::Complex(float _real, float _imag)
  {
    real = _real;
    imag = _imag;
  }
  
  /// @brief Obtain the complex-conjugate of the
  ///        current Complex number. Does not affect
  ///        the current Complex number.
  ///
  /// @return The complex-conjugate of the current Complex 
  Complex Complex::conj()
  {
    return Complex(real, -imag);
  }
  
  /// @brief Take the complex-conjugate of the current Complex
  ///        number. This DOES affect the current Complex number.
  void Complex::conjugate()
  {
    imag = -imag;
  }
}