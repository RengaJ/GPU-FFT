#ifndef GPU_FFT_COMPLEX_H
#define GPU_FFT_COMPLEX_H

// Denote if a function is CUDA callable
//
// Source found at:
// https://stackoverflow.com/questions/6978643/cuda-and-classes
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#include <string>
#include <sstream>

namespace gpuFFT
{
  struct Complex
  {
    //public:
    //
    //  /// @brief Default constructor for a Complex number
      CUDA_CALLABLE Complex()
      {
        real = 0.0f;
        imag = 0.0f;
      }
    //  
    //  /// @brief Full constructor for a Complex number
    //  ///
    //  /// @param [in] _real The real part of the complex number
    //  /// @param [in] _imag The imaginary part of the complex number
    //  CUDA_CALLABLE Complex(float _real, float _imag)
    //  {
    //    real = _real;
    //    imag = _imag;
    //  }
    //  
    //  /// @brief Destructor for Complex number
    //  CUDA_CALLABLE virtual ~Complex() {}
    //  
    //  /// @brief Obtain the complex-conjugate of the
    //  ///        current Complex number. Does not affect
    //  ///        the current Complex number.
    //  ///
    //  /// @return The complex-conjugate of the current Complex
    //  CUDA_CALLABLE Complex conj()
    //  {
    //    return Complex(real, -imag);
    //  }
    // 
    //  /// @brief Take the complex-conjugate of the current Complex
    //  ///        number. This DOES affect the current Complex number.
    //  CUDA_CALLABLE void conjugate()
    //  {
    //    imag = -imag;
    //  }
    //  
    //  CUDA_CALLABLE Complex& operator=(const Complex& other)
    //  {
    //    real = other.real;
    //    imag = other.imag;
    //    
    //    return *this;
    //  }
    //  
    //  /// @brief Addition of two Complex numbers
    //  ///
    //  /// @param [in] other The other complex number
    //  ///
    //  /// @return The resulting complex number
    //  CUDA_CALLABLE Complex operator+(const Complex& other)
    //  {
    //    return Complex(real + other.real, imag + other.imag);
    //  }
    //  
    //  /// @brief Subtraction of two Complex numbers
    //  ///
    //  /// @param [in] other The other complex number
    //  ///
    //  /// @return The resulting complex number
    //  CUDA_CALLABLE Complex operator-(const Complex& other)
    //  {
    //    return Complex(real - other.real, imag - other.imag);
    //  }
    //  
    //  /// @brief Multiplication of two Complex numbers
    //  ///
    //  /// @param [in] other The other complex number
    //  ///
    //  /// @return The resulting complex number
    //  CUDA_CALLABLE Complex operator*(const Complex& other)
    //  {
    //    // Complex multiplation:
    //    // (a + bi)(c + di) = ac + adi + bci - bd = (ac - bd) + (ad + bc)i
    //    return Complex((real * other.real) - (imag * other.imag),
    //                   (real * other.imag) + (imag * other.real));
    //  }
    //  
    //  /// @brief Multiplication of a Complex number by a scalar
    //  ///
    //  /// @param [in] scalar The scaling factor
    //  ///
    //  /// @return The scaled Complex number
    //  CUDA_CALLABLE Complex operator*(const float scalar)
    //  {
    //    return Complex(real * scalar, imag * scalar);
    //  }
    //  
    //  /// @brief Division of two Complex numbers
    //  ///
    //  /// @param [in] other The other complex number
    //  ///
    //  /// @return The resulting complex number
    //  CUDA_CALLABLE Complex operator/(const Complex& other)
    //  {
    //    // Complex division:
    //    // (a + bi)/(c + di) = ((ac + bd)/(c^2 + d^2)) + ((bc - ad)/(c^2 + d^2))i
    //    
    //    float denominator = (other.real * other.real) + (other.imag * other.imag);
    //    
    //    return Complex(((real * other.real) + (imag * other.imag)) / denominator,
    //                   ((imag * other.real) - (real * other.imag)) / denominator);
    //  }
    //  
    //  /// @brief Division of a Complex number by a scalar
    //  ///
    //  /// @param [in] scalar The scaling factor
    //  ///
    //  /// @return The resulting complex number
    //  CUDA_CALLABLE Complex operator/(const float other)
    //  {
    //    return Complex(real / other, imag / other);
    //  }
    //  
    //  /// @brief Add and store the addition of another complex
    //  ///        number.
    //  ///
    //  /// @param [in] other The other complex number
    //  ///
    //  /// @return The newly updated complex number
    //  CUDA_CALLABLE Complex& operator+=(const Complex& other)
    //  {
    //    real += other.real;
    //    imag += other.imag;
    //    
    //    return *this;
    //  }
    //  
    //  /// @brief Add and store the subtraction of another complex
    //  ///        number.
    //  ///
    //  /// @param [in] other The other complex number
    //  ///
    //  /// @return The newly updated complex number
    //  CUDA_CALLABLE Complex& operator-=(const Complex& other)
    //  {
    //    real -= other.real;
    //    imag -= other.imag;
    //    
    //    return *this;
    //  }
    //  
    //  /// @brief Add and store the multiplication of another complex
    //  ///        number.
    //  ///
    //  /// @param [in] other The other complex number
    //  ///
    //  /// @return The newly updated complex number
    //  CUDA_CALLABLE Complex& operator*=(const Complex& other)
    //  {
    //    // Complex multiplation:
    //    // (a + bi)(c + di) = ac + adi + bci - bd = (ac - bd) + (ad + bc)i
    //    float newReal = ((real * other.real) - (imag * other.imag));
    //    float newImag = ((real * other.imag) + (imag * other.real));

    //    real = newReal;
    //    imag = newImag;
    //    
    //    return *this;
    //  }
    //  
    //  /// @brief Multiplication of a Complex number by a scalar
    //  ///
    //  /// @param [in] scalar The scaling factor
    //  ///
    //  /// @return The newly updated complex number
    //  CUDA_CALLABLE Complex& operator*=(const float scalar)
    //  {
    //    real *= scalar;
    //    imag *= scalar;
    //    
    //    return *this;
    //  }
    //  
    //  /// @brief Division of two Complex numbers
    //  ///
    //  /// @param [in] other The other complex number
    //  ///
    //  /// @return The newly updated complex number
    //  CUDA_CALLABLE Complex& operator/=(const Complex& other)
    //  {
    //    // Complex division:
    //    // (a + bi)/(c + di) = ((ac + bd)/(c^2 + d^2)) + ((bc - ad)/(c^2 + d^2))i
    //    
    //    float denominator = (other.real * other.real) + (other.imag * other.imag);
    //    
    //    float newReal = ((real * other.real) + (imag * other.imag)) / denominator;
    //    float newImag = ((imag * other.real) - (real * other.imag)) / denominator;
    //    
    //    real = newReal;
    //    imag = newImag;
    //    
    //    return *this;
    //  }
    //  
    //  /// @brief Division of a Complex number by a scalar
    //  ///
    //  /// @param [in] scalar The scaling factor
    //  ///
    //  /// @return The newly updated complex number
    //  CUDA_CALLABLE Complex& operator/=(const float other)
    //  {
    //    real /= other;
    //    imag /= other;
    //    
    //    return *this;
    //  }
    //  
    //  std::string toString()
    //  {
    //    std::stringstream ss;
    //    
    //    ss << real;
    //    
    //    if (imag >= 0.0f)
    //    {
    //      ss << "+";
    //    }
    //    
    //    ss << imag << "i";
    //    
    //    return ss.str();
    //  }
      
      /// The real portion of the complex number
      float real;
      
      /// The imaginary portion of the complex number
      float imag;
  };
}
#endif