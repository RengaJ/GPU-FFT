#include "Filewriter.h"

namespace gpuFFT
{
  /// @brief Constructor for the Filewriter class.
  ///
  /// @param [in] filename  The name of the file to be written to
  Filewriter::Filewriter(std::string filename)
  {
    _outputStream.open(filename.c_str(), std::ios::out);
  }


  /// @brief Destructor for the Filewriter class. Closes the output stream if
  ///        it was opened.
  Filewriter::~Filewriter()
  {
    if (_outputStream.is_open())
    {
      _outputStream.close();
    }
  }

  /// @brief Write complex data to the previously opened file
  ///
  /// @param [in] real   The vector containing the real portions of the data
  /// @param [in] imag   The vector containing the imaginary portions of the data.
  void Filewriter::write(std::vector<float>& real, std::vector<float>& imag)
  {
    for (size_t i = 0; i < real.size(); ++i)
    {
      _outputStream << real[i] << " " << imag[i];
      
      if (i < real.size() - 1)
      {
        outputStream << " ";
      }
    }
  }
