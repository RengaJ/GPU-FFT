#include "Filereader.h"
#include <string>
#include <sstream>

namespace gpuFFT
{
  /// @brief Full constructor for the Filereader class
  ///
  /// @param [in] filename   The string that contains the name of the file
  ///                        to read.
  Filereader::Filereader(std::string filename)
  {
    _stream = std::ifstream(filename.c_str());
  }
  
  /// @brief Destructor for the Filereader class. Will close the stream
  ///        if it's been opened.
  Filereader::~Filereader()
  {
    if (_stream && _stream.is_open())
    {
      _stream.close();
    }
  }
  
  /// @brief Determines if the file exists (if it's been opened)
  bool Filereader::exists()
  {
    return _stream.is_open();
  }
  
  /// @brief Read the contents of the file and fill out the complex data.
  ///
  /// @param [inout] reals  The vector that contains the real-parts of the complex data
  /// @param [inout] imags  The vector that contains the imaginary-parts of the complex data
  void Filereader::readFile(std::vector<float>& reals,
                            std::vector<float>& imags)
  {
    reals.clear();
    imags.clear();
    
    float realValue;
    float imagValue;
    
    while (_stream.good())
    {
      _stream >> realValue >> imagValue;
      reals.push_back(realValue);
      imags.push_back(imagValue);
    }
  }
}