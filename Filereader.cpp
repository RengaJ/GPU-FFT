#include "Filereader.h"

namespace gpuFFT
{
  Filereader::Filereader(char* filename)
  {
    _stream = std::ifstream(filename);
  }
  
  Filereader::~Filereader()
  {
    if (_stream && _stream.is_open())
    {
      _stream.close();
    }
  }
  
  bool Filereader::exists()
  {
    return _stream.is_open();
  }
}