#include "Filereader.h"
#include <string>
#include <sstream>

// Expected file-format:
//
// <number of entries>
// <entry list>

namespace gpuFFT
{
  Filereader::Filereader(std::string filename)
  {
    _stream = std::ifstream(filename.c_str());
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