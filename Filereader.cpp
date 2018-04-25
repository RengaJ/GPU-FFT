#include "Filereader.h"
#include <string>
#include <sstream>

// Expected file-format:
//
// <number of entries>
// <entry list>

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
  
  void Filereader::readFile(std::vector<Complex>& data)
  {
    // Get the number of entries from the file
    std::string dataLine;
    std::getline(_stream, dataLine);
    
    int numberOfEntries = atoi(dataLine.c_str());
    
    // Ensure the data vector is empty
    
    // Populate the data vector
    std::getline(_stream, dataLine);
    
    float real;
    std::stringstream ss(dataLine);
    for (int i = 0; i < numberOfEntries; ++i)
    {
      ss >> real;
      
	  Complex entry;
	  entry.real = real;
      data.push_back(entry);
    }
  }
  
  void Filereader::readFile(std::vector<float>& reals,
                            std::vector<float>& imags)
  {
    std::string dataLine;
    std::getline(_stream, dataLine);
    
    int numberOfEntries = atoi(dataLine.c_str());
    
    // Ensure the data vector is empty
    
    // Populate the data vector
    std::getline(_stream, dataLine);
    
    float real;
    std::stringstream ss(dataLine);
    for (int i = 0; i < numberOfEntries; ++i)
    {
      ss >> real;
      reals.push_back(real);   // Real
      imags.push_back(0.0f);   // Imaginary
    }
  }
}