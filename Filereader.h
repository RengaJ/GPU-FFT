#ifndef GPU_FFT_FILEREADER_H
#define GPU_FFT_FILEREADER_H

#include <vector>
#include <fstream>
#include "Complex.h"

namespace gpuFFT
{
  class Filereader
  {
    public:
      Filereader(char* filename);
      
      ~Filereader();
      
      bool exists();
      
      void readFile(std::vector<Complex>& data);
      void readFile(std::vector<float>& reals, std::vector<float>& imags);
      
    private:
      std::ifstream _stream;
  };
}

#endif