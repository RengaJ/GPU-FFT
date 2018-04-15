#ifndef GPU_FFT_FILEREADER_H
#define GPU_FFT_FILEREADER_H

#include <fstream>

namespace gpuFFT
{
  class Filereader
  {
    public:
      Filereader(char* filename);
      
      ~Filereader();
      
      bool exists();
      
      void readFile();
      
    private:
      std::ifstream _stream;
  };
}

#endif