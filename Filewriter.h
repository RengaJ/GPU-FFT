#ifndef GPUFFT_FILEWRITER_H
#define GPUFFT_FILEWRITER_H

#include <string>
#include <fstream>
#include <vector>

namespace gpuFFT
{
  /// @brief Class that provides file writing capabilities.
  class Filewriter
  {
    public:
      Filewriter(std::string filename);
      ~Filewriter();

      void write(std::vector<float>& real, std::vector<float>& imag);
    private:
      std::ofstream _outputStream;
  };
}
#endif