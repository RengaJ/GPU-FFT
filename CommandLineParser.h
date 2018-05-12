#ifndef GPU_FFT_COMMAND_LINE_PARSER_H
#define GPU_FFT_COMMAND_LINE_PARSER_H

#include "InputData.h"

namespace gpuFFT
{
  /// @brief Class to handle the parsing of input arguments
	class CommandLineParser
	{
		public:
			static InputData ParseInput(int argc, char* argv[]);
		private:
			CommandLineParser() {}   // DO NOTHING
			~CommandLineParser() {}  // DO NOTHING
	};
}
#endif