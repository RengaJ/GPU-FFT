#ifndef GPU_FFT_INPUT_DATA_H
#define GPU_FFT_INPUT_DATA_H

#include <string>

namespace gpuFFT
{

	enum ModeFlag
	{
		UNKNOWN       = 0x0,   // Current Mode is UNKNOWN
		GPU_DFT_ONLY  = 0x1,   // Execute only the GPU DFT operations
		GPU_IDFT_ONLY = 0x2,   // Execute only the GPU IDFT operations
		CPU_DFT_ONLY  = 0x4,   // Execute only the CPU DFT operations
		CPU_IDFT_ONLY = 0x8,   // Execute only the CPU IDFT operations
		COMPARISON    = 0xF    // Execute all comparison operations
	};

	struct InputData
	{
		/// @brief Constructor for the InputData structure
		InputData()
		{
			inputFile = "__INVALID__";
			mode      = UNKNOWN;
			valid     = false;
		}

		std::string inputFile; // The name of the input file
		ModeFlag mode;         // What is the current mode
		bool valid;            // Was the input successfully parsed?
	};
}
#endif