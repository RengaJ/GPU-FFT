#include "CommandLineParser.h"

#include <iostream>
#include <string>

namespace gpuFFT
{
  InputData CommandLineParser::ParseInput(int argc, char* argv[])
  {
    // Create an empty InputData structure to be filled out
    InputData parsedInputs;

    // Check the number of arguments to ensure they are correct:
    if (argc != 3)
    {
      // If the number of arguments is not correct, return the empty
      // (and invalid) InputData structure.
      //
      // This should be handled by the caller.

      std::cout << "<< ERROR >> INVALID NUMBER OF ARGUMENTS PROVIDED";
      std::cout << " - EXPECTED 3, DETECTED " << argc << std::endl;

      return parsedInputs;
    }

    // We know that the number of arguments is correct. Now it's time to
    // properly parse the values. First is the input file-name. It should
    // terminate with a '.dat' extension:
    std::string inputFilename(argv[1]);
    size_t inputSize = inputFilename.length();
    if (inputFilename.substr(inputSize - 4, 4) != ".dat")
    {
      // If .dat is not the final four characters of the input filename,
      // return the empty (and invalid) InputData structure.

      std::cout << "<< ERROR >> PROVIDED INPUT FILE DOES NOT HAVE .DAT EXTENSION" << std::endl;

      return parsedInputs;
    }

    // At this point, the input file should be okay, so assign the value into the
    // InputData structure
    parsedInputs.inputFile = inputFilename;

    // Check the operating mode flags
    std::string modeFlag = argv[2];

    if (modeFlag == "--gpu-dft")       { parsedInputs.mode = GPU_DFT_ONLY;  }
    else if (modeFlag == "--gpu-idft") { parsedInputs.mode = GPU_IDFT_ONLY; }
    else if (modeFlag == "--cpu-dft")  { parsedInputs.mode = CPU_DFT_ONLY;  }
    else if (modeFlag == "--cpu-idft") { parsedInputs.mode = CPU_IDFT_ONLY; }
    else if (modeFlag == "--compare")  { parsedInputs.mode = COMPARISON;    }
    else
    {
      // If an invalid flag is found, return the 
      // current state of the parsed inputs
      std::cout << "<< ERROR >> INVALID MODE FLAG DETECTED" << std::endl;

      return parsedInputs;
    }

    // At this point, the inputs are properly assigned and valid, so make the
    // structure indicate this.
    parsedInputs.valid = true;

    // Return the completed InputData structure
    return parsedInputs;
  }
}