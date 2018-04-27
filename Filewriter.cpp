#include "Filewriter.h"

namespace gpuFFT
{

	Filewriter::Filewriter(std::string filename)
	{
		_outputStream.open(filename.c_str(), std::ios::out);
	}


	Filewriter::~Filewriter()
	{
		if (_outputStream.is_open())
		{
			_outputStream.close();
		}
	}

	void Filewriter::write(std::vector<float>& real, std::vector<float>& imag)
	{
		for (size_t i = 0; i < real.size(); ++i)
		{
			_outputStream << real[i];

			if (imag[i] >= 0.0f)
			{
				_outputStream << "+";
			}
			_outputStream << imag[i] << std::endl;
		}
	}
}