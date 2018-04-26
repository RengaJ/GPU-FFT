#ifndef GPUFFT_CPUDFT_H
#define GPUFFT_CPUDFT_H

#include <vector>

namespace gpuFFT
{
	class CPUDFT
	{
	public:
		CPUDFT(std::vector<float> real, std::vector<float> imag);
		~CPUDFT();

		void dft (std::vector<float>& realTransformed, std::vector<float>& imagTransformed);
		void idft(std::vector<float>& realTransformed, std::vector<float>& imagTransformed);
	private:
		std::vector<float> _realParts;
		std::vector<float> _imagParts;
	};
}

#endif
