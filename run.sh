#!/bin/bash

echo "Attempting to compile gpuFFT.exe"

if [ -z "$NVCC_LIBRARY_PATH" ]; then
  echo "Please set the nvcc library path variable (NVCC_LIBRARY_PATH) for proper compilation."
  exit 1
fi

nvcc CPUDFT.cpp FileWriter.cpp CommandLineParser.cpp Filereader.cpp main.cu -L "$NVCC_LIBRARY_PATH" -lcudart -o gpuFFT.exe

./gpuFFT.exe test_input.dat --cpu-dft
