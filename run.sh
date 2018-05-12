#!/bin/bash

echo "Attempting to compile gpuFFT.exe"

if [ -z "$NVCC_LIBRARY_PATH" ]; then
  echo "Please set the nvcc library path variable (NVCC_LIBRARY_PATH) for proper compilation."
  exit 1
fi

nvcc CPUDFT.cpp FileWriter.cpp CommandLineParser.cpp Filereader.cpp main.cu -L "$NVCC_LIBRARY_PATH" -lcudart -o gpuFFT.exe

echo ""
echo "*********************************"

# Prompt the user for an input file name
echo "Provide Input File Path:"
read INPUT_FILE

echo ""
echo ""

# Provide the menu of operation modes that the program can accept
# (ask them to type in the value)
echo "Select one operation mode:"
echo "gpu-dft   : Performs the GPU-Based DFT"
echo "gpu-idft  : Performs the GPU-Based Inverse DFT"
echo "cpu-dft   : Performs the CPU-Based DFT"
echo "cpu-idft  : Performs the CPU-Based Inverse DFT"
echo "compare   : Performs all of the operations on the input"
read OPERATION_MODE

echo ""
echo "*********************************"

echo ""

OPERATION_FLAG="--$OPERATION_MODE"

./gpuFFT.exe $INPUT_FILE $OPERATION_FLAG
