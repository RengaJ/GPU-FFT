function generateRandomComplexNumbers( filename, numElements )
% Randomly generate complex numbers and write to a file

  % check to see if the provided filename has a .dat extension
  [~, ~, ext] = fileparts(filename);
  if (~strcmp(ext, '.dat'))
      error('<< ERROR >> Provided file name does not have a .dat extension.');
  end
  
  % open the file
  FileHandle = fopen(filename, 'w');

  ComplexNumbers = complex(rand(1, numElements), rand(1, numElements));
  
  for ComplexIndex = 1:numElements
      ComplexNumber = ComplexNumbers(ComplexIndex);
      fprintf(FileHandle, '%f %f', real(ComplexNumber), imag(ComplexNumber));
      
      if (ComplexIndex < numElements)
          fprintf(FileHandle, ' ');
      end
  end
  
  % Close the file
  fclose(FileHandle);
end