function PlotData( filename )
% Plots Fourier Transform Data as either time or frequency space

  % First, check to see if the provided file exists
  if (~exist(filename, 'file'))
      error('<< ERROR >> Provided file does not exist.');
  end

  % Second, check to see if the provided file has a .dat extension
  [~, FileName, extension] = fileparts(filename);
  if (~strcmp(extension, '.dat'))
      error('<< ERROR >> Provided file does not have a .dat extension.');
  end

  % Now, let's read in the data
  FileData = dlmread(filename, ' ');

  % Separate the data into real and imaginary parts
  NumElements = length(FileData);
  Reals = zeros(1, NumElements/2);
  Imags = zeros(1, NumElements/2);

  ComplexCounter = 1;

  for DataIdx = 1:2:length(FileData)
      Reals(ComplexCounter) = FileData(DataIdx);
      Imags(ComplexCounter) = FileData(DataIdx + 1);
      
      ComplexCounter = ComplexCounter + 1;
  end

  Magnitudes = sqrt((Reals .^ 2) + (Imags .^ 2));

  TitleString = sprintf('Complex Magnitudes - %s', FileName);

  figure('Name', 'Complex Magnitudes', 'NumberTitle', 'off');
  plot(Magnitudes, '-b', 'LineWidth', 2);
  grid on;
  grid minor;

  xlabel('Complex Number Index');
  ylabel('Complex Magnitude');
  title (TitleString, 'Interpreter', 'none');
end