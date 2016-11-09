% --------------------------------------------------------------------
function sequence_string = printWeightInitSequence(sequence)
% --------------------------------------------------------------------
  sequence_string = '';
  for i=1:length(sequence)
    sequence_string = sprintf('%s%s, ', sequence_string, sequence{i});
  end
  sequence_string = sequence_string(1:end-1);
