% -------------------------------------------------------------------------
function processor_list = ifNotMacSetGpu(gpu_index)
% -------------------------------------------------------------------------
  if ispc
    processor_list = [gpu_index]; % GPU at index 1
  else
    processor_list = [];
  end
