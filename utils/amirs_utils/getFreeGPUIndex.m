% -------------------------------------------------------------------------
function randomGPUIndex = getFreeGPUIndex()
% -------------------------------------------------------------------------
  totalCount = gpuDeviceCount();
  freeGPUs = [];
  fprintf('[INFO] Found %d mounted GPUs.\n', totalCount);
  for i = 1:gpuDeviceCount()
      gpuInfo = gpuDevice(i);
      totalMemory = gpuInfo.TotalMemory;
      availableMemory = gpuInfo.AvailableMemory;
      if availableMemory / totalMemory > .95
        freeGPUs(end + 1) = i;
      end
  end
  fprintf('[INFO] Identified %d free GPUs (based on %%95+ free memory)...\n', length(freeGPUs));
  if length(freeGPUs) > 0
    randomGPUIndex = freeGPUs(1);
    fprintf('[INFO] Randomly choosing GPU #%d to run on!\n', randomGPUIndex);
    fprintf('[INFO] Flushing GPU... ');
    reset(gpuDevice(randomGPUIndex));
    fprintf('done.\n\n');
  else
    fprintf('[INFO] no GPUs are free. Running on CPU instead...\n\n');
    randomGPUIndex = -1;
  end
