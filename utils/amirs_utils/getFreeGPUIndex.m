% -------------------------------------------------------------------------
function randomGPUIndex = getFreeGPUIndex()
% -------------------------------------------------------------------------
% Copyright (c) 2017, Amir-Hossein Karimi
% All rights reserved.

% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution

% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.

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
