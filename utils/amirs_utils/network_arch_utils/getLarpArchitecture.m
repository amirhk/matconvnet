% -------------------------------------------------------------------------
function net = getLarpArchitecture(dataset, network_arch)
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

  fh = networkInitializationUtils;
  net.layers = {};
  index_of_smooth_boolean = findstr(network_arch, 'S');
  weight_init_type = 'compRand';
  if index_of_smooth_boolean < length(network_arch) % if the S# flag even exists in the architecture.
    switch network_arch(index_of_smooth_boolean + 1)
      case 'T'
        smoothed_kernels = true;
        weight_init_type = 'compRandSmoothed';
      case 'F'
        smoothed_kernels = false;
        weight_init_type = 'compRand';
      otherwise
        throwException('[ERROR] unrecognizable flag for smoothness of kernels.');
    end
    network_arch = network_arch(1:index_of_smooth_boolean-1);
  else
    throwException('[ERROR] no indication of smootherness in the larp architecture.');
  end


  switch network_arch
    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'larpV0P0'
      % LARP
      % N/A

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'larpV1P0'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 64, 1/100, 2, weight_init_type, 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'larpV1P1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 64, 1/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'larpV3P0'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 32, 1/100, 2, weight_init_type, 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 32, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'larpV3P1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 32, 1/100, 2, weight_init_type, 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 32, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'larpV3P3'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 32, 1/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 32, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'larpV5aP0'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 96, 5/1000, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 96, 256, 5/1000, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 3, 256, 384, 5/1000, 1, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 3, 384, 384, 5/1000, 1, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 3, 384, 64, 5/1000, 1, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'larpV5aP1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 96, 5/1000, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 96, 256, 5/1000, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 3, 256, 384, 5/1000, 1, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 3, 384, 384, 5/1000, 1, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 3, 384, 64, 5/1000, 1, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'larpV5aP3'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 96, 5/1000, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 96, 256, 5/1000, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 3, 256, 384, 5/1000, 1, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 3, 384, 384, 5/1000, 1, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 3, 384, 64, 5/1000, 1, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'larpV5aP5'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 96, 5/1000, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 96, 256, 5/1000, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 3, 256, 384, 5/1000, 1, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 3, 384, 384, 5/1000, 1, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 3, 384, 64, 5/1000, 1, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'larpV5hP0'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 64, 1/100, 2, weight_init_type, 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 64, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 64, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 64, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 64, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'larpV5hP1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 64, 1/100, 2, weight_init_type, 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 64, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 64, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 64, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 64, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'larpV5hP3'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 64, 1/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 64, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 64, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 64, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 64, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'larpV5hP5'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 64, 1/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 64, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 64, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 64, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 64, 64, 5/100, 2, weight_init_type, 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);


    % case 'larpV0sP0'
    % case 'larpV1sP0'
    % case 'larpV2sP0'
    % case 'larpV1lP0'
    % case 'larpV1lP1'
    % case 'larpV2lP0'
    % case 'larpV2lP1'
    % case 'larpV2lP2'
  end
