% -------------------------------------------------------------------------
function net = getConvArchitecture(dataset, network_arch)
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

  switch network_arch

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV0P0RL0+fcV1-RF32CH3'
      % FULLY CONNECTED
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 32, 3, 64, 5/100, 0, 'gaussian', 'gen');
      % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 3, 1, 64, 5/100, [0 0 1 1], 'gaussian', 'gen'); % Gaussian 3D
      % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 50, 1, 64, 5/100, [0 0 24 25], 'gaussian', 'gen'); % Gaussian 3D
      % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 34, 1, 64, 5/100, [0 0 16 17], 'gaussian', 'gen'); % UCI-ion
      % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 57, 1, 64, 5/100, [0 0 28 28], 'gaussian', 'gen'); % UCI-spam
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

      % LOSS LAYER
      % net.layers{end+1} = fh.softmaxlossLayer();
      net.layers{end+1} = fh.nnlossLayer();

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV0P0RL0+fcV1-RF32CH64'
      % FULLY CONNECTED
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 32, 64, 64, 5/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

      % LOSS LAYER
      % net.layers{end+1} = fh.softmaxlossLayer();
      net.layers{end+1} = fh.nnlossLayer();

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV0P0RL0+fcV1-RF16CH64'
      % FULLY CONNECTED
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 16, 64, 64, 5/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

      % LOSS LAYER
      % net.layers{end+1} = fh.softmaxlossLayer();
      net.layers{end+1} = fh.nnlossLayer();

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV0P0RL0+fcV1-RF4CH64'
      % FULLY CONNECTED
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 4, 64, 64, 5/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

      % LOSS LAYER
      % net.layers{end+1} = fh.softmaxlossLayer();
      net.layers{end+1} = fh.nnlossLayer();









    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV1P0RL0-RF32CH3+fcV1-RF32CH64'
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 64, 1/100, 2, 'gaussian', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      % net.layers{end+1} = fh.reluLayer(layer_number);

      % FULLY CONNECTED
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 32, 64, 64, 5/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

      % LOSS LAYER
      % net.layers{end+1} = fh.softmaxlossLayer();
      net.layers{end+1} = fh.nnlossLayer();

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV1P0RL1-RF32CH3+fcV1-RF32CH64'
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 64, 1/100, 2, 'gaussian', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % FULLY CONNECTED
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 32, 64, 64, 5/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

      % LOSS LAYER
      % net.layers{end+1} = fh.softmaxlossLayer();
      net.layers{end+1} = fh.nnlossLayer();

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV1P1RL1-RF32CH3+fcV1-RF16CH64'
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 64, 1/100, 2, 'gaussian', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % FULLY CONNECTED
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 16, 64, 64, 5/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

      % LOSS LAYER
      % net.layers{end+1} = fh.softmaxlossLayer();
      net.layers{end+1} = fh.nnlossLayer();

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV3P0RL0-RF32CH3+fcV1-RF32CH64'
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 32, 1/100, 2, 'gaussian', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      % net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 32, 5/100, 2, 'gaussian', 'gen');
      % net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 64, 5/100, 2, 'gaussian', 'gen');
      % net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 32, 64, 64, 5/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

      % LOSS LAYER
      % net.layers{end+1} = fh.softmaxlossLayer();
      net.layers{end+1} = fh.nnlossLayer();

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV3P0RL3-RF32CH3+fcV1-RF32CH64'
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 32, 1/100, 2, 'gaussian', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 32, 5/100, 2, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 64, 5/100, 2, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 32, 64, 64, 5/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

      % LOSS LAYER
      % net.layers{end+1} = fh.softmaxlossLayer();
      net.layers{end+1} = fh.nnlossLayer();

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV3P1RL3-RF32CH3+fcV1-RF16CH64'
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 32, 1/100, 2, 'gaussian', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 32, 5/100, 2, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 64, 5/100, 2, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 16, 64, 64, 5/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

      % LOSS LAYER
      % net.layers{end+1} = fh.softmaxlossLayer();
      net.layers{end+1} = fh.nnlossLayer();

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV3P3RL0-RF32CH3+fcV1-RF4CH64'
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 32, 1/100, 2, 'gaussian', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      % net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 32, 5/100, 2, 'gaussian', 'gen');
      % net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 64, 5/100, 2, 'gaussian', 'gen');
      % net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 4, 64, 64, 5/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

      % LOSS LAYER
      % net.layers{end+1} = fh.softmaxlossLayer();
      net.layers{end+1} = fh.nnlossLayer();

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV3P3RL3-RF32CH3+fcV1-RF4CH64'
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 32, 1/100, 2, 'gaussian', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 32, 5/100, 2, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 64, 5/100, 2, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 4, 64, 64, 5/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

      % LOSS LAYER
      % net.layers{end+1} = fh.softmaxlossLayer();
      net.layers{end+1} = fh.nnlossLayer();

  end

function number_of_output_nodes = getNumberOfOutputNodes(dataset)
  if isTwoClassImdb(dataset) || isSyntheticImdb(dataset)
    number_of_output_nodes = 2;
  elseif strcmp(dataset, 'coil-100')
    number_of_output_nodes = 100;
  else
    number_of_output_nodes = 10;
  end



