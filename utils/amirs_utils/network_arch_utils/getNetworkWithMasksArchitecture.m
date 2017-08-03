% -------------------------------------------------------------------------
function net = getNetworkWithMasksArchitecture(dataset, network_arch, weight_init_sequence)
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








    case 'larpV1M0P1+convV0P0+fcV1'
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 64, 1/100, 2, char(weight_init_sequence{1}), 'gen');
      % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, 64, 1/100, 0, 'gaussian', 'gen');
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
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV1M1P1+convV0P0+fcV1'
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 64, 1/100, 2, char(weight_init_sequence{1}), 'gen');
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, 64, 1/100, 0, 'gaussian', 'gen');
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
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --








    case 'larpV3M0P1+convV0P0+fcV1'
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 32, 1/100, 2, char(weight_init_sequence{1}), 'gen');
      % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, 64, 1/100, 0, 'gaussian', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 32, 5/100, 2, char(weight_init_sequence{2}), 'gen');
      % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, 64, 1/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 64, 5/100, 2, char(weight_init_sequence{3}), 'gen');
      % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, 64, 1/100, 0, 'gaussian', 'gen');
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
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV3M1P1+convV0P0+fcV1'
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 32, 1/100, 2, char(weight_init_sequence{1}), 'gen');
      % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 32, 32, 1/100, 0, 'gaussian', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 32, 5/100, 2, char(weight_init_sequence{2}), 'gen');
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 32, 32, 1/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 64, 5/100, 2, char(weight_init_sequence{3}), 'gen');
      % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, 64, 1/100, 0, 'gaussian', 'gen');
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
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV3M2P1+convV0P0+fcV1'
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 32, 1/100, 2, char(weight_init_sequence{1}), 'gen');
      % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 32, 32, 1/100, 0, 'gaussian', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 32, 5/100, 2, char(weight_init_sequence{2}), 'gen');
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 32, 32, 1/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 64, 5/100, 2, char(weight_init_sequence{3}), 'gen');
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, 64, 1/100, 0, 'gaussian', 'gen');
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
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV3M3P1+convV0P0+fcV1'
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 32, 1/100, 2, char(weight_init_sequence{1}), 'gen');
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 32, 32, 1/100, 0, 'gaussian', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 32, 5/100, 2, char(weight_init_sequence{2}), 'gen');
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 32, 32, 1/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 64, 5/100, 2, char(weight_init_sequence{3}), 'gen');
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, 64, 1/100, 0, 'gaussian', 'gen');
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
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --








    case 'larpV3M0P3+convV0P0+fcV1'
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 32, 1/100, 2, char(weight_init_sequence{1}), 'gen');
      % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 32, 32, 1/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 32, 5/100, 2, char(weight_init_sequence{2}), 'gen');
      % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 32, 32, 1/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 64, 5/100, 2, char(weight_init_sequence{3}), 'gen');
      % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, 64, 1/100, 0, 'gaussian', 'gen');
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
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV3M1P3+convV0P0+fcV1'
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 32, 1/100, 2, char(weight_init_sequence{1}), 'gen');
      % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 32, 32, 1/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 32, 5/100, 2, char(weight_init_sequence{2}), 'gen');
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 32, 32, 1/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 64, 5/100, 2, char(weight_init_sequence{3}), 'gen');
      % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, 64, 1/100, 0, 'gaussian', 'gen');
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
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV3M3P3+convV0P0+fcV1'
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 3, 32, 1/100, 2, char(weight_init_sequence{1}), 'gen');
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 32, 32, 1/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 32, 5/100, 2, char(weight_init_sequence{2}), 'gen');
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 32, 32, 1/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 5, 32, 64, 5/100, 2, char(weight_init_sequence{3}), 'gen');
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, 64, 1/100, 0, 'gaussian', 'gen');
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
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --








  end

