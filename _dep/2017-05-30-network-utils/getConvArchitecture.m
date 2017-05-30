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
    case 'convV0P0+fcV1RF32CH3'
      % FULLY CONNECTED
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 32, 3, 64, 5/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV0P0+fcV1RF32CH64'
      % FULLY CONNECTED
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 32, 64, 64, 5/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV0P0+fcV1RF16CH64'
      % FULLY CONNECTED
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 16, 64, 64, 5/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV0P0+fcV1RF4CH64'
      % FULLY CONNECTED
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 4, 64, 64, 5/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV0P0+fcV1RF2CH64'
      % FULLY CONNECTED
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 2, 64, 64, 5/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV0P0+fcV1RF1CH64'
      % FULLY CONNECTED
      layer_number = numel(net.layers) + 1;
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, 64, 5/100, 0, 'gaussian', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = numel(net.layers) + 1;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();




    % TODO: fix
    % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 4, 256, 128, 5/1000, 0, 'gaussian', 'gen');
    % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 128, 64, 5/100, 0, 'gaussian', 'gen');
    % vs
    % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 4, 64, 500, 5/100, 0, 'gaussian', 'gen');
    % net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 500, 100, 5/100, 0, 'gaussian', 'gen');
    % % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    % case 'convV0P0+fcV2RF32CH3'
    %   % FULLY CONNECTED
    %   layer_number = numel(net.layers) + 1;
    %   net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 32, 3, 500, 5/100, 0, 'gaussian', 'gen');
    %   net.layers{end+1} = fh.reluLayer(layer_number);

    %   layer_number = numel(net.layers) + 1;
    %   net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 500, 100, 5/100, 0, 'gaussian', 'gen');
    %   net.layers{end+1} = fh.reluLayer(layer_number);

    %   layer_number = numel(net.layers) + 1;
    %   number_of_output_nodes = getNumberOfOutputNodes(dataset);
    %   net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 100, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

    %   % LOSS LAYER
    %   net.layers{end+1} = fh.softmaxlossLayer();

    % % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    % case 'convV0P0+fcV2RF32CH64'
    %   % FULLY CONNECTED
    %   layer_number = numel(net.layers) + 1;
    %   net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 32, 64, 500, 5/100, 0, 'gaussian', 'gen');
    %   net.layers{end+1} = fh.reluLayer(layer_number);

    %   layer_number = numel(net.layers) + 1;
    %   net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 500, 100, 5/100, 0, 'gaussian', 'gen');
    %   net.layers{end+1} = fh.reluLayer(layer_number);

    %   layer_number = numel(net.layers) + 1;
    %   number_of_output_nodes = getNumberOfOutputNodes(dataset);
    %   net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 100, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

    %   % LOSS LAYER
    %   net.layers{end+1} = fh.softmaxlossLayer();

    % % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    % case 'convV0P0+fcV2RF16CH64'
    %   % FULLY CONNECTED
    %   layer_number = numel(net.layers) + 1;
    %   net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 16, 64, 500, 5/100, 0, 'gaussian', 'gen');
    %   net.layers{end+1} = fh.reluLayer(layer_number);

    %   layer_number = numel(net.layers) + 1;
    %   net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 500, 100, 5/100, 0, 'gaussian', 'gen');
    %   net.layers{end+1} = fh.reluLayer(layer_number);

    %   layer_number = numel(net.layers) + 1;
    %   number_of_output_nodes = getNumberOfOutputNodes(dataset);
    %   net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 100, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

    %   % LOSS LAYER
    %   net.layers{end+1} = fh.softmaxlossLayer();

    % % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    % case 'convV0P0+fcV2RF4CH64'
    %   % FULLY CONNECTED
    %   layer_number = numel(net.layers) + 1;
    %   net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 4, 64, 500, 5/100, 0, 'gaussian', 'gen');
    %   net.layers{end+1} = fh.reluLayer(layer_number);

    %   layer_number = numel(net.layers) + 1;
    %   net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 500, 100, 5/100, 0, 'gaussian', 'gen');
    %   net.layers{end+1} = fh.reluLayer(layer_number);

    %   layer_number = numel(net.layers) + 1;
    %   number_of_output_nodes = getNumberOfOutputNodes(dataset);
    %   net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 100, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

    %   % LOSS LAYER
    %   net.layers{end+1} = fh.softmaxlossLayer();

    % % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    % case 'convV0P0+fcV2RF2CH64'
    %   % FULLY CONNECTED
    %   layer_number = numel(net.layers) + 1;
    %   net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 2, 64, 500, 5/100, 0, 'gaussian', 'gen');
    %   net.layers{end+1} = fh.reluLayer(layer_number);

    %   layer_number = numel(net.layers) + 1;
    %   net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 500, 100, 5/100, 0, 'gaussian', 'gen');
    %   net.layers{end+1} = fh.reluLayer(layer_number);

    %   layer_number = numel(net.layers) + 1;
    %   number_of_output_nodes = getNumberOfOutputNodes(dataset);
    %   net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 100, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

    %   % LOSS LAYER
    %   net.layers{end+1} = fh.softmaxlossLayer();

    % % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    % case 'convV0P0+fcV2RF1CH64'
    %   % FULLY CONNECTED
    %   layer_number = numel(net.layers) + 1;
    %   net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 64, 500, 5/100, 0, 'gaussian', 'gen');
    %   net.layers{end+1} = fh.reluLayer(layer_number);

    %   layer_number = numel(net.layers) + 1;
    %   net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 500, 100, 5/100, 0, 'gaussian', 'gen');
    %   net.layers{end+1} = fh.reluLayer(layer_number);

    %   layer_number = numel(net.layers) + 1;
    %   number_of_output_nodes = getNumberOfOutputNodes(dataset);
    %   net.layers{end+1} = fh.convLayer(dataset, network_arch, layer_number, 1, 100, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

    %   % LOSS LAYER
    %   net.layers{end+1} = fh.softmaxlossLayer();

    % case 'convV0P0+fcV1'

    % case 'convV0P0+fcV2'
    % case 'convV1sP1+fcV1'
    % case 'convV1lP1+fcV1'
    % case 'convV3lP1+fcV1'
    % case 'convV3P3+fcV1'

  end

function number_of_output_nodes = getNumberOfOutputNodes(dataset)
  if isTwoClassImdb(dataset)
    number_of_output_nodes = 2;
  elseif strcmp(dataset, 'coil-100')
    number_of_output_nodes = 100;
  else
    number_of_output_nodes = 10;
  end



