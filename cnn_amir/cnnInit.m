function network_opts = cnnInit(input_opts)
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

  % -------------------------------------------------------------------------
  %                                                              Parse inputs
  % -------------------------------------------------------------------------

  opts.dataset = input_opts.general.dataset; % Used in 2 places: 1) convLayer loading weights 2) based on the dataset, networks decide how many outputs nodes in FC
  opts.network_arch = input_opts.general.network_arch;
  opts.weight_init_source = input_opts.net.weight_init_source;
  opts.weight_init_sequence = input_opts.net.weight_init_sequence;

  % -------------------------------------------------------------------------
  %                                                         Set learning rate
  % -------------------------------------------------------------------------

  tic;
  s = rng;
  rng(0);
  net.layers = {};
  if strcmp(input_opts.train.learning_rate, 'default_keyword')
    network_opts.train.learning_rate = getLearningRate(opts.dataset, opts.network_arch);
  else
    network_opts.train.learning_rate = input_opts.train.learning_rate;
  end
  network_opts.train.num_epochs = numel(network_opts.train.learning_rate);

  fh = networkInitializationUtils;
  flag = false;
  flag2 = false;
  if flag
    architecture_type = opts.network_arch(1:4);
    switch architecture_type
      case 'larp'
        net = getLarpArchitecture(opts.dataset, opts.network_arch);
      case 'conv'
        net = getConvArchitecture(opts.dataset, opts.network_arch);
      otherwise
        throwException('[ERROR] architecture type can only be `larp` or `conv`.')
    end
  % elseif flag2
  %   net.layers = {};
  %   % first construct the larp architecture (if any, bc could be no-projection)
  %   % then construct the conv / mlp architecture (and set bpd accordingly)
  elseif flag2
    net = getNetworkWithMasksArchitecture(opts.dataset, opts.network_arch, opts.weight_init_sequence);
  else
    switch opts.network_arch


      case 'larpV0P0'
        % empty

      case 'larpV1P1'
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, char(opts.weight_init_sequence{1}), 'gen');
        net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
        net.layers{end+1} = fh.reluLayer(layer_number);

      case 'larpV3P1'
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, char(opts.weight_init_sequence{1}), 'gen');
        % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
        net.layers{end+1} = fh.reluLayer(layer_number);

        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, char(opts.weight_init_sequence{2}), 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);
        net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, char(opts.weight_init_sequence{3}), 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);
        % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      case 'larpV3P3'
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, char(opts.weight_init_sequence{1}), 'gen');
        net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
        net.layers{end+1} = fh.reluLayer(layer_number);

        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, char(opts.weight_init_sequence{2}), 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);
        net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, char(opts.weight_init_sequence{3}), 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);
        net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);





























      case 'larpV0P0+convV0P0+fcV1'
        % -----------------------------------------------------------------------
        %                                                      FC LENET WITH CONV
        % -----------------------------------------------------------------------
        % LARP
        % N/A

        % CONV NET
        % N/A

        % FULLY CONNECTED
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 32, 3, 64, 5/100, 0, 'gaussian', 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);

        layer_number = numel(net.layers) + 1;
        number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

        % LOSS LAYER
        net.layers{end+1} = fh.softmaxlossLayer();
        % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --



      case 'larpV0P0+convV1P1+fcV1'
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, char(opts.weight_init_sequence{1}), 'gen');
        net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
        net.layers{end+1} = fh.reluLayer(layer_number);

        % FULLY CONNECTED
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 64, 5/100, 0, 'gaussian', 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);

        layer_number = numel(net.layers) + 1;
        number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

        % LOSS LAYER
        net.layers{end+1} = fh.softmaxlossLayer();
        % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

      case 'larpV1P1+convV0P0+fcV1'
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, char(opts.weight_init_sequence{1}), 'gen');
        net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
        net.layers{end+1} = fh.reluLayer(layer_number);

        % FULLY CONNECTED
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 64, 5/100, 0, 'gaussian', 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);

        layer_number = numel(net.layers) + 1;
        number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

        % LOSS LAYER
        net.layers{end+1} = fh.softmaxlossLayer();
        % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      case 'larpV3P1+convV0P0+fcV1'
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, char(opts.weight_init_sequence{1}), 'gen');
        % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
        net.layers{end+1} = fh.reluLayer(layer_number);

        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, char(opts.weight_init_sequence{2}), 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);
        net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, char(opts.weight_init_sequence{3}), 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);
        % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

        % FULLY CONNECTED
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 64, 5/100, 0, 'gaussian', 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);

        layer_number = numel(net.layers) + 1;
        number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

        % LOSS LAYER
        net.layers{end+1} = fh.softmaxlossLayer();
        % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      case 'larpV3P3+convV0P0+fcV1'
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, char(opts.weight_init_sequence{1}), 'gen');
        net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
        net.layers{end+1} = fh.reluLayer(layer_number);

        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, char(opts.weight_init_sequence{2}), 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);
        net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, char(opts.weight_init_sequence{3}), 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);
        net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

        % FULLY CONNECTED
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 4, 64, 64, 5/100, 0, 'gaussian', 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);

        layer_number = numel(net.layers) + 1;
        number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

        % LOSS LAYER
        net.layers{end+1} = fh.softmaxlossLayer();
        % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      % ------------------------------------------------------------------------------------------------------------------------------------------------------------
      case 'convV0P0+fcV1RF16CH64'
        % FULLY CONNECTED
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(dataset, opts.network_arch, layer_number, 16, 64, 64, 5/100, 0, 'gaussian', 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);

        layer_number = numel(net.layers) + 1;
        number_of_output_nodes = getNumberOfOutputNodes(dataset);
        net.layers{end+1} = fh.convLayer(dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

        % LOSS LAYER
        net.layers{end+1} = fh.softmaxlossLayer();
      % ------------------------------------------------------------------------------------------------------------------------------------------------------------
      case 'convV0P0+fcV1RF4CH64'
        % FULLY CONNECTED
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(dataset, opts.network_arch, layer_number, 4, 64, 64, 5/100, 0, 'gaussian', 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);

        layer_number = numel(net.layers) + 1;
        number_of_output_nodes = getNumberOfOutputNodes(dataset);
        net.layers{end+1} = fh.convLayer(dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

        % LOSS LAYER
        net.layers{end+1} = fh.softmaxlossLayer();







      case 'larpV0P0+convV0P0+fcV2'
        % -----------------------------------------------------------------------
        %                                                      FC LENET WITH CONV
        % -----------------------------------------------------------------------
        % FULLY CONNECTED
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 32, 3, 500, 5/100, 0, 'gaussian', 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);

        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 500, 100, 5/100, 0, 'gaussian', 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);

        layer_number = numel(net.layers) + 1;
        number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 100, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

        % LOSS LAYER
        net.layers{end+1} = fh.softmaxlossLayer();
        % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      case 'larpV3P1+convV0P0+fcV2'
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, char(opts.weight_init_sequence{1}), 'gen');
        % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
        net.layers{end+1} = fh.reluLayer(layer_number);

        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, char(opts.weight_init_sequence{2}), 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);
        net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, char(opts.weight_init_sequence{3}), 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);
        % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

        % FULLY CONNECTED
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 500, 5/100, 0, 'gaussian', 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);

        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 500, 100, 5/100, 0, 'gaussian', 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);

        layer_number = numel(net.layers) + 1;
        number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 100, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

        % LOSS LAYER
        net.layers{end+1} = fh.softmaxlossLayer();
        % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      % ------------------------------------------------------------------------------------------------------------------------------------------------------------




      % TODO: remove
      case 'TESTINGlarpV1P0+convV0P0+fcV1'
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 3, 1, 1/100, 1, 'testing', 'gen');
        % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
        % net.layers{end+1} = fh.reluLayer(layer_number);

        % layer_number = numel(net.layers) + 1;
        % net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 1, 1, 1/100, 1, 'testing', 'gen');
        % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
        % net.layers{end+1} = fh.reluLayer(layer_number);

        % FULLY CONNECTED
        layer_number = numel(net.layers) + 1;
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 1, 64, 5/100, 0, 'gaussian', 'gen');
        net.layers{end+1} = fh.reluLayer(layer_number);

        layer_number = numel(net.layers) + 1;
        number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
        net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'gaussian', 'gen');

        % LOSS LAYER
        net.layers{end+1} = fh.softmaxlossLayer();
        % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    end
  end

  % -------------------------------------------------------------------------
  %    VERY IMPORTANT: reset this afterwards so other modules are true random
  % -------------------------------------------------------------------------
  rng(s);
  network_opts.net = net;

function number_of_output_nodes = getNumberOfOutputNodes(dataset)
  if isTwoClassImdb(dataset)
    number_of_output_nodes = 2;
  elseif strcmp(dataset, 'coil-100')
    number_of_output_nodes = 100;
  else
    number_of_output_nodes = 10;
  end










