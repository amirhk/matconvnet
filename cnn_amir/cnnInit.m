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
  % flag2 = true;
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
  else
  switch opts.network_arch
    case 'larpV0P0+convV0P0+fcV1'
      % -----------------------------------------------------------------------
      %                                                      FC LENET WITH CONV
      % -----------------------------------------------------------------------
      % LARP
      % N/A

      % CONV NET
      % N/A

      % FULLY CONNECTED
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 32, 3, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV3P1+convV0P0+fcV1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, char(opts.weight_init_sequence{1}), 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, char(opts.weight_init_sequence{2}), 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, char(opts.weight_init_sequence{3}), 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV0P0+fcV1RF16CH64'
      % FULLY CONNECTED
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(dataset, opts.network_arch, layer_number, 16, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
    % ------------------------------------------------------------------------------------------------------------------------------------------------------------
    case 'convV0P0+fcV1RF4CH64'
      % FULLY CONNECTED
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(dataset, opts.network_arch, layer_number, 4, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(dataset);
      net.layers{end+1} = fh.convLayer(dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();







    case 'larpV0P0+convV0P0+fcV2'
      % -----------------------------------------------------------------------
      %                                                      FC LENET WITH CONV
      % -----------------------------------------------------------------------
      % FULLY CONNECTED
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 32, 3, 500, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 500, 100, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 100, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV3P1+convV0P0+fcV2'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, char(opts.weight_init_sequence{1}), 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, char(opts.weight_init_sequence{2}), 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, char(opts.weight_init_sequence{3}), 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 500, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 500, 100, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 100, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    % ------------------------------------------------------------------------------------------------------------------------------------------------------------




    % TODO: remove
    case 'TESTINGlarpV1P0+convV0P0+fcV1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 3, 1, 1/100, 1, 'testing', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      % net.layers{end+1} = fh.reluLayer(layer_number);

      % layer_number = layer_number + 1;
      % net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 1, 1, 1/100, 1, 'testing', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      % net.layers{end+1} = fh.reluLayer(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 1, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

















    % case 'lenet_bu'
    %   layer_number = 1;
    %   net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, char(opts.weight_init_sequence{1}), opts.weight_init_source);
    %   net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
    %   net.layers{end+1} = fh.reluLayer(layer_number);

    %   layer_number = layer_number + 3;
    %   net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, char(opts.weight_init_sequence{2}), opts.weight_init_source);
    %   net.layers{end+1} = fh.reluLayer(layer_number);
    %   net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

    %   layer_number = layer_number + 3;
    %   net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, char(opts.weight_init_sequence{3}), opts.weight_init_source);
    %   net.layers{end+1} = fh.reluLayer(layer_number);
    %   net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

    %   % FULLY CONNECTED
    %   layer_number = layer_number + 3;
    %   net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 4, 64, 64, 5/100, 0, 'compRand', 'gen');
    %   net.layers{end+1} = fh.reluLayer(layer_number);

    %   layer_number = layer_number + 2;
    %   if isTwoClassImdb(opts.dataset)
    %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 2, 5/100, 0, 'compRand', 'gen');
    %   elseif strcmp(opts.dataset, 'coil-100')
    %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 100, 5/100, 0, 'compRand', 'gen');
    %   else
    %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 10, 5/100, 0, 'compRand', 'gen');
    %   end

    %   % LOSS LAYER
    %   net.layers{end+1} = fh.softmaxlossLayer();


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
    case 'larpV0P0SF+convV0P0+fcV1'
      % -----------------------------------------------------------------------
      %                                                      FC LENET WITH CONV
      % -----------------------------------------------------------------------
      % LARP
      % N/A

      % CONV NET
      % N/A

      % FULLY CONNECTED
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 32, 3, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV1P0SF+convV0P0+fcV1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 32, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV1P1SF+convV0P0+fcV1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV3P0SF+convV0P0+fcV1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 32, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV3P1SF+convV0P0+fcV1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV3P3SF+convV0P0+fcV1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 4, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV5hP0SF+convV0P0+fcV1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 32, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV5hP1SF+convV0P0+fcV1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV5hP3SF+convV0P0+fcV1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 4, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV5hP5SF+convV0P0+fcV1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV5aP0SF+convV0P0+fcV1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 96, 5/1000, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 96, 256, 5/1000, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 256, 384, 5/1000, 1, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384, 384, 5/1000, 1, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384, 64, 5/1000, 1, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 32, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV5aP1SF+convV0P0+fcV1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 96, 5/1000, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 96, 256, 5/1000, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 256, 384, 5/1000, 1, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384, 384, 5/1000, 1, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384, 64, 5/1000, 1, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV5aP3SF+convV0P0+fcV1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 96, 5/1000, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 96, 256, 5/1000, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 256, 384, 5/1000, 1, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384, 384, 5/1000, 1, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384, 64, 5/1000, 1, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 4, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV5aP5SF+convV0P0+fcV1'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 96, 5/1000, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 96, 256, 5/1000, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 256, 384, 5/1000, 1, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384, 384, 5/1000, 1, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384, 64, 5/1000, 1, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%


    case 'larpV0P0SF+convV0P0+fcV2'
      % -----------------------------------------------------------------------
      %                                                      FC LENET WITH CONV
      % -----------------------------------------------------------------------
      % FULLY CONNECTED
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 32, 3, 500, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 500, 100, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 100, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV1P0SF+convV0P0+fcV2'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 32, 64, 500, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 500, 100, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 100, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV1P1SF+convV0P0+fcV2'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 500, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 500, 100, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 100, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV3P0SF+convV0P0+fcV2'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 32, 64, 500, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 500, 100, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 100, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV3P1SF+convV0P0+fcV2'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 500, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 500, 100, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 100, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV3P3SF+convV0P0+fcV2'
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 3;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 4, 64, 500, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 500, 100, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 100, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%


    case 'larpV0sP0SF+convV1sP1+fcV1'
      % LARP
      % N/A

      % CONV NET
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV1sP0SF+convV1sP1+fcV1'
      % LARP
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 3, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % CONV NET
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV2sP0SF+convV1sP1+fcV1'
      % LARP
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 3, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 3, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % CONV NET
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV1lP0SF+convV1lP1+fcV1'
      % LARP
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % CONV NET
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV2lP0SF+convV1lP1+fcV1'
      % LARP
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % CONV NET
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%


    case 'larpV0P0SF+convV3lP1+fcV1'
      % LARP
      % N/A

      % CONV NET
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV1lP0SF+convV3lP1+fcV1'
      % LARP
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % CONV NET
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV2lP0SF+convV3lP1+fcV1'
      % LARP
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 64, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % CONV NET
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 64, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 16, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%


    case 'larpV0P0SF+convV3P3+fcV1'
      % LARP
      % N/A

      % CONV NET
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 4, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV1lP0SF+convV3P3+fcV1'
      % LARP
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % CONV NET
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 4, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV1lP1SF+convV3P3+fcV1'
      % LARP
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % CONV NET
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 2, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV2lP0SF+convV3P3+fcV1'
      % LARP
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % CONV NET
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 4, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV2lP1SF+convV3P3+fcV1'
      % LARP
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % CONV NET
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 2, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    case 'larpV2lP2SF+convV3P3+fcV1'
      % LARP
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      % CONV NET
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 1/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);
      net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % FULLY CONNECTED
      layer_number = layer_number + 2;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = fh.reluLayer(layer_number);

      layer_number = layer_number + 2;
      number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % LOSS LAYER
      net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%


    case 'TMP_NETWORK'
      % LARP
      layer_number = 1;
      net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 1/100, 2, 'compRand', 'gen');
      % % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      % net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 1/100, 2, 'compRand', 'gen');
      % % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      % net.layers{end+1} = fh.reluLayer(layer_number);

      % layer_number = 1;
      % net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
      % net.layers{end+1} = fh.reluLayer(layer_number);

      % layer_number = layer_number + 3;
      % net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % layer_number = layer_number + 3;
      % net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, 'compRand', 'gen');
      % net.layers{end+1} = fh.reluLayer(layer_number);
      % net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

      % CONV NET
      % N/A

      % % FULLY CONNECTED
      % layer_number = layer_number + 3;
      % net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 4, 64, 64, 5/100, 0, 'compRand', 'gen');
      % net.layers{end+1} = fh.reluLayer(layer_number);

      % layer_number = layer_number + 2;
      % number_of_output_nodes = getNumberOfOutputNodes(opts.dataset);
      % net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, number_of_output_nodes, 5/100, 0, 'compRand', 'gen');

      % % LOSS LAYER
      % net.layers{end+1} = fh.softmaxlossLayer();
      % -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%



  %   case 'lenet_with_larger_fc_conv'
  %     % -----------------------------------------------------------------------
  %     %                                               lenet_with_larger_fc_conv
  %     % -----------------------------------------------------------------------
  %     layer_number = 1;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, char(opts.weight_init_sequence{1}), opts.weight_init_source);
  %     net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
  %     net.layers{end+1} = fh.reluLayer(layer_number);

  %     layer_number = layer_number + 3;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, char(opts.weight_init_sequence{2}), opts.weight_init_source);
  %     net.layers{end+1} = fh.reluLayer(layer_number);
  %     net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

  %     layer_number = layer_number + 3;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, char(opts.weight_init_sequence{3}), opts.weight_init_source);
  %     net.layers{end+1} = fh.reluLayer(layer_number);
  %     net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

  %     % FULLY CONNECTED
  %     layer_number = layer_number + 3;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 4, 64, 500, 5/100, 0, 'compRand', 'gen');
  %     net.layers{end+1} = fh.reluLayer(layer_number);

  %     layer_number = layer_number + 2;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 500, 100, 5/100, 0, 'compRand', 'gen');
  %     net.layers{end+1} = fh.reluLayer(layer_number);

  %     layer_number = layer_number + 2;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 100, 10, 5/100, 0, 'compRand', 'gen');
  %     % LOSS LAYER
  %     net.layers{end+1} = fh.softmaxlossLayer();
  %   case 'fc_lenet_with_larger_fc_conv'
  %     % -----------------------------------------------------------------------
  %     %                                            fc_lenet_with_larger_fc_conv
  %     % -----------------------------------------------------------------------
  %     % FULLY CONNECTED
  %     layer_number = 1;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 32, 3, 500, 5/100, 0, 'compRand', 'gen');
  %     net.layers{end+1} = fh.reluLayer(layer_number);

  %     layer_number = layer_number + 2;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 500, 100, 5/100, 0, 'compRand', 'gen');
  %     net.layers{end+1} = fh.reluLayer(layer_number);

  %     layer_number = layer_number + 2;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 100, 10, 5/100, 0, 'compRand', 'gen');
  %     % LOSS LAYER
  %     net.layers{end+1} = fh.softmaxlossLayer();
  %   case 'lenet+1'
  %     % -----------------------------------------------------------------------
  %     %                                                               LENET + 1
  %     % -----------------------------------------------------------------------
  %     layer_number = 1;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 3, 1/100, 2, char(opts.weight_init_sequence{1}), opts.weight_init_source);
  %     % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
  %     net.layers{end+1} = fh.reluLayer(layer_number);

  %     layer_number = layer_number + 3;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 5/100, 2, char(opts.weight_init_sequence{2}), opts.weight_init_source);
  %     net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
  %     net.layers{end+1} = fh.reluLayer(layer_number);

  %     layer_number = layer_number + 3;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, char(opts.weight_init_sequence{2}), opts.weight_init_source);
  %     net.layers{end+1} = fh.reluLayer(layer_number);
  %     net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

  %     layer_number = layer_number + 3;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, char(opts.weight_init_sequence{3}), opts.weight_init_source);
  %     net.layers{end+1} = fh.reluLayer(layer_number);
  %     net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

  %     % FULLY CONNECTED
  %     layer_number = layer_number + 3;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 4, 64, 64, 5/100, 0, 'compRand', 'gen');
  %     net.layers{end+1} = fh.reluLayer(layer_number);

  %     layer_number = layer_number + 2;
  %     if isTwoClassImdb(opts.dataset)
  %       net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 2, 5/100, 0, 'compRand', 'gen');
  %     elseif strcmp(opts.dataset, 'coil-100')
  %       net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 100, 5/100, 0, 'compRand', 'gen');
  %     else
  %       net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 10, 5/100, 0, 'compRand', 'gen');
  %     end

  %     % LOSS LAYER
  %     net.layers{end+1} = fh.softmaxlossLayer();
  %   case 'lenet++1'
  %     % -----------------------------------------------------------------------
  %     %                                                               LENET + 1
  %     % -----------------------------------------------------------------------
  %     layer_number = 1;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, char(opts.weight_init_sequence{1}), opts.weight_init_source);
  %     % net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
  %     net.layers{end+1} = fh.reluLayer(layer_number);

  %     layer_number = layer_number + 3;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, char(opts.weight_init_sequence{2}), opts.weight_init_source);
  %     net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
  %     net.layers{end+1} = fh.reluLayer(layer_number);

  %     layer_number = layer_number + 3;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, char(opts.weight_init_sequence{2}), opts.weight_init_source);
  %     net.layers{end+1} = fh.reluLayer(layer_number);
  %     net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

  %     layer_number = layer_number + 3;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, char(opts.weight_init_sequence{3}), opts.weight_init_source);
  %     net.layers{end+1} = fh.reluLayer(layer_number);
  %     net.layers{end+1} = fh.poolingLayerLeNetAvg(layer_number);

  %     % FULLY CONNECTED
  %     layer_number = layer_number + 3;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 4, 64, 64, 5/100, 0, 'compRand', 'gen');
  %     net.layers{end+1} = fh.reluLayer(layer_number);

  %     layer_number = layer_number + 2;
  %     if isTwoClassImdb(opts.dataset)
  %       net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 2, 5/100, 0, 'compRand', 'gen');
  %     elseif strcmp(opts.dataset, 'coil-100')
  %       net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 100, 5/100, 0, 'compRand', 'gen');
  %     else
  %       net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 10, 5/100, 0, 'compRand', 'gen');
  %     end

  %     % LOSS LAYER
  %     net.layers{end+1} = fh.softmaxlossLayer();
  %   case 'alexnet'
  %     % -----------------------------------------------------------------------
  %     %                                                                 ALEXNET
  %     % -----------------------------------------------------------------------
  %     layer_number = 1;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 96, 5/1000, 2, char(opts.weight_init_sequence{1}), opts.weight_init_source);
  %     net.layers{end+1} = fh.reluLayer(layer_number);

  %     layer_number = layer_number + 2;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 5, 96, 256, 5/1000, 2, char(opts.weight_init_sequence{2}), opts.weight_init_source);
  %     net.layers{end+1} = fh.reluLayer(layer_number);
  %     net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

  %     layer_number = layer_number + 3;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 256, 384, 5/1000, 1, char(opts.weight_init_sequence{3}), opts.weight_init_source);
  %     net.layers{end+1} = fh.reluLayer(layer_number);
  %     net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

  %     layer_number = layer_number + 3;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384, 384, 5/1000, 1, char(opts.weight_init_sequence{4}), opts.weight_init_source);
  %     net.layers{end+1} = fh.reluLayer(layer_number);

  %     layer_number = layer_number + 2;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384, 256, 5/1000, 1, char(opts.weight_init_sequence{5}), opts.weight_init_source);
  %     net.layers{end+1} = fh.reluLayer(layer_number);
  %     net.layers{end+1} = fh.poolingLayerAlexNet(layer_number);

  %     % FULLY CONNECTED
  %     layer_number = layer_number + 3;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 4, 256, 128, 5/1000, 0, 'compRand', 'gen');
  %     net.layers{end+1} = fh.reluLayer(layer_number);

  %     layer_number = layer_number + 2;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 128, 64, 5/100, 0, 'compRand', 'gen');
  %     net.layers{end+1} = fh.reluLayer(layer_number);

  %     layer_number = layer_number + 2;
  %     net.layers{end+1} = fh.convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 10, 5/100, 0, 'compRand', 'gen');
  %     net.layers{end+1} = fh.reluLayer(layer_number);

  %     % LOSS LAYER
  %     net.layers{end+1} = fh.softmaxlossLayer();
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










