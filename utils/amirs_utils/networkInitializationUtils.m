% -------------------------------------------------------------------------
function fh = networkInitializationUtils()
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

  % assign function handles so we can call these local functions from elsewhere
  fh.convLayer = @convLayer;
  fh.loadWeights = @loadWeights;
  fh.constructConvLayer = @constructConvLayer;
  fh.reluLayer = @reluLayer;
  fh.tanhLayer = @tanhLayer;
  fh.poolingLayer = @poolingLayer;
  fh.poolingLayerAlexNet = @poolingLayerAlexNet;
  fh.poolingLayerLeNetAvg = @poolingLayerLeNetAvg;
  fh.poolingLayerLeNetMax = @poolingLayerLeNetMax;
  fh.dropoutLayer = @dropoutLayer;
  fh.bnormLayer = @bnormLayer;
  fh.softmaxlossLayer = @softmaxlossLayer;

  % fh.getConvLayerIndices = @poolingLayer;
  % fh.createBottleneckLayersFromSingleConvLayer = @poolingLayer;

% --------------------------------------------------------------------
function structuredLayer = convLayer(dataset, network_arch, layer_number, k, m, n, init_multiplier, pad, weight_init_type, weight_init_source);
% --------------------------------------------------------------------
  switch weight_init_source
    case 'load'
      layerWeights = loadWeights(dataset, network_arch, layer_number, weight_init_type);
    case 'gen'
      switch weight_init_type
        case 'compRand'
          layerWeights{1} = init_multiplier * randn(k, k, m, n, 'single');
          layerWeights{2} = zeros(1, n, 'single');
        case 'quasiRandSobol'
          q = sobolset(1);
          q = scramble(q, 'MatousekAffineOwen');
          layerWeights{1} = single(init_multiplier * reshape(net(q, k * k * m * n), [k, k, m, n]));
          layerWeights{2} = zeros(1, n, 'single');
        case 'quasiRandSobolSkip'
          q = sobolset(1);
          q = scramble(q, 'MatousekAffineOwen');
          layerWeights{1} = single(init_multiplier * reshape(net(q, k * k * m * n), [k, k, m, n]));
          layerWeights{2} = zeros(1, n, 'single');
        otherwise
          throwException('[ERROR] Generating non-compRand weights not supported from this code.');
      end
  end
  structuredLayer = constructConvLayer(network_arch, layer_number, layerWeights, pad, weight_init_type, weight_init_source);

% --------------------------------------------------------------------
function weights = loadWeights(dataset, network_arch, layer_number, weight_init_type)
% --------------------------------------------------------------------
  fprintf( ...
    '[INFO] Loading %s weights (layer %d) from saved directory...\t', ...
    weight_init_type, ...
    layer_number);
  dev_path = getDevPath();

  % sub_dir_path = fullfile('data', 'cifar-alexnet', sprintf('w_%s', weight_init_type));
  % TODO: search subtstring... if network_arch starts with 'alexnet' use the 'alexnet' folder
  sub_dir_path = fullfile( ...
    'data', ...
    'generated_weights', ...
    sprintf('%s', network_arch), ...
    sprintf('w-%s', weight_init_type));
  file_name_suffix = sprintf('-layer-%d.mat', layer_number);
  tmp = load(fullfile(dev_path, sub_dir_path, sprintf('W1%s', file_name_suffix)));
  weights{1} = tmp.W1;
  tmp = load(fullfile(dev_path, sub_dir_path, sprintf('W2%s', file_name_suffix)));
  weights{2} = tmp.W2;
  fprintf('Done!\n');

% --------------------------------------------------------------------
function structuredLayer = constructConvLayer(network_arch, layer_number, weights, pad, weight_init_type, weight_init_source)
% --------------------------------------------------------------------
  lr = [.1 2];
  if strcmp(network_arch, 'alexnet') && layer_number == 18
    lr = lr * .1;
  elseif strcmp(network_arch, 'lenet') && layer_number == 12
  end
  structuredLayer = struct( ...
    'type', 'conv', ...
    'name', sprintf('conv%s-%s-%s', layer_number, weight_init_type, weight_init_source), ...
    'weights', {weights}, ...
    'learning_rate', lr, ...
    'stride', 1, ...
    'pad', pad);

% --------------------------------------------------------------------
function structuredLayer = reluLayer(layer_number)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'relu', ...
    'name', sprintf('relu%s', layer_number));

% --------------------------------------------------------------------
function structuredLayer = tanhLayer(layer_number)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'tanh', ...
    'name', sprintf('tanh%s', layer_number));

% --------------------------------------------------------------------
function structuredLayer = poolingLayer(layer_number)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'pool', ...
    'name', sprintf('pool%s', layer_number), ...
    'method', 'max', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad', 0); % Emulate caffe_n

% --------------------------------------------------------------------
function structuredLayer = poolingLayerAlexNet(layer_number)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'pool', ...
    'name', sprintf('pool%s', layer_number), ...
    'method', 'max', ...
    'pool', [3 3], ...
    'stride', 2, ...
    'pad', [0 1 0 1]); % Emulate caffe_n

% --------------------------------------------------------------------
function structuredLayer = poolingLayerLeNetAvg(layer_number)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'pool', ...
    'name', sprintf('pool%s', layer_number), ...
    'method', 'avg', ...
    'pool', [3 3], ...
    'stride', 2, ...
    'pad', [0 1 0 1]); % Emulate caffe_n

% --------------------------------------------------------------------
function structuredLayer = poolingLayerLeNetMax(layer_number)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'pool', ...
    'name', sprintf('pool%s', layer_number), ...
    'method', 'max', ...
    'pool', [3 3], ...
    'stride', 2, ...
    'pad', [0 1 0 1]); % Emulate caffe_n

% --------------------------------------------------------------------
function structuredLayer = dropoutLayer(layer_number, dropout_ratio)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'dropout', ...
    'name', sprintf('dropout%s', layer_number), ...
    'rate', dropout_ratio);

% --------------------------------------------------------------------
function structuredLayer = bnormLayer(layer_number, ndim)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'bnorm', ...
    'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
    'learningRate', [1 1], ...
    'weightDecay', [0 0]);

% --------------------------------------------------------------------
function structuredLayer = softmaxlossLayer()
% --------------------------------------------------------------------
  structuredLayer = struct('type', 'softmaxloss');


% % --------------------------------------------------------------------
% function conv_layer_indices = getConvLayerIndices(net)
% % --------------------------------------------------------------------
%   conv_layer_indices = [];
%   for i = 1:numel(net.layers)
%     if strcmp(net.layers{i}.type, 'conv')
%       conv_layer_indices(end + 1) = i;
%     end
%   end

% % --------------------------------------------------------------------
% function [pre_layer, post_layer] = createBottleneckLayersFromSingleConvLayer(opts, layer_object, bottleneck_size)
% % --------------------------------------------------------------------
%   layer_number = 0; % not important
%   conv_layer_sizes = size(layer_object.weights{1});
%   filter_size = conv_layer_sizes(1); % conv_layer_sizes(2);
%   input_size = conv_layer_sizes(3);
%   output_size = conv_layer_sizes(4);
%   pad = layer_object.pad;
%   pre_layer = convLayer(opts.dataset, opts.network_arch, layer_number, filter_size, input_size, bottleneck_size, 5/100, pad, 'compRand', 'gen');
%   post_layer = convLayer(opts.dataset, opts.network_arch, layer_number, filter_size, bottleneck_size, output_size, 5/100, pad, 'compRand', 'gen');
