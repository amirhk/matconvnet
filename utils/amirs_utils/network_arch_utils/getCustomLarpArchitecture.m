% -------------------------------------------------------------------------
function net = getCustomLarpArchitecture(dataset, network_arch, weight_init_sequence)
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

  net.layers = {};

  assert(logical(strfind(network_arch, 'custom-')), 'this file is only to be used to construct `custom` larp architectures');

  % example: network_arch = 'custom-5-L-3-256-relu-max-pool'
  %          5 blocks layers, each containing:
  %            larp w/ 3 x 3 x input x 256        % can vary these further later, atm every block has the same larp architecture... can create fattening and narrowing archs later
  %            relu non-lin                       % can vary these further later, type, ...         also, note that there is 1 of these per EVERY larp layer
  %            max-pooling (poolingLayerLeNetMax) % can vary these further later, type, stride, ... also, note that there is 1 of these per EVERY larp layer

  number_of_blocks = str2num(getStringParameterStartingAtIndex(network_arch, 8));
  larp_layer_kernel_width = str2num(getStringParameterStartingAtIndex(network_arch, 12)); assert(mod(larp_layer_kernel_width, 2) == 1);
  larp_layer_kernel_count = str2num(getStringParameterStartingAtIndex(network_arch, 12 + length(num2str(larp_layer_kernel_width)) + 1)); % what a hack, smh
  final_larp_layer_kernel_count = str2num(getStringParameterStartingAtIndex(network_arch, 12 + length(num2str(larp_layer_kernel_width)) + length(num2str(larp_layer_kernel_count)) + 2)); % what a hack, smh

  if strfind(dataset, 'mnist-784')
    previous_layer_feature_map_channel_count = 1; % input BW
  else
    previous_layer_feature_map_channel_count = 3; % input RGB
  end

  for block_number = 1 : number_of_blocks - 1 % -1 see below... the final block is assigned differentlty

    current_layer_kernel_count = larp_layer_kernel_count;
    tmp = addBlockLayerElements( ...
      block_number, ...
      dataset, ...
      network_arch, ...
      larp_layer_kernel_width, ...
      previous_layer_feature_map_channel_count, ...
      current_layer_kernel_count, ...
      weight_init_sequence, ...
      number_of_blocks);
    net.layers = cat(2, net.layers, tmp.layers);

    previous_layer_feature_map_channel_count = larp_layer_kernel_count;
  end

  % we want the last block to have fewer kernels so the output dimension is small!
  % final_larp_layer_kernel_count = 64;
  % keyboard
  % if block_number ~= 1
  %   block_number = block_number + 1;
  %   assert(block_number == number_of_blocks);
  % else
  %   block_number = 1;
  % end

  if numel(block_number) == 0 % the network is single layer, and so the loop above never happened, hence block_number = []
    block_number = 1;
  else
    block_number = block_number + 1;
    assert(block_number == number_of_blocks);
  end

  current_layer_kernel_count = final_larp_layer_kernel_count;
  tmp = addBlockLayerElements( ...
    block_number, ...
    dataset, ...
    network_arch, ...
    larp_layer_kernel_width, ...
    previous_layer_feature_map_channel_count, ...
    current_layer_kernel_count, ...
    weight_init_sequence, ...
    number_of_blocks);
  net.layers = cat(2, net.layers, tmp.layers);


% -------------------------------------------------------------------------
function tmp_net = addBlockLayerElements(block_number, dataset, network_arch, larp_layer_kernel_width, previous_layer_feature_map_channel_count, current_layer_kernel_count, weight_init_sequence, number_of_blocks)
% -------------------------------------------------------------------------
  should_add_relu_per_block = false;
  should_add_max_pooling_per_block = false;
  if strfind(network_arch, 'relu')
    should_add_relu_per_block = true;
  end
  if strfind(network_arch, 'max-pool')
    should_add_max_pooling_per_block = true;
  end

  tmp_net.layers = {};

  % if block_number == 1 || block_number == 2
  %   larp_layer_kernel_width = 5;
  %   padding = (larp_layer_kernel_width - 1) / 2; % to retain size
  % else
  %   larp_layer_kernel_width = 3;
  %   padding = (larp_layer_kernel_width - 1) / 2; % to retain size
  % end
  padding = (larp_layer_kernel_width - 1) / 2; % to retain size

  fh = networkInitializationUtils;
  tmp_net.layers{end+1} = fh.convLayer( ...
    dataset, ...
    network_arch, ...
    block_number, ...
    larp_layer_kernel_width, ...
    previous_layer_feature_map_channel_count, ...
    current_layer_kernel_count, ...
    1/100, ...
    padding, ...
    char(weight_init_sequence{block_number}), ...
    'gen');

  if should_add_relu_per_block
      tmp_net.layers{end+1} = fh.reluLayer(block_number);
  end
  if should_add_max_pooling_per_block
    % if block_number ~= number_of_blocks
    % % if block_number == 1 || block_number == 2 || block_number == 3
    % % if block_number == 2 || block_number == 3 || block_number == 5
    %   % tmp_net.layers{end+1} = fh.poolingLayerAlexNet(block_number);
    %   tmp_net.layers{end+1} = fh.poolingLayerLeNetMax(block_number);
    %   % tmp_net.layers{end+1} = fh.poolingLayerLeNetAvg(block_number);
    % else
    %   % continue
    %   % tmp_net.layers{end+1} = fh.poolingLayerLeNetMax(block_number);
    %   % tmp_net.layers{end+1} = fh.poolingLayerLeNetAvg(block_number);
    % end
    tmp_net.layers{end+1} = fh.poolingLayerLeNetMax(block_number);
    % tmp_net.layers{end+1} = fh.poolingLayerLeNetAvg(block_number);
  end


