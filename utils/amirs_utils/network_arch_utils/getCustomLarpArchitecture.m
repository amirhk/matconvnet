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

  fh = networkInitializationUtils;
  net.layers = {};

  assert(logical(strfind(network_arch, 'custom-')), 'this file is only to be used to construct `custom` larp architectures');

  % example: network_arch = 'custom-5-L-3-256-relu-max-pool'
  %          5 blocks layers, each containing:
  %            larp w/ 3 x 3 x input x 256        % can vary these further later, atm every block has the same larp architecture... can create fattening and narrowing archs later
  %            relu non-lin                       % can vary these further later, type, ...         also, note that there is 1 of these per EVERY larp layer
  %            max-pooling (poolingLayerLeNetMax) % can vary these further later, type, stride, ... also, note that there is 1 of these per EVERY larp layer

  number_of_blocks = str2num(getStringParameterStartingAtIndex(network_arch, 8));
  larp_layer_kernel_width = str2num(getStringParameterStartingAtIndex(network_arch, 12)); assert(mod(larp_layer_kernel_width, 2) == 1);
  larp_layer_kernel_count = str2num(getStringParameterStartingAtIndex(network_arch, 14));

  should_add_relu_per_block = false;
  should_add_max_pooling_per_block = false;
  if strfind(network_arch, 'relu')
    should_add_relu_per_block = true;
  end
  if strfind(network_arch, 'max-pool')
    should_add_max_pooling_per_block = true;
  end

  previous_layer_feature_map_channel_count = 3; % input RGB
  for i = 1 : number_of_blocks
    layer_number = numel(net.layers) + 1;
    padding = (larp_layer_kernel_width - 1) / 2; % to retain size

    net.layers{end+1} = fh.convLayer( ...
      dataset, ...
      network_arch, ...
      layer_number, ...
      larp_layer_kernel_width, ...
      previous_layer_feature_map_channel_count, ...
      larp_layer_kernel_count, ...
      1/100, ...
      padding, ...
      char(weight_init_sequence{i}), ...
      'gen');

    previous_layer_feature_map_channel_count = larp_layer_kernel_count;

    if should_add_relu_per_block
      net.layers{end+1} = fh.reluLayer(layer_number);
    end
    if should_add_max_pooling_per_block
      net.layers{end+1} = fh.poolingLayerLeNetMax(layer_number);
    end
  end



% -------------------------------------------------------------------------
function string_parameter = getStringParameterStartingAtIndex(input_string, start_index)
% -------------------------------------------------------------------------
  delimeter = '-';
  string_parameter = input_string(start_index : start_index + strfind(input_string(start_index:end), delimeter) - 2);
