function main_cnn_amir(varargin)
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

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                   -- ==
% -- ==                        NETWORK ARCH                               -- ==
% -- ==                                                                   -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
  % dataset_list = {'cifar', 'stl-10', 'coil-100'}; % {'mnist', 'cifar', 'stl-10', 'coil-100'}
  % dataset_list = {'cifar', 'coil-100', 'mnist', 'stl-10'};
  dataset_list = {'mnist-two-class-9-4'};

  % network_arch = 'mnistnet';
  % % backprop_depth_list = [8, 6, 4];
  % backprop_depth_list = [4];

  % network_arch = 'prostatenet';
  % % backprop_depth_list = [13, 10, 7, 4];
  % % backprop_depth_list = [13, 4];
  % backprop_depth_list = [4];
  % % leave_out_type = 'sample';
  % leave_out_indices = 1:1:266;
  leave_out_type = 'patient';
  % leave_out_indices = 1:1:104;
  leave_out_indices = 1:1:1;


  network_arch = 'lenet';
  % backprop_depth_list = [13, 10, 7, 4]; % no dropout
  % backprop_depth_list = [14, 10, 7, 4]; % 1 x dropout after 1st layer
  % backprop_depth_list = [14, 11, 8, 4]; % 1 x dropout after 3rd layer
  % backprop_depth_list = [14, 11, 8, 5]; % 1 x dropout in FC
  % backprop_depth_list = [15, 11, 8, 4]; % 2 x dropout after 1st and 3rd layers
  backprop_depth_list = [13];
  % backprop_depth_list = [4];

  % network_arch = 'lenet';
  % backprop_depth_list = [13, 10, 7, 4];
  % backprop_depth_list = [4];

  % network_arch = 'alexnet';
  % % backprop_depth_list = [20, 18, 15, 12, 10, 7];
  % backprop_depth_list = [20];

  % network_arch = 'alexnet-bnorm';
  % % backprop_depth_list = [20, 18, 15, 12, 10, 7];
  % backprop_depth_list = [22];

  % network_arch = 'alexnet-bottleneck';
  % backprop_depth_list = [21];
  % bottleneck_divide_by_list = [1,2,4,8,16,32];
  bottleneck_divide_by_list = [1];

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                   -- ==
% -- ==                          MORE PARAMS                              -- ==
% -- ==                                                                   -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==

  weight_init_source = 'gen';  % {'load' | 'gen'}
  weight_init_sequence_list = {{'compRand', 'compRand', 'compRand'}};

  % weight_init_source = 'load';  % {'load' | 'gen'}
  % % weight_init_sequence_list = {{'compRand', 'layerwise-1D-from-cifar', 'layerwise-1D-from-cifar'}};
  % weight_init_sequence_list = { ...
  %   ... % {'1-clustered-layerwise-1D-from-cifar', '1-clustered-layerwise-1D-from-cifar', '1-clustered-layerwise-1D-from-cifar'}, ...
  %   ... % {'1-clustered-layerwise-1D-from-coil-100', '1-clustered-layerwise-1D-from-coil-100', '1-clustered-layerwise-1D-from-coil-100'}, ...
  %   ... % {'1-clustered-layerwise-1D-from-mnist', '1-clustered-layerwise-1D-from-mnist', '1-clustered-layerwise-1D-from-mnist'}, ...
  %   ... % {'1-clustered-layerwise-1D-from-stl-10', '1-clustered-layerwise-1D-from-stl-10', '1-clustered-layerwise-1D-from-stl-10'}, ...
  %   ... % {'2-clustered-layerwise-1D-from-cifar', '2-clustered-layerwise-1D-from-cifar', '2-clustered-layerwise-1D-from-cifar'}, ...
  %   ... % {'2-clustered-layerwise-1D-from-coil-100', '2-clustered-layerwise-1D-from-coil-100', '2-clustered-layerwise-1D-from-coil-100'}, ...
  %   ... % {'2-clustered-layerwise-1D-from-mnist', '2-clustered-layerwise-1D-from-mnist', '2-clustered-layerwise-1D-from-mnist'}, ...
  %   ... % {'2-clustered-layerwise-1D-from-stl-10', '2-clustered-layerwise-1D-from-stl-10', '2-clustered-layerwise-1D-from-stl-10'}, ...
  %   ... % {'4-clustered-layerwise-1D-from-cifar', '4-clustered-layerwise-1D-from-cifar', '4-clustered-layerwise-1D-from-cifar'}, ...
  %   ... % {'4-clustered-layerwise-1D-from-coil-100', '4-clustered-layerwise-1D-from-coil-100', '4-clustered-layerwise-1D-from-coil-100'}, ...
  %   ... % {'4-clustered-layerwise-1D-from-mnist', '4-clustered-layerwise-1D-from-mnist', '4-clustered-layerwise-1D-from-mnist'}, ...
  %   ... % {'4-clustered-layerwise-1D-from-stl-10', '4-clustered-layerwise-1D-from-stl-10', '4-clustered-layerwise-1D-from-stl-10'}, ...
  %   ... % {'8-clustered-layerwise-1D-from-cifar', '8-clustered-layerwise-1D-from-cifar', '8-clustered-layerwise-1D-from-cifar'}, ...
  %   ... % {'8-clustered-layerwise-1D-from-coil-100', '8-clustered-layerwise-1D-from-coil-100', '8-clustered-layerwise-1D-from-coil-100'}, ...
  %   ... % {'8-clustered-layerwise-1D-from-mnist', '8-clustered-layerwise-1D-from-mnist', '8-clustered-layerwise-1D-from-mnist'}, ...
  %   ... % {'8-clustered-layerwise-1D-from-stl-10', '8-clustered-layerwise-1D-from-stl-10', '8-clustered-layerwise-1D-from-stl-10'}, ...
  %   ... % {'16-clustered-layerwise-1D-from-cifar', '16-clustered-layerwise-1D-from-cifar', '16-clustered-layerwise-1D-from-cifar'}, ...
  %   ... % {'16-clustered-layerwise-1D-from-coil-100', '16-clustered-layerwise-1D-from-coil-100', '16-clustered-layerwise-1D-from-coil-100'}, ...
  %   ... % {'16-clustered-layerwise-1D-from-mnist', '16-clustered-layerwise-1D-from-mnist', '16-clustered-layerwise-1D-from-mnist'}, ...
  %   ... % {'16-clustered-layerwise-1D-from-stl-10', '16-clustered-layerwise-1D-from-stl-10', '16-clustered-layerwise-1D-from-stl-10'}, ...
  %   {'compRand', 'compRand', 'compRand'}, ...
  %   {'kernelwise-1D-from-cifar', 'kernelwise-1D-from-cifar', 'kernelwise-1D-from-cifar'}, ...
  %   ... % {'kernelwise-1D-from-coil-100', 'kernelwise-1D-from-coil-100', 'kernelwise-1D-from-coil-100'}, ...
  %   ... % {'kernelwise-1D-from-mnist', 'kernelwise-1D-from-mnist', 'kernelwise-1D-from-mnist'}, ...
  %   ... % {'kernelwise-1D-from-stl-10', 'kernelwise-1D-from-stl-10', 'kernelwise-1D-from-stl-10'}, ...
  %   {'layerwise-1D-from-cifar', 'layerwise-1D-from-cifar', 'layerwise-1D-from-cifar'}, ...
  %   ... % {'layerwise-1D-from-coil-100', 'layerwise-1D-from-coil-100', 'layerwise-1D-from-coil-100'}, ...
  %   ... % {'layerwise-1D-from-mnist', 'layerwise-1D-from-mnist', 'layerwise-1D-from-mnist'}, ...
  %   ... % {'layerwise-1D-from-stl-10', 'layerwise-1D-from-stl-10', 'layerwise-1D-from-stl-10'}, ...
  % };

  % imdb_portion_list = [0.1, 0.25, 0.5, 1.0];
  imdb_portion_list = [1.0];

  % weight_decay_list = [0.1, 0.01, 0.001, 0.0001, 0]; % Works: {0.001, 0.0001, 0} Doesn't Work: {0.1, 0.01}
  weight_decay_list = [0.0001];

  debug_flag = true;

  % posneg_balance = 'balanced-low';

% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
% -- ==                                                                   -- ==
% -- ==                           MAIN LOOP                               -- ==
% -- ==                                                                   -- ==
% -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- == -- ==
  for dataset = dataset_list
    for weight_init_sequence = weight_init_sequence_list
      for bottleneck_divide_by = bottleneck_divide_by_list
        for weight_decay = weight_decay_list
          for backprop_depth = backprop_depth_list
            for leave_out_index = leave_out_indices
              tmp_opts.dataset = dataset;
              tmp_opts.posneg_balance = posneg_balance;
              opts.imdb = loadSavedImdb(tmp_opts);
              opts.dataset = char(dataset);
              opts.network_arch = network_arch;
              opts.backprop_depth = backprop_depth;
              opts.weight_decay = weight_decay;
              opts.weight_init_source = weight_init_source;
              opts.weight_init_sequence = weight_init_sequence{1};
              opts.bottleneck_divide_by = bottleneck_divide_by;
              opts.leave_out_type = leave_out_type;
              opts.leave_out_index = leave_out_index;
              opts.debug_flag = debug_flag;
              opts.regen = true;
              opts.gpus = ifNotMacSetGpu(1);
              cnnAmir(opts);
            end
          end
        end
      end
    end
  end
