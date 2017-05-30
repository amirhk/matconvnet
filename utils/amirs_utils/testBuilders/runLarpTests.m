% -------------------------------------------------------------------------
function runLarpTests(experiment_parent_dir, dataset, posneg_balance, larp_network_arch, non_larp_network_arch, larp_weight_init_type, gpus)
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

  % -------------------------------------------------------------------------
  %                                                              opts.general
  % -------------------------------------------------------------------------
  opts.general.dataset = dataset;

  % -------------------------------------------------------------------------
  %                                                                 opts.imdb
  % -------------------------------------------------------------------------
  opts.imdb.posneg_balance = posneg_balance;
  % opts.imdb.projection = projection;

  % -------------------------------------------------------------------------
  %                                                         opts.network_arch
  % -------------------------------------------------------------------------
  opts.net.larp_network_arch = larp_network_arch;
  opts.net.larp_weight_init_type = larp_weight_init_type;
  opts.net.non_larp_network_arch = non_larp_network_arch;

  % -------------------------------------------------------------------------
  %                                                                opts.train
  % -------------------------------------------------------------------------
  opts.train.gpus = gpus;

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.paths.experiment_parent_dir = experiment_parent_dir;
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'test-larp-tests-%s-%s-%s-GPU-%d', ...
    opts.paths.time_string, ...
    opts.general.dataset, ...
    opts.imdb.posneg_balance, ...
    opts.train.gpus));
  if ~exist(opts.paths.experiment_dir)
    mkdir(opts.paths.experiment_dir);
  end
  opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, 'options.txt');
  opts.paths.results_file_path = fullfile(opts.paths.experiment_dir, 'results.txt');

  % -------------------------------------------------------------------------
  %                                                    save experiment setup!
  % -------------------------------------------------------------------------
  saveStruct2File(opts, opts.paths.options_file_path, 0);

  % TODO:
  % ~~~    1. experiment_parent_dir code
  % ###    2. merge test*.m files (with shared loop function)

  % -------------------------------------------------------------------------
  %                                                            shared options
  % -------------------------------------------------------------------------
  % if strcmp(larp_network_arch, 'larpV1P0-ensemble-sparse-rp-no-nl')
  %   % experiment_options.number_of_folds = 1;
  %   experiment_options.number_of_folds = 3;
  % else
  %   experiment_options.number_of_folds = 3;
  % end
  experiment_options.number_of_folds = 3;
  experiment_options.experiment_parent_dir = opts.paths.experiment_dir;
  experiment_options.dataset = opts.general.dataset;
  experiment_options.posneg_balance = opts.imdb.posneg_balance;
  % experiment_options.projection = opts.imdb.projection;
  experiment_options.gpus = opts.train.gpus;

  % % -------------------------------------------------------------------------
  % %                                                                single cnn
  % % -------------------------------------------------------------------------
  experiment_options.training_method = 'single-cnn';

  experiment_options.larp_network_arch = larp_network_arch;
  experiment_options.larp_weight_init_sequence = getLarpWeightInitSequence(larp_weight_init_type, larp_network_arch);

  mlp_version = lower(non_larp_network_arch(end-1:end));
  conv_arch = getMatchingConvArchitectureForLarpArchitecture(larp_network_arch, mlp_version);
  experiment_options.network_arch = conv_arch;
  experiment_options.backprop_depth = getFullBackPropDepthForConvArchitecture(conv_arch);

  % experiment_options.network_arch = network_arch;
  % experiment_options.backprop_depth = getFullBackPropDepthForNetworkArch(non_larp_network_arch);


  base_learning_rate = [0.1*ones(1,25) 0.03*ones(1,25) 0.01*ones(1,50)];

  if strcmp(dataset, 'cifar') || strcmp(dataset, 'cifar-multi-class-subsampled')
    learning_rate_dividers = [1, 3, 10, 30]; % other
    % learning_rate_dividers = [30]; % other
  elseif strcmp(dataset, 'stl-10') || strcmp(dataset, 'stl-10-multi-class-subsampled')
    learning_rate_dividers = [1, 3, 10, 30] / 10; % stl-10
    % learning_rate_dividers = [30] / 10; % stl-10
  elseif strcmp(dataset, 'mnist') || strcmp(dataset, 'mnist-multi-class-subsampled')
    learning_rate_dividers = [1, 3, 10, 30]; % other
    % learning_rate_dividers = [30]; % other
  elseif strcmp(dataset, 'svhn') || strcmp(dataset, 'svhn-multi-class-subsampled')
    learning_rate_dividers = [1, 3, 10, 30] / 3; % svhn
    % learning_rate_dividers = [30] / 3; % svhn
  end

  batch_sizes = [50, 100];
  weight_decays = [0.01, 0.001, 0.0001];

  for learning_rate_divider = learning_rate_dividers
    experiment_options.learning_rate = base_learning_rate / learning_rate_divider;
    for batch_size = batch_sizes
      experiment_options.batch_size = batch_size;
      for weight_decay = weight_decays
        experiment_options.weight_decay = weight_decay;
        testKFold(experiment_options);
      end
    end
  end

































