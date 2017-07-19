% -------------------------------------------------------------------------
function [best_test_accuracy_mean, best_test_accuracy_std] = getSimpleTestAccuracyFromCnn(dataset, imdb, conv_network_arch, gpus)
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
  opts.general.imdb = imdb;
  opts.train.gpus = gpus;

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
    {}, ... % no input_opts here! :)
    'experiment_parent_dir', ...
    fullfile(vl_rootnn, 'experiment_results'));
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'simple-test-acc-CNN-%s-%s-GPU-%d', ...
    opts.paths.time_string, ...
    opts.general.dataset, ...
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



  training_options.experiment_parent_dir = opts.paths.experiment_dir;
  training_options.imdb = opts.general.imdb;
  training_options.network_arch = conv_network_arch;
  training_options.backprop_depth = getFullBackPropDepthForConvArchitecture(conv_network_arch); % compute `backprop_depth` automatically based on `conv_network_arch`

  % remember, we're training conv_network_arch, so the network is going to be initialized with random weights then trained!
  % training_options.weight_init_sequence = weight_init_sequence;

  training_options.gpus = ifNotMacSetGpu(opts.train.gpus);
  training_options.return_performance_summary = true;
  training_options.debug_flag = false;

  % base_learning_rate = [0.1*ones(1,25) 0.03*ones(1,25) 0.01*ones(1,50)];
  base_learning_rate = [0.1*ones(1,15) 0.03*ones(1,15) 0.01*ones(1,15)];

  if strcmp(dataset, 'cifar') || strcmp(dataset, 'cifar-multi-class-subsampled')
    learning_rate_divider_list = [1, 3, 10, 30];
  elseif strcmp(dataset, 'stl-10') || strcmp(dataset, 'stl-10-multi-class-subsampled')
    learning_rate_divider_list = [1, 3, 10, 30] / 10;
  elseif strcmp(dataset, 'mnist') || strcmp(dataset, 'mnist-multi-class-subsampled')
    learning_rate_divider_list = [1, 3, 10, 30];
  elseif strcmp(dataset, 'svhn') || strcmp(dataset, 'svhn-multi-class-subsampled')
    learning_rate_divider_list = [1, 3, 10, 30] / 3;
  else
    throwException('[ERROR] unrecognized dataset.')
  end

  batch_size_list = [50, 100];
  weight_decay_list = [0.01, 0.001, 0.0001];




  learning_rate_divider_list = [3, 10];
  batch_size_list = [50];
  weight_decay_list = [0.01];
  base_learning_rate = [0.1*ones(1,3)];




  number_of_trials = 3;
  test_accuracies_mean = [];
  test_accuracies_std = [];
  experiments = {};
  total_number_of_hyperparams = ...
    length(learning_rate_divider_list) * ...
    length(batch_size_list) * ...
    length(weight_decay_list);

  hyperparam_counter = 1;
  % loop through hyperparameters
  for learning_rate_divider = learning_rate_divider_list
    for batch_size = batch_size_list
      for weight_decay = weight_decay_list
        training_options.learning_rate = base_learning_rate / learning_rate_divider;;
        training_options.batch_size = batch_size;
        training_options.weight_decay = weight_decay;
        % repeat experiment and get averaged results
        tmp_test_accuracies = [];
        afprintf(sprintf('[INFO] Testing hyperparameter setup #%d / %d ...\n', hyperparam_counter, total_number_of_hyperparams));
        for i = 1 : number_of_trials
          afprintf(sprintf('[INFO] Testing repeat #%d / %d ...\n', i, number_of_trials), 1);
          [~, performance_summary] = testCnn(training_options);
          tmp_test_accuracies(end+1) = performance_summary.testing.test.accuracy;
        end

        tmp_results = {};
        tmp_results.learning_rate = training_options.learning_rate;
        tmp_results.batch_size = training_options.batch_size;
        tmp_results.weight_decay = training_options.weight_decay;
        tmp_results.test_accuracy.number_of_trials = number_of_trials;
        tmp_results.test_accuracy.all_results = tmp_test_accuracies;
        tmp_results.test_accuracy.mean = mean(tmp_test_accuracies);
        tmp_results.test_accuracy.std = std(tmp_test_accuracies);
        experiments{end+1} = tmp_results;

        hyperparam_counter  = hyperparam_counter + 1;
      end
    end
  end

  keyboard
  saveStruct2File(experiments, opts.paths.results_file_path, 0);
  keyboard

  best_test_accuracy_mean = 0;
  best_test_accuracy_std = 0;
  for i = 1 : total_number_of_hyperparams
    if experiments{end+1}.test_accuracy.mean > best_test_accuracy_mean
      best_test_accuracy_mean = experiments{end+1}.test_accuracy.mean;
      best_test_accuracy_std = experiments{end+1}.test_accuracy.std;
    end
  end

  % [~, indices] = sort(test_accuracies_mean, 'descend');
  % index_of_best_test_perf = indices(1);

  % best_test_accuracy_mean = test_accuracies_mean(index_of_best_test_perf)
  % best_test_accuracy_std = test_accuracies_std(index_of_best_test_perf)








