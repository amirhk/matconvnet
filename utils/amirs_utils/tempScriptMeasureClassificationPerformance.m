% -------------------------------------------------------------------------
function tempScriptMeasureClassificationPerformance(dataset, posneg_balance, classification_method, save_results)
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
  %                                                                     Setup
  % -------------------------------------------------------------------------
  % classification_method = 'cnn';
  % classification_method = '1-knn';
  % classification_method = '3-knn';
  % classification_method = 'c-sep';
  % classification_method = 'libsvm';
  % classification_method = 'mlp-64-10';
  % classification_method = 'mlp-500-100';
  % classification_method = 'mlp-500-1000-100';
  number_of_trials = 3;
  all_experiments_multi_run = {};

  % -------------------------------------------------------------------------
  %                                                              opts.general
  % -------------------------------------------------------------------------
  opts.general.dataset = dataset;
  opts.general.posneg_balance = posneg_balance;
  opts.general.classification_method = classification_method;
  opts.general.number_of_trials = number_of_trials;

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
    {}, ... % no input_opts here! :)
    'experiment_parent_dir', ...
    fullfile(vl_rootnn, 'experiment_results'));
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'test-classification-perf-%s-rp-tests-%s-%s-%s', ...
    opts.paths.time_string, ...
    classification_method, ...
    opts.general.dataset, ...
    opts.general.posneg_balance));
  if ~exist(opts.paths.experiment_dir)
    mkdir(opts.paths.experiment_dir);
  end
  opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, '_options.txt');
  opts.paths.results_file_path = fullfile(opts.paths.experiment_dir, '_results.txt');

  % -------------------------------------------------------------------------
  %                          save experiment setup (don't save imdb or net!!)
  % -------------------------------------------------------------------------
  saveStruct2File(opts, opts.paths.options_file_path, 0);


  % -------------------------------------------------------------------------
  %                                                                      beef
  % -------------------------------------------------------------------------
  % [~, tmp_experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, false, false);
  % for i = 1 : numel(tmp_experiments)
  larp_network_arch_list = getLarpNetworkArchList();
  for i = 1 : numel(larp_network_arch_list)
    all_experiments_multi_run{i}.performance = [];
  end

  for kk = 1:number_of_trials
    afprintf(sprintf('[INFO] Testing trial #%d / %d ...\n', kk, number_of_trials));
    all_experiments_single_run = runAllExperimentsOnce(opts.paths.experiment_dir, dataset, posneg_balance, classification_method);
    for i = 1 : numel(all_experiments_single_run)
      all_experiments_multi_run{i}.performance(end+1) = all_experiments_single_run{i}.performance;
    end
    afprintf(sprintf('[INFO] done!\n'));

    % for i = 1 : numel(tmp_experiments)
    for i = 1 : numel(larp_network_arch_list)
      all_experiments_multi_run{i}.performance_metric = classification_method;
      all_experiments_multi_run{i}.performance_mean = mean(all_experiments_multi_run{i}.performance);
      all_experiments_multi_run{i}.performance_std = std(all_experiments_multi_run{i}.performance);
    end

    % -------------------------------------------------------------------------
    %                                                               save output
    % -------------------------------------------------------------------------
    % don't amend file, but overwrite...
    if exist(opts.paths.results_file_path)
      delete(opts.paths.results_file_path);
    end
    saveStruct2File(all_experiments_multi_run, opts.paths.results_file_path, 0);
  end

  % plot_title = sprintf('classification perf - %s - %s - %s', classification_method, dataset, posneg_balance);
  % tempScriptPlotRPTests(all_experiments_multi_run, plot_title, save_results);








% % -------------------------------------------------------------------------
% function all_experiments_single_run = runAllExperimentsOnce(experiment_dir, dataset, posneg_balance, classification_method)
% % -------------------------------------------------------------------------
%   afprintf(sprintf('[INFO] Setting up experiment imdbs...\n'));
%   [~, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, false, true);
%   afprintf(sprintf('done!\n'));
%   for i = 1 : numel(experiments)
%     experiment_options = {};
%     experiment_options.imdb = experiments{i}.imdb;
%     experiment_options.dataset = dataset;
%     experiment_options.posneg_balance = posneg_balance;
%     experiment_options.experiment_parent_dir = experiment_dir;
%     experiment_options.debug_flag = false;
%     switch classification_method
%       case '1-knn'
%         experiment_options.number_of_nearest_neighbors = 1;
%         performance = getSimpleTestAccuracyFromKnn(experiment_options);
%       case '3-knn'
%         experiment_options.number_of_nearest_neighbors = 3;
%         performance = getSimpleTestAccuracyFromKnn(experiment_options);
%       case 'c-sep'
%         performance = getAverageClassCSeparation(experiment_options.imdb);
%       case 'libsvm'
%         performance = getSimpleTestAccuracyFromLibSvm(experiment_options);
%       case 'mlp-64-10'
%         experiment_options.number_of_hidden_nodes = [64, 10];
%         performance = getSimpleTestAccuracyFromMLP(experiment_options);
%       case 'mlp-500-100'
%         experiment_options.number_of_hidden_nodes = [500, 100];
%         performance = getSimpleTestAccuracyFromMLP(experiment_options);
%       case 'mlp-500-1000-100'
%         experiment_options.number_of_hidden_nodes = [500, 1000, 100];
%         performance = getSimpleTestAccuracyFromMLP(experiment_options);
%       case 'cnn'

%         experiment_options.gpus = 3;

%         % TODO: this has to somehow be detected automatically....
%         % experiment_options.conv_network_arch = 'convV0P0RL0+fcV1-RF16CH64';
%         % experiment_options.conv_network_arch = 'convV0P0RL0+fcV1-RF32CH64';

%         % experiment_options.conv_network_arch = 'convV1P1RL1-RF32CH3+fcV1-RF16CH64';
%         % experiment_options.conv_network_arch = 'convV3P1RL3-RF32CH3+fcV1-RF16CH64';
%         % experiment_options.conv_network_arch = 'convV5P1RL5-RF32CH3+fcV1-RF16CH64';

%         % experiment_options.conv_network_arch = 'convV1P0RL1-RF32CH3+fcV1-RF32CH64';
%         % experiment_options.conv_network_arch = 'convV3P0RL3-RF32CH3+fcV1-RF32CH64';
%         % experiment_options.conv_network_arch = 'convV5P0RL5-RF32CH3+fcV1-RF32CH64';

%         [best_test_accuracy_mean, best_test_accuracy_std] = getSimpleTestAccuracyFromCnn(experiment_options);
%         performance = best_test_accuracy_mean;
%     end
%     experiments{i}.performance = performance;
%   end

%   all_experiments_single_run = experiments;











% -------------------------------------------------------------------------
function all_experiments_single_run = runAllExperimentsOnce(experiment_dir, dataset, posneg_balance, classification_method)
% -------------------------------------------------------------------------
  % afprintf(sprintf('[INFO] Setting up experiment imdbs...\n'));
  % [~, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, false, true);
  % afprintf(sprintf('done!\n'));
  larp_network_arch_list = getLarpNetworkArchList();

  larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';

  [~, tmp_experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, false, false);
  original_imdb = tmp_experiments{1}.imdb;

  for i = 1 : numel(larp_network_arch_list)
    larp_network_arch = larp_network_arch_list{i};
    afprintf(sprintf('[INFO] Testing experiment #%d / %d (larp_network_arch: %s) ...', i, numel(larp_network_arch_list), larp_network_arch));


    experiment_options = {};
    % experiment_options.imdb = experiments{i}.imdb;
    experiment_options.imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, -1);

    experiment_options.dataset = dataset;
    experiment_options.posneg_balance = posneg_balance;
    experiment_options.experiment_parent_dir = experiment_dir;
    experiment_options.debug_flag = false;
    switch classification_method
      case '1-knn'
        experiment_options.number_of_nearest_neighbors = 1;
        performance = getSimpleTestAccuracyFromKnn(experiment_options);
      case '3-knn'
        experiment_options.number_of_nearest_neighbors = 3;
        performance = getSimpleTestAccuracyFromKnn(experiment_options);
      case 'c-sep'
        performance = getAverageClassCSeparation(experiment_options.imdb);
      case 'libsvm'
        performance = getSimpleTestAccuracyFromLibSvm(experiment_options);
      case 'mlp-64-10'
        experiment_options.number_of_hidden_nodes = [64, 10];
        performance = getSimpleTestAccuracyFromMLP(experiment_options);
      case 'mlp-500-100'
        experiment_options.number_of_hidden_nodes = [500, 100];
        performance = getSimpleTestAccuracyFromMLP(experiment_options);
      case 'mlp-500-1000-100'
        experiment_options.number_of_hidden_nodes = [500, 1000, 100];
        performance = getSimpleTestAccuracyFromMLP(experiment_options);
      case 'cnn'

        experiment_options.gpus = 3;

        % TODO: this has to somehow be detected automatically....
        % experiment_options.conv_network_arch = 'convV0P0RL0+fcV1-RF16CH64';
        % experiment_options.conv_network_arch = 'convV0P0RL0+fcV1-RF32CH64';

        % experiment_options.conv_network_arch = 'convV1P1RL1-RF32CH3+fcV1-RF16CH64';
        % experiment_options.conv_network_arch = 'convV3P1RL3-RF32CH3+fcV1-RF16CH64';
        % experiment_options.conv_network_arch = 'convV5P1RL5-RF32CH3+fcV1-RF16CH64';

        % experiment_options.conv_network_arch = 'convV1P0RL1-RF32CH3+fcV1-RF32CH64';
        % experiment_options.conv_network_arch = 'convV3P0RL3-RF32CH3+fcV1-RF32CH64';
        % experiment_options.conv_network_arch = 'convV5P0RL5-RF32CH3+fcV1-RF32CH64';

        [best_test_accuracy_mean, best_test_accuracy_std] = getSimpleTestAccuracyFromCnn(experiment_options);
        performance = best_test_accuracy_mean;
    end
    experiments{i}.performance = performance;
    afprintf(sprintf('[INFO] done!\n'));
  end

  all_experiments_single_run = experiments;






% -------------------------------------------------------------------------
function projected_imdb = getRandomlyProjectedImdb(original_imdb, dataset, larp_weight_init_type, larp_network_arch, projection_depth)
% -------------------------------------------------------------------------
  fh_projection_utils = projectionUtils;
  larp_weight_init_sequence = getLarpWeightInitSequence(larp_weight_init_type, larp_network_arch);
  projection_net = fh_projection_utils.getProjectionNetworkObject(dataset, larp_network_arch, larp_weight_init_sequence);
  projected_imdb = fh_projection_utils.projectImdbThroughNetwork(original_imdb, projection_net, projection_depth);


% -------------------------------------------------------------------------
function larp_network_arch_list = getLarpNetworkArchList()
% -------------------------------------------------------------------------
  larp_network_arch_list = { ...
    ... 'larpV3P3RL3-final-conv-16-kernels-first-max-pool-then-relu', ...
    ... 'larpV3P3RL3-final-conv-16-kernels-first-relu-then-max-pool', ...
    ...
    ...
    'custom-3-L-3-4-4-max-pool', ... % relu-
    'custom-3-L-3-4-16-max-pool', ... % relu-
    'custom-3-L-3-4-64-max-pool', ... % relu-
    'custom-3-L-3-4-256-max-pool', ... % relu-
    ... ...
    'custom-3-L-5-4-4-max-pool', ... % relu-
    'custom-3-L-5-4-16-max-pool', ... % relu-
    'custom-3-L-5-4-64-max-pool', ... % relu-
    'custom-3-L-5-4-256-max-pool', ... % relu-
    ... ...
    'custom-3-L-7-4-4-max-pool', ... % relu-
    'custom-3-L-7-4-16-max-pool', ... % relu-
    'custom-3-L-7-4-64-max-pool', ... % relu-
    'custom-3-L-7-4-256-max-pool', ... % relu-
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    'custom-3-L-3-16-4-max-pool', ... % relu-
    'custom-3-L-3-16-16-max-pool', ... % relu-
    'custom-3-L-3-16-64-max-pool', ... % relu-
    'custom-3-L-3-16-256-max-pool', ... % relu-
    ... ...
    'custom-3-L-5-16-4-max-pool', ... % relu-
    'custom-3-L-5-16-16-max-pool', ... % relu-
    'custom-3-L-5-16-64-max-pool', ... % relu-
    'custom-3-L-5-16-256-max-pool', ... % relu-
    ... ...
    'custom-3-L-7-16-4-max-pool', ... % relu-
    'custom-3-L-7-16-16-max-pool', ... % relu-
    'custom-3-L-7-16-64-max-pool', ... % relu-
    'custom-3-L-7-16-256-max-pool', ... % relu-
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    'custom-3-L-3-64-4-max-pool', ... % relu-
    'custom-3-L-3-64-16-max-pool', ... % relu-
    'custom-3-L-3-64-64-max-pool', ... % relu-
    'custom-3-L-3-64-256-max-pool', ... % relu-
    ... ...
    'custom-3-L-5-64-4-max-pool', ... % relu-
    'custom-3-L-5-64-16-max-pool', ... % relu-
    'custom-3-L-5-64-64-max-pool', ... % relu-
    'custom-3-L-5-64-256-max-pool', ... % relu-
    ... ...
    'custom-3-L-7-64-4-max-pool', ... % relu-
    'custom-3-L-7-64-16-max-pool', ... % relu-
    'custom-3-L-7-64-64-max-pool', ... % relu-
    'custom-3-L-7-64-256-max-pool', ... % relu-
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    'custom-3-L-3-256-4-max-pool', ... % relu-
    'custom-3-L-3-256-16-max-pool', ... % relu-
    'custom-3-L-3-256-64-max-pool', ... % relu-
    'custom-3-L-3-256-256-max-pool', ... % relu-
    ... ...
    'custom-3-L-5-256-4-max-pool', ... % relu-
    'custom-3-L-5-256-16-max-pool', ... % relu-
    'custom-3-L-5-256-64-max-pool', ... % relu-
    'custom-3-L-5-256-256-max-pool', ... % relu-
    ... ...
    'custom-3-L-7-256-4-max-pool', ... % relu-
    'custom-3-L-7-256-16-max-pool', ... % relu-
    'custom-3-L-7-256-64-max-pool', ... % relu-
    'custom-3-L-7-256-256-max-pool', ... % relu-
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... 'custom-5-L-3-4-4-relu-max-pool', ...
    ... 'custom-5-L-3-4-32-relu-max-pool', ...
    ... 'custom-5-L-3-4-256-relu-max-pool', ...
    ... 'custom-5-L-3-4-2048-relu-max-pool', ...
    ... ...
    ... 'custom-5-L-3-32-4-relu-max-pool', ...
    ... 'custom-5-L-3-32-32-relu-max-pool', ...
    ... 'custom-5-L-3-32-256-relu-max-pool', ...
    ... 'custom-5-L-3-32-2048-relu-max-pool', ...
    ... ...
    ... 'custom-5-L-3-256-4-relu-max-pool', ...
    ... 'custom-5-L-3-256-32-relu-max-pool', ...
    ... 'custom-5-L-3-256-256-relu-max-pool', ...
    ... 'custom-5-L-3-256-2048-relu-max-pool', ...
    ... ...
    ... 'custom-5-L-3-2048-4-relu-max-pool', ...
    ... 'custom-5-L-3-2048-32-relu-max-pool', ...
    ... 'custom-5-L-3-2048-256-relu-max-pool', ...
    ... 'custom-5-L-3-2048-2048-relu-max-pool', ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... 'custom-5-L-5-4-4-relu-max-pool', ...
    ... 'custom-5-L-5-4-32-relu-max-pool', ...
    ... 'custom-5-L-5-4-256-relu-max-pool', ...
    ... 'custom-5-L-5-4-2048-relu-max-pool', ...
    ... ...
    ... 'custom-5-L-5-32-4-relu-max-pool', ...
    ... 'custom-5-L-5-32-32-relu-max-pool', ...
    ... 'custom-5-L-5-32-256-relu-max-pool', ...
    ... 'custom-5-L-5-32-2048-relu-max-pool', ...
    ... ...
    ... 'custom-5-L-5-256-4-relu-max-pool', ...
    ... 'custom-5-L-5-256-32-relu-max-pool', ...
    ... 'custom-5-L-5-256-256-relu-max-pool', ...
    ... 'custom-5-L-5-256-2048-relu-max-pool', ...
    ... ...
    ... 'custom-5-L-5-2048-4-relu-max-pool', ...
    ... 'custom-5-L-5-2048-32-relu-max-pool', ...
    ... 'custom-5-L-5-2048-256-relu-max-pool', ...
    ... 'custom-5-L-5-2048-2048-relu-max-pool', ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... 'custom-5-L-7-4-4-relu-max-pool', ...
    ... 'custom-5-L-7-4-32-relu-max-pool', ...
    ... 'custom-5-L-7-4-256-relu-max-pool', ...
    ... 'custom-5-L-7-4-2048-relu-max-pool', ...
    ... ...
    ... 'custom-5-L-7-32-4-relu-max-pool', ...
    ... 'custom-5-L-7-32-32-relu-max-pool', ...
    ... 'custom-5-L-7-32-256-relu-max-pool', ...
    ... 'custom-5-L-7-32-2048-relu-max-pool', ...
    ... ...
    ... 'custom-5-L-7-256-4-relu-max-pool', ...
    ... 'custom-5-L-7-256-32-relu-max-pool', ...
    ... 'custom-5-L-7-256-256-relu-max-pool', ...
    ... 'custom-5-L-7-256-2048-relu-max-pool', ...
    ... ...
    ... 'custom-5-L-7-2048-4-relu-max-pool', ...
    ... 'custom-5-L-7-2048-32-relu-max-pool', ...
    ... 'custom-5-L-7-2048-256-relu-max-pool', ...
    ... 'custom-5-L-7-2048-2048-relu-max-pool', ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... ...
    ... 'larpV0P0RL0', ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... 'custom-1-L-3-4-16-', ...                                                      % proj dim = 32 x 32 x 4 = 4096
    ... ... 'custom-1-L-3-4-16-relu', ...                                                  % proj dim = 32 x 32 x 4 = 4096
    ... 'custom-1-L-3-4-16-max-pool', ...                                              % proj dim = 32 x 32 x 4 / (4 ^ 1) = 1024
    ... 'custom-1-L-3-4-16-relu-max-pool', ...                                         % proj dim = 32 x 32 x 4 / (4 ^ 1) = 1024
    ... ... 'custom-1-L-7-4-16-', ...                                                      % proj dim = 32 x 32 x 4 = 4096
    ... ... 'custom-1-L-7-4-16-relu', ...                                                  % proj dim = 32 x 32 x 4 = 4096
    ... 'custom-1-L-7-4-16-max-pool', ...                                              % proj dim = 32 x 32 x 4 / (4 ^ 1) = 1024
    ... 'custom-1-L-7-4-16-relu-max-pool', ...                                         % proj dim = 32 x 32 x 4 / (4 ^ 1) = 1024
    ... ... 'custom-1-L-11-4-16-', ...                                                     % proj dim = 32 x 32 x 4 = 4096
    ... ... 'custom-1-L-11-4-16-relu', ...                                                 % proj dim = 32 x 32 x 4 = 4096
    ... 'custom-1-L-11-4-16-max-pool', ...                                             % proj dim = 32 x 32 x 4 / (4 ^ 1) = 1024
    ... 'custom-1-L-11-4-16-relu-max-pool', ...                                        % proj dim = 32 x 32 x 4 / (4 ^ 1) = 1024
    ... ... ...
    ... ... 'custom-1-L-3-16-16-', ...                                                     % proj dim = 32 x 32 x 16 = 16384
    ... ... 'custom-1-L-3-16-16-relu', ...                                                 % proj dim = 32 x 32 x 16 = 16384
    ... 'custom-1-L-3-16-16-max-pool', ...                                             % proj dim = 32 x 32 x 16 / (4 ^ 1) = 4096
    ... 'custom-1-L-3-16-16-relu-max-pool', ...                                        % proj dim = 32 x 32 x 16 / (4 ^ 1) = 4096
    ... ... 'custom-1-L-7-16-16-', ...                                                     % proj dim = 32 x 32 x 16 = 16384
    ... ... 'custom-1-L-7-16-16-relu', ...                                                 % proj dim = 32 x 32 x 16 = 16384
    ... 'custom-1-L-7-16-16-max-pool', ...                                             % proj dim = 32 x 32 x 16 / (4 ^ 1) = 4096
    ... 'custom-1-L-7-16-16-relu-max-pool', ...                                        % proj dim = 32 x 32 x 16 / (4 ^ 1) = 4096
    ... ... 'custom-1-L-11-16-16-', ...                                                    % proj dim = 32 x 32 x 16 = 16384
    ... ... 'custom-1-L-11-16-16-relu', ...                                                % proj dim = 32 x 32 x 16 = 16384
    ... 'custom-1-L-11-16-16-max-pool', ...                                            % proj dim = 32 x 32 x 16 / (4 ^ 1) = 4096
    ... 'custom-1-L-11-16-16-relu-max-pool', ...                                       % proj dim = 32 x 32 x 16 / (4 ^ 1) = 4096
    ... ... ...
    ... ... 'custom-1-L-3-64-16-', ...                                                     % proj dim = 32 x 32 x 64 = 65536
    ... ... 'custom-1-L-3-64-16-relu', ...                                                 % proj dim = 32 x 32 x 64 = 65536
    ... 'custom-1-L-3-64-16-max-pool', ...                                             % proj dim = 32 x 32 x 64 / (4 ^ 1) = 16384
    ... 'custom-1-L-3-64-16-relu-max-pool', ...                                        % proj dim = 32 x 32 x 64 / (4 ^ 1) = 16384
    ... ... 'custom-1-L-7-64-16-', ...                                                     % proj dim = 32 x 32 x 64 = 65536
    ... ... 'custom-1-L-7-64-16-relu', ...                                                 % proj dim = 32 x 32 x 64 = 65536
    ... 'custom-1-L-7-64-16-max-pool', ...                                             % proj dim = 32 x 32 x 64 / (4 ^ 1) = 16384
    ... 'custom-1-L-7-64-16-relu-max-pool', ...                                        % proj dim = 32 x 32 x 64 / (4 ^ 1) = 16384
    ... ... 'custom-1-L-11-64-16-', ...                                                    % proj dim = 32 x 32 x 64 = 65536
    ... ... 'custom-1-L-11-64-16-relu', ...                                                % proj dim = 32 x 32 x 64 = 65536
    ... 'custom-1-L-11-64-16-max-pool', ...                                            % proj dim = 32 x 32 x 64 / (4 ^ 1) = 16384
    ... 'custom-1-L-11-64-16-relu-max-pool', ...                                       % proj dim = 32 x 32 x 64 / (4 ^ 1) = 16384
    ... ... ...
    ... ... 'custom-1-L-3-256-16-', ...                                                    % proj dim = 32 x 32 x 256 = 262144
    ... ... 'custom-1-L-3-256-16-relu', ...                                                % proj dim = 32 x 32 x 256 = 262144
    ... 'custom-1-L-3-256-16-max-pool', ...                                            % proj dim = 32 x 32 x 256 / (4 ^ 1) = 65536
    ... 'custom-1-L-3-256-16-relu-max-pool', ...                                       % proj dim = 32 x 32 x 256 / (4 ^ 1) = 65536
    ... ... 'custom-1-L-7-256-16-', ...                                                    % proj dim = 32 x 32 x 256 = 262144
    ... ... 'custom-1-L-7-256-16-relu', ...                                                % proj dim = 32 x 32 x 256 = 262144
    ... 'custom-1-L-7-256-16-max-pool', ...                                            % proj dim = 32 x 32 x 256 / (4 ^ 1) = 65536
    ... 'custom-1-L-7-256-16-relu-max-pool', ...                                       % proj dim = 32 x 32 x 256 / (4 ^ 1) = 65536
    ... ... 'custom-1-L-11-256-16-', ...                                                   % proj dim = 32 x 32 x 256 = 262144
    ... ... 'custom-1-L-11-256-16-relu', ...                                               % proj dim = 32 x 32 x 256 = 262144
    ... 'custom-1-L-11-256-16-max-pool', ...                                           % proj dim = 32 x 32 x 256 / (4 ^ 1) = 65536
    ... 'custom-1-L-11-256-16-relu-max-pool', ...                                      % proj dim = 32 x 32 x 256 / (4 ^ 1) = 65536
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... 'custom-3-L-3-4-16-', ...                                                      % proj dim = 32 x 32 x 4 = 4096
    ... ... 'custom-3-L-3-4-16-relu', ...                                                  % proj dim = 32 x 32 x 4 = 4096
    ... 'custom-3-L-3-4-16-max-pool', ...                                              % proj dim = 32 x 32 x 4 / (4 ^ 3) = 64
    ... 'custom-3-L-3-4-16-relu-max-pool', ...                                         % proj dim = 32 x 32 x 4 / (4 ^ 3) = 64
    ... ... 'custom-3-L-7-4-16-', ...                                                      % proj dim = 32 x 32 x 4 = 4096
    ... ... 'custom-3-L-7-4-16-relu', ...                                                  % proj dim = 32 x 32 x 4 = 4096
    ... 'custom-3-L-7-4-16-max-pool', ...                                              % proj dim = 32 x 32 x 4 / (4 ^ 3) = 64
    ... 'custom-3-L-7-4-16-relu-max-pool', ...                                         % proj dim = 32 x 32 x 4 / (4 ^ 3) = 64
    ... ... 'custom-3-L-11-4-16-', ...                                                     % proj dim = 32 x 32 x 4 = 4096
    ... ... 'custom-3-L-11-4-16-relu', ...                                                 % proj dim = 32 x 32 x 4 = 4096
    ... 'custom-3-L-11-4-16-max-pool', ...                                             % proj dim = 32 x 32 x 4 / (4 ^ 3) = 64
    ... 'custom-3-L-11-4-16-relu-max-pool', ...                                        % proj dim = 32 x 32 x 4 / (4 ^ 3) = 64
    ... ... ...
    ... ... 'custom-3-L-3-16-16-', ...                                                     % proj dim = 32 x 32 x 16 = 16384
    ... ... 'custom-3-L-3-16-16-relu', ...                                                 % proj dim = 32 x 32 x 16 = 16384
    ... 'custom-3-L-3-16-16-max-pool', ...                                             % proj dim = 32 x 32 x 16 / (4 ^ 3) = 256
    ... 'custom-3-L-3-16-16-relu-max-pool', ...                                        % proj dim = 32 x 32 x 16 / (4 ^ 3) = 256
    ... ... 'custom-3-L-7-16-16-', ...                                                     % proj dim = 32 x 32 x 16 = 16384
    ... ... 'custom-3-L-7-16-16-relu', ...                                                 % proj dim = 32 x 32 x 16 = 16384
    ... 'custom-3-L-7-16-16-max-pool', ...                                             % proj dim = 32 x 32 x 16 / (4 ^ 3) = 256
    ... 'custom-3-L-7-16-16-relu-max-pool', ...                                        % proj dim = 32 x 32 x 16 / (4 ^ 3) = 256
    ... ... 'custom-3-L-11-16-16-', ...                                                    % proj dim = 32 x 32 x 16 = 16384
    ... ... 'custom-3-L-11-16-16-relu', ...                                                % proj dim = 32 x 32 x 16 = 16384
    ... 'custom-3-L-11-16-16-max-pool', ...                                            % proj dim = 32 x 32 x 16 / (4 ^ 3) = 256
    ... 'custom-3-L-11-16-16-relu-max-pool', ...                                       % proj dim = 32 x 32 x 16 / (4 ^ 3) = 256
    ... ... ...
    ... ... 'custom-3-L-3-64-16-', ...                                                     % proj dim = 32 x 32 x 64 = 65536
    ... ... 'custom-3-L-3-64-16-relu', ...                                                 % proj dim = 32 x 32 x 64 = 65536
    ... 'custom-3-L-3-64-16-max-pool', ...                                             % proj dim = 32 x 32 x 64 / (4 ^ 3) = 1024
    ... 'custom-3-L-3-64-16-relu-max-pool', ...                                        % proj dim = 32 x 32 x 64 / (4 ^ 3) = 1024
    ... ... 'custom-3-L-7-64-16-', ...                                                     % proj dim = 32 x 32 x 64 = 65536
    ... ... 'custom-3-L-7-64-16-relu', ...                                                 % proj dim = 32 x 32 x 64 = 65536
    ... 'custom-3-L-7-64-16-max-pool', ...                                             % proj dim = 32 x 32 x 64 / (4 ^ 3) = 1024
    ... 'custom-3-L-7-64-16-relu-max-pool', ...                                        % proj dim = 32 x 32 x 64 / (4 ^ 3) = 1024
    ... ... 'custom-3-L-11-64-16-', ...                                                    % proj dim = 32 x 32 x 64 = 65536
    ... ... 'custom-3-L-11-64-16-relu', ...                                                % proj dim = 32 x 32 x 64 = 65536
    ... 'custom-3-L-11-64-16-max-pool', ...                                            % proj dim = 32 x 32 x 64 / (4 ^ 3) = 1024
    ... 'custom-3-L-11-64-16-relu-max-pool', ...                                       % proj dim = 32 x 32 x 64 / (4 ^ 3) = 1024
    ... ... ...
    ... ... 'custom-3-L-3-256-16-', ...                                                    % proj dim = 32 x 32 x 256 = 262144
    ... ... 'custom-3-L-3-256-16-relu', ...                                                % proj dim = 32 x 32 x 256 = 262144
    ... 'custom-3-L-3-256-16-max-pool', ...                                            % proj dim = 32 x 32 x 256 / (4 ^ 3) = 4096
    ... 'custom-3-L-3-256-16-relu-max-pool', ...                                       % proj dim = 32 x 32 x 256 / (4 ^ 3) = 4096
    ... ... 'custom-3-L-7-256-16-', ...                                                    % proj dim = 32 x 32 x 256 = 262144
    ... ... 'custom-3-L-7-256-16-relu', ...                                                % proj dim = 32 x 32 x 256 = 262144
    ... 'custom-3-L-7-256-16-max-pool', ...                                            % proj dim = 32 x 32 x 256 / (4 ^ 3) = 4096
    ... 'custom-3-L-7-256-16-relu-max-pool', ...                                       % proj dim = 32 x 32 x 256 / (4 ^ 3) = 4096
    ... ... 'custom-3-L-11-256-16-', ...                                                   % proj dim = 32 x 32 x 256 = 262144
    ... ... 'custom-3-L-11-256-16-relu', ...                                               % proj dim = 32 x 32 x 256 = 262144
    ... 'custom-3-L-11-256-16-max-pool', ...                                           % proj dim = 32 x 32 x 256 / (4 ^ 3) = 4096
    ... 'custom-3-L-11-256-16-relu-max-pool', ...                                      % proj dim = 32 x 32 x 256 / (4 ^ 3) = 4096
    ... ... ...
    ... ... 'custom-3-L-3-1024-16-', ...                                                   % proj dim = 32 x 32 x 1024 = 1048576
    ... ... 'custom-3-L-3-1024-16-relu', ...                                               % proj dim = 32 x 32 x 1024 = 1048576
    ... 'custom-3-L-3-1024-16-max-pool', ...                                           % proj dim = 32 x 32 x 1024 / (4 ^ 3) = 16384
    ... 'custom-3-L-3-1024-16-relu-max-pool', ...                                      % proj dim = 32 x 32 x 1024 / (4 ^ 3) = 16384
    ... ... 'custom-3-L-7-1024-16-', ...                                                   % proj dim = 32 x 32 x 1024 = 1048576
    ... ... 'custom-3-L-7-1024-16-relu', ...                                               % proj dim = 32 x 32 x 1024 = 1048576
    ... 'custom-3-L-7-1024-16-max-pool', ...                                           % proj dim = 32 x 32 x 1024 / (4 ^ 3) = 16384
    ... 'custom-3-L-7-1024-16-relu-max-pool', ...                                      % proj dim = 32 x 32 x 1024 / (4 ^ 3) = 16384
    ... ... 'custom-3-L-11-1024-16-', ...                                                  % proj dim = 32 x 32 x 1024 = 1048576
    ... ... 'custom-3-L-11-1024-16-relu', ...                                              % proj dim = 32 x 32 x 1024 = 1048576
    ... 'custom-3-L-11-1024-16-max-pool', ...                                          % proj dim = 32 x 32 x 1024 / (4 ^ 3) = 16384
    ... 'custom-3-L-11-1024-16-relu-max-pool', ...                                     % proj dim = 32 x 32 x 1024 / (4 ^ 3) = 16384
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... ...
    ... ... 'custom-5-L-3-4-16-', ...                                                      % proj dim = 32 x 32 x 4 = 4096
    ... ... 'custom-5-L-3-4-16-relu', ...                                                  % proj dim = 32 x 32 x 4 = 4096
    ... 'custom-5-L-3-4-16-max-pool', ...                                              % proj dim = 32 x 32 x 4 / (4 ^ 5) = 4
    ... 'custom-5-L-3-4-16-relu-max-pool', ...                                         % proj dim = 32 x 32 x 4 / (4 ^ 5) = 4
    ... ... 'custom-5-L-7-4-16-', ...                                                      % proj dim = 32 x 32 x 4 = 4096
    ... ... 'custom-5-L-7-4-16-relu', ...                                                  % proj dim = 32 x 32 x 4 = 4096
    ... 'custom-5-L-7-4-16-max-pool', ...                                              % proj dim = 32 x 32 x 4 / (4 ^ 5) = 4
    ... 'custom-5-L-7-4-16-relu-max-pool', ...                                         % proj dim = 32 x 32 x 4 / (4 ^ 5) = 4
    ... ... 'custom-5-L-11-4-16-', ...                                                     % proj dim = 32 x 32 x 4 = 4096
    ... ... 'custom-5-L-11-4-16-relu', ...                                                 % proj dim = 32 x 32 x 4 = 4096
    ... 'custom-5-L-11-4-16-max-pool', ...                                             % proj dim = 32 x 32 x 4 / (4 ^ 5) = 4
    ... 'custom-5-L-11-4-16-relu-max-pool', ...                                        % proj dim = 32 x 32 x 4 / (4 ^ 5) = 4
    ... ... ...
    ... ... 'custom-5-L-3-16-16-', ...                                                     % proj dim = 32 x 32 x 16 = 16384
    ... ... 'custom-5-L-3-16-16-relu', ...                                                 % proj dim = 32 x 32 x 16 = 16384
    ... 'custom-5-L-3-16-16-max-pool', ...                                             % proj dim = 32 x 32 x 16 / (4 ^ 5) = 16
    ... 'custom-5-L-3-16-16-relu-max-pool', ...                                        % proj dim = 32 x 32 x 16 / (4 ^ 5) = 16
    ... ... 'custom-5-L-7-16-16-', ...                                                     % proj dim = 32 x 32 x 16 = 16384
    ... ... 'custom-5-L-7-16-16-relu', ...                                                 % proj dim = 32 x 32 x 16 = 16384
    ... 'custom-5-L-7-16-16-max-pool', ...                                             % proj dim = 32 x 32 x 16 / (4 ^ 5) = 16
    ... 'custom-5-L-7-16-16-relu-max-pool', ...                                        % proj dim = 32 x 32 x 16 / (4 ^ 5) = 16
    ... ... 'custom-5-L-11-16-16-', ...                                                    % proj dim = 32 x 32 x 16 = 16384
    ... ... 'custom-5-L-11-16-16-relu', ...                                                % proj dim = 32 x 32 x 16 = 16384
    ... 'custom-5-L-11-16-16-max-pool', ...                                            % proj dim = 32 x 32 x 16 / (4 ^ 5) = 16
    ... 'custom-5-L-11-16-16-relu-max-pool', ...                                       % proj dim = 32 x 32 x 16 / (4 ^ 5) = 16
    ... ... ...
    ... ... 'custom-5-L-3-64-16-', ...                                                     % proj dim = 32 x 32 x 64 = 65536
    ... ... 'custom-5-L-3-64-16-relu', ...                                                 % proj dim = 32 x 32 x 64 = 65536
    ... 'custom-5-L-3-64-16-max-pool', ...                                             % proj dim = 32 x 32 x 64 / (4 ^ 5) = 64
    ... 'custom-5-L-3-64-16-relu-max-pool', ...                                        % proj dim = 32 x 32 x 64 / (4 ^ 5) = 64
    ... ... 'custom-5-L-7-64-16-', ...                                                     % proj dim = 32 x 32 x 64 = 65536
    ... ... 'custom-5-L-7-64-16-relu', ...                                                 % proj dim = 32 x 32 x 64 = 65536
    ... 'custom-5-L-7-64-16-max-pool', ...                                             % proj dim = 32 x 32 x 64 / (4 ^ 5) = 64
    ... 'custom-5-L-7-64-16-relu-max-pool', ...                                        % proj dim = 32 x 32 x 64 / (4 ^ 5) = 64
    ... ... 'custom-5-L-11-64-16-', ...                                                    % proj dim = 32 x 32 x 64 = 65536
    ... ... 'custom-5-L-11-64-16-relu', ...                                                % proj dim = 32 x 32 x 64 = 65536
    ... 'custom-5-L-11-64-16-max-pool', ...                                            % proj dim = 32 x 32 x 64 / (4 ^ 5) = 64
    ... 'custom-5-L-11-64-16-relu-max-pool', ...                                       % proj dim = 32 x 32 x 64 / (4 ^ 5) = 64
    ... ... ...
    ... ... 'custom-5-L-3-256-16-', ...                                                    % proj dim = 32 x 32 x 256 = 262144
    ... ... 'custom-5-L-3-256-16-relu', ...                                                % proj dim = 32 x 32 x 256 = 262144
    ... 'custom-5-L-3-256-16-max-pool', ...                                            % proj dim = 32 x 32 x 256 / (4 ^ 5) = 256
    ... 'custom-5-L-3-256-16-relu-max-pool', ...                                       % proj dim = 32 x 32 x 256 / (4 ^ 5) = 256
    ... ... 'custom-5-L-7-256-16-', ...                                                    % proj dim = 32 x 32 x 256 = 262144
    ... ... 'custom-5-L-7-256-16-relu', ...                                                % proj dim = 32 x 32 x 256 = 262144
    ... 'custom-5-L-7-256-16-max-pool', ...                                            % proj dim = 32 x 32 x 256 / (4 ^ 5) = 256
    ... 'custom-5-L-7-256-16-relu-max-pool', ...                                       % proj dim = 32 x 32 x 256 / (4 ^ 5) = 256
    ... ... 'custom-5-L-11-256-16-', ...                                                   % proj dim = 32 x 32 x 256 = 262144
    ... ... 'custom-5-L-11-256-16-relu', ...                                               % proj dim = 32 x 32 x 256 = 262144
    ... 'custom-5-L-11-256-16-max-pool', ...                                           % proj dim = 32 x 32 x 256 / (4 ^ 5) = 256
    ... 'custom-5-L-11-256-16-relu-max-pool', ...                                      % proj dim = 32 x 32 x 256 / (4 ^ 5) = 256
    ... ... ...
    ... ... 'custom-5-L-3-1024-16-', ...                                                   % proj dim = 32 x 32 x 1024 = 1048576
    ... ... 'custom-5-L-3-1024-16-relu', ...                                               % proj dim = 32 x 32 x 1024 = 1048576
    ... 'custom-5-L-3-1024-16-max-pool', ...                                           % proj dim = 32 x 32 x 1024 / (4 ^ 5) = 1024
    ... 'custom-5-L-3-1024-16-relu-max-pool', ...                                       % proj dim = 32 x 32 x 1024 / (4 ^ 5) = 1024
    ... ... 'custom-5-L-7-1024-16-', ...                                                   % proj dim = 32 x 32 x 1024 = 1048576
    ... ... 'custom-5-L-7-1024-16-relu', ...                                               % proj dim = 32 x 32 x 1024 = 1048576
    ... 'custom-5-L-7-1024-16-max-pool', ...                                           % proj dim = 32 x 32 x 1024 / (4 ^ 5) = 1024
    ... 'custom-5-L-7-1024-16-relu-max-pool', ...                                       % proj dim = 32 x 32 x 1024 / (4 ^ 5) = 1024
    ... ... 'custom-5-L-11-1024-16-', ...                                                  % proj dim = 32 x 32 x 1024 = 1048576
    ... ... 'custom-5-L-11-1024-16-relu', ...                                              % proj dim = 32 x 32 x 1024 = 1048576
    ... 'custom-5-L-11-1024-16-max-pool', ...                                          % proj dim = 32 x 32 x 1024 / (4 ^ 5) = 1024
    ... 'custom-5-L-11-1024-16-relu-max-pool', ...                                      % proj dim = 32 x 32 x 1024 / (4 ^ 5) = 1024
    ... ...
    ... ... ... 'custom-5-L-3-4096-16-', ...                                                   % proj dim = 32 x 32 x 4096 = 4194304
    ... ... ... 'custom-5-L-3-4096-16-relu', ...                                               % proj dim = 32 x 32 x 4096 = 4194304
    ... ... 'custom-5-L-3-4096-16-max-pool', ...                                            % proj dim = 32 x 32 x 4096 / (4 ^ 5) = 4096
    ... ... 'custom-5-L-3-4096-16-relu-max-pool', ...                                       % proj dim = 32 x 32 x 4096 / (4 ^ 5) = 4096
    ... ... ... 'custom-5-L-7-4096-16-', ...                                                   % proj dim = 32 x 32 x 4096 = 4194304
    ... ... ... 'custom-5-L-7-4096-16-relu', ...                                               % proj dim = 32 x 32 x 4096 = 4194304
    ... ... 'custom-5-L-7-4096-16-max-pool', ...                                            % proj dim = 32 x 32 x 4096 / (4 ^ 5) = 4096
    ... ... 'custom-5-L-7-4096-16-relu-max-pool', ...                                       % proj dim = 32 x 32 x 4096 / (4 ^ 5) = 4096
    ... ... ... 'custom-5-L-11-4096-16-', ...                                                  % proj dim = 32 x 32 x 4096 = 4194304
    ... ... ... 'custom-5-L-11-4096-16-relu', ...                                              % proj dim = 32 x 32 x 4096 = 4194304
    ... ... 'custom-5-L-11-4096-16-max-pool', ...                                           % proj dim = 32 x 32 x 4096 / (4 ^ 5) = 4096
    ... ... 'custom-5-L-11-4096-16-relu-max-pool', ...                                      % proj dim = 32 x 32 x 4096 / (4 ^ 5) = 4096
  };




















