% -------------------------------------------------------------------------
function tempScriptMeasureClassificationPerformance(dataset, posneg_balance, save_results)
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
  classification_method = '1-knn';
  % classification_method = '3-knn';
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
    classification_method, ...
    opts.paths.time_string, ...
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
  [~, tmp_experiments] = setupExperimentsUsingProjectedImbds(opts.general.dataset, opts.general.posneg_balance, 0);
  for i = 1 : numel(tmp_experiments)
    all_experiments_multi_run{i}.performance = [];
  end

  for kk = 1:number_of_trials
    afprintf(sprintf('[INFO] Testing trial #%d / %d ...\n', kk, number_of_trials));
    all_experiments_single_run = runAllExperimentsOnce(opts.paths.experiment_dir, dataset, posneg_balance, classification_method);
    for i = 1 : numel(all_experiments_single_run)
      all_experiments_multi_run{i}.performance(end+1) = all_experiments_single_run{i}.test_accuracy;
    end
    afprintf(sprintf('[INFO] done!'));

    for i = 1 : numel(tmp_experiments)
      all_experiments_multi_run{i}.performance_mean = mean(all_experiments_multi_run{i}.performance);
      all_experiments_multi_run{i}.performance_std = std(all_experiments_multi_run{i}.performance);
    end

    % -------------------------------------------------------------------------
    %                                                               save output
    % -------------------------------------------------------------------------
    % don't amend file, but overwrite...
    delete(opts.paths.results_file_path);
    saveStruct2File(all_experiments_multi_run, opts.paths.results_file_path, 0);
  end

  % plot_title = sprintf('classification perf - %s - %s - %s', classification_method, dataset, posneg_balance);
  % tempScriptPlotRPTests(all_experiments_multi_run, plot_title, save_results);








% -------------------------------------------------------------------------
function all_experiments_single_run = runAllExperimentsOnce(experiment_dir, dataset, posneg_balance, classification_method)
% -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Setting up experiment imdbs...'));
  [~, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, 0);
  fprintf('done!\n');
  for i = 1 : numel(experiments)
    experiment_options = {};
    experiment_options.imdb = experiments{i}.imdb;
    experiment_options.dataset = dataset;
    experiment_options.posneg_balance = posneg_balance;
    experiment_options.experiment_parent_dir = experiment_dir;
    experiment_options.debug_flag = false;
    switch classification_method
      case '1-knn'
        experiment_options.number_of_nearest_neighbors = 1;
        test_accuracy = getSimpleTestAccuracyFromKnn(experiment_options);
      case '3-knn'
        experiment_options.number_of_nearest_neighbors = 3;
        test_accuracy = getSimpleTestAccuracyFromKnn(experiment_options);
      case 'libsvm'
        test_accuracy = getSimpleTestAccuracyFromLibSvm(experiment_options);
      case 'mlp-64-10'
        experiment_options.number_of_hidden_nodes = [64, 10];
        test_accuracy = getSimpleTestAccuracyFromMLP(experiment_options);
      case 'mlp-500-100'
        experiment_options.number_of_hidden_nodes = [500, 100];
        test_accuracy = getSimpleTestAccuracyFromMLP(experiment_options);
      case 'mlp-500-1000-100'
        experiment_options.number_of_hidden_nodes = [500, 1000, 100];
        test_accuracy = getSimpleTestAccuracyFromMLP(experiment_options);
      case 'cnn'

        experiment_options.gpus = 2;

        % TODO: this has to somehow be detected automatically....
        % experiment_options.conv_network_arch = 'convV0P0RL0+fcV1-RF16CH64';
        experiment_options.conv_network_arch = 'convV1P1RL1-RF32CH3+fcV1-RF16CH64';

        [best_test_accuracy_mean, best_test_accuracy_std] = getSimpleTestAccuracyFromCnn(experiment_options);
        test_accuracy = best_test_accuracy_mean;
    end
    experiments{i}.test_accuracy = test_accuracy;
  end

  all_experiments_single_run = experiments;


























