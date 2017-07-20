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
  % classification_method = 'mlp-64-10';
  % classification_method = 'mlp-500-100';
  % classification_method = 'mlp-500-1000-100';
  repeat_count = 5;
  all_experiments_multi_run = {};

  for i = 1 : 22
    all_experiments_multi_run{i}.performance = [];
  end

  for kk = 1:repeat_count
    all_experiments_single_run = runAllExperimentsOnce(dataset, posneg_balance, classification_method);
    % for i = 1 : numel(all_experiments_single_run)
    %   all_experiments_multi_run{i}.performance(end + 1) = ...
    %     all_experiments_single_run{i}.performance_summary.testing.test.accuracy;
    % end
    for i = 1 : numel(all_experiments_single_run)
      all_experiments_multi_run{i}.performance(end + 1) = all_experiments_single_run{i}.test_accuracy;
    end
  end

  plot_title = sprintf('classification perf - %s - %s - %s', classification_method, dataset, posneg_balance);
  tempScriptPlotRPTests(all_experiments_multi_run, plot_title, save_results);

% -------------------------------------------------------------------------
function all_experiments_single_run = runAllExperimentsOnce(dataset, posneg_balance, classification_method)
% -------------------------------------------------------------------------
  % opts.general.dataset = dataset;
  % opts.general.posneg_balance = posneg_balance;
  % [~, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, 0);

  % % -------------------------------------------------------------------------
  % %                                                                opts.paths
  % % -------------------------------------------------------------------------
  % opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  % opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
  %   {}, ... % no input_opts here! :)
  %   'experiment_parent_dir', ...
  %   fullfile(vl_rootnn, 'experiment_results'));
  % opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
  %   'test-classification-perf-rp-tests-%s-%s-%s-GPU-%d', ...
  %   opts.paths.time_string, ...
  %   opts.general.dataset, ...
  %   opts.general.posneg_balance));
  % if ~exist(opts.paths.experiment_dir)
  %   mkdir(opts.paths.experiment_dir);
  % end
  % opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, 'options.txt');




  [~, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, 0);
  for i = 1 : numel(experiments)

    tmp_imdb = experiments{i}.imdb;
    switch classification_method
      case '1-knn'
        test_accuracy = getSimpleTestAccuracyFromKnn(tmp_imdb, 1);
      case '3-knn'
        test_accuracy = getSimpleTestAccuracyFromKnn(tmp_imdb, 3);
      case 'mlp-64-10'
        test_accuracy = getSimpleTestAccuracyFromMLP(tmp_imdb, [64, 10]);
      case 'mlp-500-100'
        test_accuracy = getSimpleTestAccuracyFromMLP(tmp_imdb, [500, 100]);
      case 'mlp-500-1000-100'
        test_accuracy = getSimpleTestAccuracyFromMLP(tmp_imdb, [500, 1000, 100]);
      case 'cnn'
        gpu = 1;
        [best_test_accuracy_mean, best_test_accuracy_std] = getSimpleTestAccuracyFromCnn( ...
          dataset, ...
          posneg_balance, ...
          tmp_imdb, ...
          'convV0P0RL0+fcV1-RF16CH64', ... % TODO: this has to somehow be detected automatically....
          gpu);
        test_accuracy = best_test_accuracy_mean;
    end
    experiments{i}.test_accuracy = test_accuracy;

  end



  % % -------------------------------------------------------------------------
  % %                                       opts.single_training_method_options
  % % -------------------------------------------------------------------------
  % opts.single_training_method_options.experiment_parent_dir = opts.paths.experiment_dir;
  % opts.single_training_method_options.dataset = dataset;
  % opts.single_training_method_options.return_performance_summary = true;
  % switch classification_method
  %   case '1-knn'
  %     classificationMethodFunctonHandle = @testKnn;
  %     opts.single_training_method_options.number_of_nearest_neighbors = 1;
  %   case '3-knn'
  %     classificationMethodFunctonHandle = @testKnn;
  %     opts.single_training_method_options.number_of_nearest_neighbors = 3;
  %   case 'mlp-64-10'
  %     classificationMethodFunctonHandle = @testMlp;
  %     opts.single_training_method_options.number_of_hidden_nodes = [64, 10];
  %   case 'mlp-500-100'
  %     classificationMethodFunctonHandle = @testMlp;
  %     opts.single_training_method_options.number_of_hidden_nodes = [500, 100];
  %   case 'mlp-500-1000-100'
  %     classificationMethodFunctonHandle = @testMlp;
  %     opts.single_training_method_options.number_of_hidden_nodes = [500, 1000, 100];
  %   case 'cnn'
  %     classificationMethodFunctonHandle = @testCnn;
  %     opts.single_training_method_options.network_arch = 'convV0P0RL0+fcV1-RF32CH3';
  %     opts.single_training_method_options.backprop_depth = 4;
  %     opts.single_training_method_options.gpus = ifNotMacSetGpu(1);
  %     opts.single_training_method_options.debug_flag = false;
  %     opts.single_training_method_options.learning_rate = [0.1*ones(1,15) 0.03*ones(1,15) 0.01*ones(1,15)];
  %     % opts.single_training_method_options.learning_rate = [0.1*ones(1,3)];
  %     opts.single_training_method_options.weight_decay = 0.0001;
  %     opts.single_training_method_options.batch_size = 50;
  % end

  % % -------------------------------------------------------------------------
  % %                                                    save experiment setup!
  % % -------------------------------------------------------------------------
  % saveStruct2File(opts, opts.paths.options_file_path, 0);

  % for i = 1 : numel(experiments)
  %   opts.single_training_method_options.imdb = experiments{i}.imdb;
  %   [~, experiments{i}.performance_summary] = classificationMethodFunctonHandle(opts.single_training_method_options);
  %   % all_experiments_repeated{i}.performance_summary.testing.test.accuracy = []
  % end

  % for i = 1 : numel(experiments)
  %   afprintf(sprintf( ...
  %     '[INFO] 1-KNN Results for `%s`: \t\t train acc = %.4f, test acc = %.4f \n\n', ...
  %     experiments{i}.title, ...
  %     experiments{i}.performance_summary.testing.train.accuracy, ...
  %     experiments{i}.performance_summary.testing.test.accuracy));
  % end

  % all_experiments_single_run = experiments;









