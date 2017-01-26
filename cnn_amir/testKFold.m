% -------------------------------------------------------------------------
function folds = testKFold(input_opts)
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

  opts.general.dataset = getValueFromFieldOrDefault(input_opts, 'dataset', 'mnist-two-class-9-4');

  % -------------------------------------------------------------------------
  %                                                       opts.k_fold_options
  % -------------------------------------------------------------------------
  opts.k_fold_options.training_method = getValueFromFieldOrDefault(input_opts, 'training_method', 'ensemble-cnn');
  opts.k_fold_options.number_of_folds = getValueFromFieldOrDefault(input_opts, 'number_of_folds', 5);

  % -------------------------------------------------------------------------
  %                                                                 opts.imdb
  % -------------------------------------------------------------------------
  opts.imdb.posneg_balance = getValueFromFieldOrDefault(input_opts, 'posneg_balance', 'unbalanced');
  if strcmp(opts.general.dataset, 'prostate-v2-20-patients')
    switch opts.imdb.posneg_balance
      case 'k=5-fold-unbalanced'
        opts.k_fold_options.number_of_folds == 5;
      case 'k=5-fold-balanced-high'
        opts.k_fold_options.number_of_folds == 5;
      case 'leave-one-out-balanced-640-640'
        opts.k_fold_options.number_of_folds = 20;
      case 'leave-one-out-balanced-1280-1280'
        opts.k_fold_options.number_of_folds = 20;
      case 'leave-one-out-balanced-low'
        opts.k_fold_options.number_of_folds = 20;
      case 'leave-one-out-unbalanced'
        opts.k_fold_options.number_of_folds = 20;
      case 'leave-one-out-balanced-high'
        opts.k_fold_options.number_of_folds = 20;
      otherwise
        assert(false);
    end
  elseif strcmp(opts.general.dataset, 'prostate-v3-104-patients')
    switch opts.imdb.posneg_balance
      case 'leave-one-out-balanced-low'
        opts.k_fold_options.number_of_folds = 104;
      case 'leave-one-out-unbalanced'
        opts.k_fold_options.number_of_folds = 104;
      case 'leave-one-out-balanced-high'
        opts.k_fold_options.number_of_folds = 104;
    end
  end

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
    input_opts, ...
    'experiment_parent_dir', ...
    fullfile(vl_rootnn, 'experiment_results'));
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'k=%d-fold-%s-%s-%s', ...
    opts.k_fold_options.number_of_folds, ...
    opts.general.dataset, ...
    opts.paths.time_string, ...
    opts.k_fold_options.training_method));
  if ~exist(opts.paths.experiment_dir)
    mkdir(opts.paths.experiment_dir);
  end
  opts.paths.folds_file_path = fullfile(opts.paths.experiment_dir, 'folds.mat');
  opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, 'options.txt');
  opts.paths.results_file_path = fullfile(opts.paths.experiment_dir, 'results.txt');

  % -------------------------------------------------------------------------
  %                                       opts.single_training_method_options
  % -------------------------------------------------------------------------
  opts.single_training_method_options.dataset = opts.general.dataset;
  opts.single_training_method_options.experiment_parent_dir = opts.paths.experiment_dir;
  switch opts.k_fold_options.training_method
    case 'svm'
      % no additional options
    case 'forest'
      opts.single_training_method_options.boosting_method = getValueFromFieldOrDefault(input_opts, 'boosting_method', 'RUSBoost');
    case 'single-cnn'
      opts.single_training_method_options.network_arch = getValueFromFieldOrDefault(input_opts, 'network_arch', 'lenet');
      opts.single_training_method_options.backprop_depth = getValueFromFieldOrDefault(input_opts, 'backprop_depth', 4);
      opts.single_training_method_options.gpus = ifNotMacSetGpu(getValueFromFieldOrDefault(input_opts, 'gpus', 1));
      opts.single_training_method_options.debug_flag = getValueFromFieldOrDefault(input_opts, 'debug_flag', false);
      opts.single_training_method_options.learning_rate = getValueFromFieldOrDefault(input_opts, 'learning_rate', [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20)]);
    case 'ensemble-cnn'
      % ensemble options
      opts.single_training_method_options.boosting_method = getValueFromFieldOrDefault(input_opts, 'boosting_method', 'rusboost');
      opts.single_training_method_options.iteration_count = getValueFromFieldOrDefault(input_opts, 'iteration_count', 5);
      opts.single_training_method_options.number_of_samples_per_model = getValueFromFieldOrDefault(input_opts, 'number_of_samples_per_model', 1000);
      opts.single_training_method_options.loss_calculation_method = getValueFromFieldOrDefault(input_opts, 'loss_calculation_method', 'default_in_literature');
      opts.single_training_method_options.training_method = 'cnn';
      % cnn options
      opts.single_training_method_options.network_arch = getValueFromFieldOrDefault(input_opts, 'network_arch', 'lenet');
      opts.single_training_method_options.backprop_depth = getValueFromFieldOrDefault(input_opts, 'backprop_depth', 4);
      opts.single_training_method_options.gpus = ifNotMacSetGpu(getValueFromFieldOrDefault(input_opts, 'gpus', 1));
      opts.single_training_method_options.debug_flag = getValueFromFieldOrDefault(input_opts, 'debug_flag', false);
    case 'ensemble-svm'
      % ensemble options
      opts.single_training_method_options.boosting_method = getValueFromFieldOrDefault(input_opts, 'boosting_method', 'rusboost');
      opts.single_training_method_options.iteration_count = getValueFromFieldOrDefault(input_opts, 'iteration_count', 5);
      opts.single_training_method_options.number_of_samples_per_model = getValueFromFieldOrDefault(input_opts, 'number_of_samples_per_model', 1000);
      opts.single_training_method_options.loss_calculation_method = getValueFromFieldOrDefault(input_opts, 'loss_calculation_method', 'default_in_literature');
      opts.single_training_method_options.training_method = 'svm';
      % svm options
      % no additional options
  end

  % -------------------------------------------------------------------------
  %                                                    save experiment setup!
  % -------------------------------------------------------------------------
  saveStruct2File(opts, opts.paths.options_file_path, 0);

  % -------------------------------------------------------------------------
  %                                                                     start
  % -------------------------------------------------------------------------
  afprintf(sprintf( ...
    '[INFO] Running K-fold `%s` (K = %d)...\n', ...
    opts.k_fold_options.training_method, ...
    opts.k_fold_options.number_of_folds), 1);

  % -------------------------------------------------------------------------
  %                                             create the imdb for each fold
  % -------------------------------------------------------------------------
  imdbs = {}; % separate so don't have to save ~1.5 GB of imdbs!!!

  for i = 1:opts.k_fold_options.number_of_folds
    afprintf(sprintf('\n'));
    afprintf(sprintf('[INFO] Loading imdb for fold #%d...\n', i));
    tmp_opts.dataset = opts.general.dataset;
    tmp_opts.posneg_balance = opts.imdb.posneg_balance;
    tmp_opts.fold_number = i; % currently only implemented for prostate data
    imdbs{i} = loadSavedImdb(tmp_opts);
    afprintf(sprintf('[INFO] done!\n'));
  end

  % -------------------------------------------------------------------------
  %                                                      train for each fold!
  % -------------------------------------------------------------------------
  switch opts.k_fold_options.training_method
    case 'svm'
      trainingMethodFunctionHandle = @testSvm;
    case 'forest'
      trainingMethodFunctionHandle = @testForest;
    case 'single-cnn'
      trainingMethodFunctionHandle = @testCnn;
    case 'ensemble-cnn'
      trainingMethodFunctionHandle = @testEnsemble;
      % switch getValueFromFieldOrDefault(input_opts, 'ensemble_version', 'v1');
      %   case 'v1'
      %     trainingMethodFunctionHandle = @testEnsemble1;
      %   case 'v2'
      %     trainingMethodFunctionHandle = @testEnsemble2;
      %   case 'v3'
      %     trainingMethodFunctionHandle = @testEnsemble3;
      % end
    case 'ensemble-svm'
      trainingMethodFunctionHandle = @testEnsemble;
      % switch getValueFromFieldOrDefault(input_opts, 'ensemble_version', 'v1');
      %   case 'v1'
      %     trainingMethodFunctionHandle = @testEnsemble1;
      %   case 'v2'
      %     trainingMethodFunctionHandle = @testEnsemble2;
      %   case 'v3'
      %     trainingMethodFunctionHandle = @testEnsemble3;
      % end
  end

  for i = 1:opts.k_fold_options.number_of_folds
    % -------------------------------------------------------------------------
    %                                                                train fold
    % -------------------------------------------------------------------------
    afprintf(sprintf('[INFO] Running `%s` on fold #%d...\n', opts.k_fold_options.training_method, i));
    opts.single_training_method_options.imdb = imdbs{i};
    [ ...
      trained_model, ...
      performance_summary, ...
    ] = trainingMethodFunctionHandle(opts.single_training_method_options);
    afprintf(sprintf('[INFO] done!\n'));

    % -------------------------------------------------------------------------
    %                        save / overwrite incremental performance summaries
    % -------------------------------------------------------------------------
    afprintf(sprintf('[INFO] Saving incremental k-fold results...\n'));
    folds.(sprintf('fold_%d', i)).performance_summary = performance_summary;
    saveIncrementalKFoldResults(folds, opts.paths.results_file_path);
    save(opts.paths.folds_file_path, 'folds');
    afprintf(sprintf('[INFO] done!\n\n'));
  end

% -------------------------------------------------------------------------
function saveIncrementalKFoldResults(folds, results_file_path)
% -------------------------------------------------------------------------
  number_of_folds = numel(fields(folds));
  k_fold_results = {};
  for i = 1:number_of_folds
    performance_summary_for_fold = folds.(sprintf('fold_%d', i)).performance_summary;
    % copy all fields; mandatory fields:
    %  * train.accuracy
    %  * train.sensitivity
    %  * train.specificity
    %  * test.accuracy
    %  * test.sensitivity
    %  * test.specificity
    for fn = fieldnames(performance_summary_for_fold)'
      k_fold_results.(sprintf('fold_%d', i)).(fn{1}) = performance_summary_for_fold.(fn{1});
    end
    % k_fold_results.(sprintf('fold_%d', i)) = {};
    % mergeStructs(k_fold_results.(sprintf('fold_%d', i)), performance_summary_for_fold);
  end

  for set = {'train', 'test'}
    set = char(set);
    all_folds_accuracy = [];
    all_folds_sensitivity = [];
    all_folds_specificity = [];
    for i = 1:number_of_folds
      all_folds_accuracy(i) = k_fold_results.(sprintf('fold_%d', i)).(set).accuracy;
      all_folds_sensitivity(i) = k_fold_results.(sprintf('fold_%d', i)).(set).sensitivity;
      all_folds_specificity(i) = k_fold_results.(sprintf('fold_%d', i)).(set).specificity;
    end

    % sanitize
    all_folds_accuracy = all_folds_accuracy(~isnan(all_folds_accuracy));
    all_folds_sensitivity = all_folds_sensitivity(~isnan(all_folds_sensitivity));
    all_folds_specificity = all_folds_specificity(~isnan(all_folds_specificity));

    % save avg and std
    k_fold_results.k_fold.(set).accuracy_avg = mean(all_folds_accuracy);
    k_fold_results.k_fold.(set).sensitivity_avg = mean(all_folds_sensitivity);
    k_fold_results.k_fold.(set).specificity_avg = mean(all_folds_specificity);
    k_fold_results.k_fold.(set).accuracy_std = std(all_folds_accuracy);
    k_fold_results.k_fold.(set).sensitivity_std = std(all_folds_sensitivity);
    k_fold_results.k_fold.(set).specificity_std = std(all_folds_specificity);
  end

  % don't amend file, but overwrite...
  delete(results_file_path);
  saveStruct2File(k_fold_results, results_file_path, 0);
