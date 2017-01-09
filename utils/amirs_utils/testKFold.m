% -------------------------------------------------------------------------
function folds = testKFold(input_opts)
% -------------------------------------------------------------------------
  % -------------------------------------------------------------------------
  %                                                              opts.general
  % -------------------------------------------------------------------------
  opts.general.training_method = getValueFromFieldOrDefault(input_opts, 'training_method', 'ensemble-cnn');
  opts.general.dataset = getValueFromFieldOrDefault(input_opts, 'dataset', 'mnist-two-class-9-4');
  opts.general.network_arch = 'lenet';
  opts.general.number_of_folds = getValueFromFieldOrDefault(input_opts, 'number_of_folds', 5);

  % -------------------------------------------------------------------------
  %                                                                 opts.imdb
  % -------------------------------------------------------------------------
  opts.imdb.posneg_balance = getValueFromFieldOrDefault(input_opts, 'posneg_balance', 'unbalanced');
  if strcmp(opts.general.dataset, 'prostate-v2-20-patients')
    switch opts.imdb.posneg_balance
      case 'k=5-fold-unbalanced'
        opts.general.number_of_folds == 5
      case 'k=5-fold-balanced-high'
        opts.general.number_of_folds == 5
      case 'leave-one-out-unbalanced'
        opts.general.number_of_folds = 20;
      case 'leave-one-out-balanced-high'
        opts.general.number_of_folds = 20;
      otherwise
        assert(false);
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
    'k=%d-fold-%s-%s', ...
    opts.general.number_of_folds, ...
    opts.general.training_method, ...
    opts.paths.time_string));
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
  switch opts.general.training_method
    case 'single-cnn'
      opts.single_training_method_options.backprop_depth = getValueFromFieldOrDefault(input_opts, 'backprop_depth', 4);
      opts.single_training_method_options.gpus = ifNotMacSetGpu(getValueFromFieldOrDefault(input_opts, 'gpus', 1));
    case 'ensemble-cnn'
      opts.single_training_method_options.network_arch = opts.general.network_arch;
      opts.single_training_method_options.iteration_count = getValueFromFieldOrDefault(input_opts, 'iteration_count', 10);
      opts.single_training_method_options.gpus = ifNotMacSetGpu(getValueFromFieldOrDefault(input_opts, 'gpus', 1));
      opts.single_training_method_options.backprop_depth = getValueFromFieldOrDefault(input_opts, 'backprop_depth', 4);
      opts.single_training_method_options.symmetric_weight_updates = getValueFromFieldOrDefault(input_opts, 'symmetric_weight_updates', false);
      opts.single_training_method_options.symmetric_loss_updates = getValueFromFieldOrDefault(input_opts, 'symmetric_loss_updates', false);
    case 'forest'
      opts.single_training_method_options.boosting_method = getValueFromFieldOrDefault(input_opts, 'boosting_method', 'RUSBoost');
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
    opts.general.training_method, ...
    opts.general.number_of_folds), 1);

  % -------------------------------------------------------------------------
  %                                             create the imdb for each fold
  % -------------------------------------------------------------------------
  imdbs = {}; % separate so don't have to save ~1.5 GB of imdbs!!!

  for i = 1:opts.general.number_of_folds
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

  switch opts.general.training_method
    case 'single-cnn'
      trainingMethodFunctionHandle = @testSingleNetwork;
    case 'ensemble-cnn'
      trainingMethodFunctionHandle = @cnnRusboost;
    case 'forest'
      trainingMethodFunctionHandle = @testForest;
  end

  for i = 1:opts.general.number_of_folds
    afprintf(sprintf('[INFO] Running `%s` on fold #%d...\n', opts.general.training_method, i));
    opts.single_training_method_options.imdb = imdbs{i};
    folds.(sprintf('fold_%d', i)).performance_summary = ...
      trainingMethodFunctionHandle(opts.single_training_method_options);
    % overwrite and save results so far
    save(opts.paths.folds_file_path, 'folds');
    saveKFoldResults(folds, opts.paths.results_file_path);
  end


% -------------------------------------------------------------------------
function saveKFoldResults(folds, results_file_path)
% -------------------------------------------------------------------------
  number_of_folds = numel(fields(folds));
  for i = 1:number_of_folds
    performance_summary_for_fold = folds.(sprintf('fold_%d', i)).performance_summary;
    % copy all fields; mandatory fields:
    %  * weighted_test_accuracy
    %  * weighted_test_sensitivity
    %  * weighted_test_specificity
    for fn = fieldnames(performance_summary_for_fold)'
      k_fold_results.(sprintf('fold_%d', i)).(fn{1}) = performance_summary_for_fold.(fn{1});
    end
  end

  all_folds_accuracy = [];
  all_folds_sensitivity = [];
  all_folds_specificity = [];
  for i = 1:number_of_folds
    all_folds_accuracy(i) = k_fold_results.(sprintf('fold_%d', i)).weighted_test_accuracy;
    all_folds_sensitivity(i) = k_fold_results.(sprintf('fold_%d', i)).weighted_test_sensitivity;
    all_folds_specificity(i) = k_fold_results.(sprintf('fold_%d', i)).weighted_test_specificity;
  end

  all_folds_accuracy = all_folds_accuracy(~isnan(all_folds_accuracy));
  all_folds_sensitivity = all_folds_sensitivity(~isnan(all_folds_sensitivity));
  all_folds_specificity = all_folds_specificity(~isnan(all_folds_specificity));

  k_fold_results.kfold_accuracy_avg = mean(all_folds_accuracy);
  k_fold_results.kfold_sensitivity_avg = mean(all_folds_sensitivity);
  k_fold_results.kfold_specificity_avg = mean(all_folds_specificity);
  k_fold_results.kfold_accuracy_std = std(all_folds_accuracy);
  k_fold_results.kfold_sensitivity_std = std(all_folds_sensitivity);
  k_fold_results.kfold_specificity_std = std(all_folds_specificity);

  % don't amend file, but overwrite...
  delete(results_file_path);
  saveStruct2File(k_fold_results, results_file_path, 0);
