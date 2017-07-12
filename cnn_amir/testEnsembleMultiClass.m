% -------------------------------------------------------------------------
function [trained_model, performance_summary] = testEnsemble(input_opts)
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
  opts.general.dataset = getValueFromFieldOrDefault(input_opts, 'dataset', 'mnist');
  opts.general.return_performance_summary = getValueFromFieldOrDefault(input_opts, 'return_performance_summary', true);

  % -------------------------------------------------------------------------
  %                                                     opts.ensemble_options
  % -------------------------------------------------------------------------
  % TODO... implement boosting methods in addition to RUSBoost
  opts.ensemble_options.boosting_method = getValueFromFieldOrDefault(input_opts, 'boosting_method', 'rusboost'); % {'adaboost.m1', 'rusboost', ...}
  opts.ensemble_options.uni_model_boosting = getValueFromFieldOrDefault(input_opts, 'uni_model_boosting', false);
  opts.ensemble_options.training_method = getValueFromFieldOrDefault(input_opts, 'training_method', 'cnn');
  opts.ensemble_options.loss_calculation_method = getValueFromFieldOrDefault(input_opts, 'loss_calculation_method', 'default_in_literature');
  opts.ensemble_options.iteration_count = getValueFromFieldOrDefault(input_opts, 'iteration_count', 5);
  opts.ensemble_options.number_of_samples_per_model = getValueFromFieldOrDefault(input_opts, 'number_of_samples_per_model', 1000);

  switch opts.ensemble_options.boosting_method
    case 'adaboost.m1'
      opts.ensemble_options.resampling_method = 'ada';
    case 'rusboost'
      opts.ensemble_options.resampling_method = 'rus';
    % TODO
    % case 'smoteboost'
  end

  % -------------------------------------------------------------------------
  %                                                                 opts.imdb
  % -------------------------------------------------------------------------
  imdb = getValueFromFieldOrDefault(input_opts, 'imdb', struct());

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
    input_opts, ...
    'experiment_parent_dir', ...
    fullfile(vl_rootnn, 'experiment_results'));
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'ensemble-%s-%s-%s-%s-loss-%s-%d-samples-max-%d-iters', ...
    opts.paths.time_string, ...
    opts.general.dataset, ...
    opts.ensemble_options.boosting_method, ...
    opts.ensemble_options.training_method, ...
    opts.ensemble_options.loss_calculation_method, ...
    opts.ensemble_options.number_of_samples_per_model, ...
    opts.ensemble_options.iteration_count));
  if ~exist(opts.paths.experiment_dir)
    mkdir(opts.paths.experiment_dir);
  end
  opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, 'options.txt');
  opts.paths.results_file_path = fullfile(opts.paths.experiment_dir, 'results.txt');
  opts.paths.ensemble_models_file_path = fullfile(opts.paths.experiment_dir, 'ensemble_models.mat');

  % -------------------------------------------------------------------------
  %                                                 opts.single_model_options
  % -------------------------------------------------------------------------
  opts.single_model_options.dataset = opts.general.dataset;
  opts.single_model_options.experiment_parent_dir = opts.paths.experiment_dir;
  opts.single_model_options.return_performance_summary = false;
  switch opts.ensemble_options.training_method
    case 'svm'
      % no additional options
    case 'cnn'
      opts.single_model_options.network_arch = getValueFromFieldOrDefault(input_opts, 'network_arch', 'lenet');
      opts.single_model_options.backprop_depth = getValueFromFieldOrDefault(input_opts, 'backprop_depth', 4);
      opts.single_model_options.gpus = ifNotMacSetGpu(getValueFromFieldOrDefault(input_opts, 'gpus', 1));
      opts.single_model_options.debug_flag = getValueFromFieldOrDefault(input_opts, 'debug_flag', false);
      opts.single_model_options.learning_rate = getValueFromFieldOrDefault(input_opts, 'learning_rate', 'default_keyword');
      opts.single_model_options.weight_init_sequence = getValueFromFieldOrDefault(input_opts, 'weight_init_sequence', {'gaussian', 'gaussian', 'gaussian'});
  end

  % -------------------------------------------------------------------------
  %                                                    save experiment setup!
  % -------------------------------------------------------------------------
  saveStruct2File(opts, opts.paths.options_file_path, 0);

  % -------------------------------------------------------------------------
  %                     1. process the imdb to separate positive and negative
  %                               samples (to be randomly-undersampled later)
  % -------------------------------------------------------------------------
  % TODO: can still use twoClassImdbUtils for multi-class imdbs if we ignore some outputs... but should move into it's own file...
  fh_imdb_utils = imdbTwoClassUtils;
  [ ...
    data, ...
    data_train, ...
    ~, ... % data_train_positive, ...
    ~, ... % data_train_negative, ...
    data_train_indices, ...
    ~, ... % data_train_positive_indices, ...
    ~, ... % data_train_negative_indices, ...
    data_train_count, ...
    ~, ... % data_train_positive_count, ...
    ~, ... % data_train_negative_count, ...
    labels_train, ...
    ~, ... % labels_train_positive, ...
    ~, ... % labels_train_negative, ...
    data_test, ...
    ~, ... % data_test_positive, ...
    ~, ... % data_test_negative, ...
    data_test_indices, ...
    ~, ... % data_test_positive_indices, ...
    ~, ... % data_test_negative_indices, ...
    data_test_count, ...
    ~, ... % data_test_positive_count, ...
    ~, ... % data_test_negative_count, ...
    labels_test, ...
    ~, ... % labels_test_positive, ...
    ~, ... % labels_test_negative, ...
  ] = fh_imdb_utils.getImdbInfo(imdb, 1);

  % -------------------------------------------------------------------------
  %                                     2. initialize training sample weights
  % -------------------------------------------------------------------------
  % W stores the weights of the instances in each row for every iteration of
  % boosting. Weights for all the instances are initialized by 1/m for the
  % first iteration.
  sample_weights = 1 / data_train_count * ones(1, data_train_count);

  % L stores pseudo loss values, H stores hypothesis, B stores (1/beta)
  % values that is used as the weight of the % hypothesis while forming the
  % final hypothesis. % All of the following are of length <=T and stores
  % values for every iteration of the boosting process.
  loss_per_iteration = [];
  model_per_iteration = {};
  model_weight_per_iteration = [];

  iteration = 1; % loop counter
  count = 1; % number of times the same boosting iteration have been repeated

  % -------------------------------------------------------------------------
  %                       3. create training (barebones) and validation imdbs
  % -------------------------------------------------------------------------
  initial_imdb_with_all_train_and_all_test = imdb;
  training_resampled_imdb = fh_imdb_utils.constructPartialImdb([], [], 3); % barebones; filled in below
  test_imdb = fh_imdb_utils.constructPartialImdb(data_test, labels_test, 3);

  % -------------------------------------------------------------------------
  %        4. go through T iterations of *boost, each training a single model
  % -------------------------------------------------------------------------
  printConsoleOutputSeparator();
  ensemble = struct();
  while iteration <= opts.ensemble_options.iteration_count
    afprintf(sprintf('\n'));
    printConsoleOutputSeparator();
    afprintf(sprintf('[INFO] Boosting iteration #%d (attempt %d)...\n', iteration, count), 1);
    printConsoleOutputSeparator();

    % -------------------------------------------------------------------------
    %                                       4.5. Resample data based on weights
    % -------------------------------------------------------------------------
    afprintf(sprintf(...
      '[INFO] Resampling positive and negative data (strategy: %s)... ', ...
      opts.ensemble_options.boosting_method));

    try
      % TODO: remove
      % [resampled_data, resampled_labels] = fh_imdb_utils.resampleTrainData( ...
      %   opts.ensemble_options.resampling_method, ...
      %   opts.ensemble_options.number_of_samples_per_model, ...
      %   data_train, ...
      %   labels_train, ...
      %   sample_weights(iteration, :));
      output_cap = opts.ensemble_options.number_of_samples_per_model;
      weighted_resample_indices = randsample( ...
        1:1:size(data_train, 4), ...
        output_cap, ...
        true, ...
        sample_weights(iteration, :));
      resampled_data = data_train(:,:,:, weighted_resample_indices);
      resampled_labels = labels_train(weighted_resample_indices);
    catch
      fprintf('\n');
      afprintf(sprintf('[INFO] Weights no longer large enough to take sample from; terminating!\n'));
      break;
    end
    fprintf('done!\n');

    training_resampled_imdb.images.data = single(resampled_data);
    training_resampled_imdb.images.labels = single(resampled_labels);
    training_resampled_imdb.images.set = 1 * ones(length(resampled_labels), 1);

    % -------------------------------------------------------------------------
    %                                                            5. Train model
    % -------------------------------------------------------------------------
    afprintf(sprintf('[INFO] Training model on `resampled train set`...\n'));
    opts.single_model_options.imdb = training_resampled_imdb;

    switch opts.ensemble_options.training_method
      case 'svm'
        [model, ~] = testEcocSvm(opts.single_model_options);
      case 'cnn'
        if opts.ensemble_options.uni_model_boosting && iteration > 1
          opts.single_model_options.net = model_per_iteration{iteration - 1};
        end
        [model, ~] = testCnn(opts.single_model_options);
    end

    % -------------------------------------------------------------------------
    %                          6. Compute the pseudo loss of hypothesis 'model'
    % -------------------------------------------------------------------------
    % IMPORTANT NOTE: we randomly undersample when training a model, but then,
    % we use all of the training samples (in their order) to update weights.
    afprintf(sprintf('[INFO] Computing `validation set` predictions...\n'));
    % NOTE: this asks for predictions on a cnn, whereas below we ask for predicitons on ensemble-{cnn, svm}
    % TODO: bug in all_validation_predictions!!
    [top_validation_predictions, ~] = ...
      getPredictionsFromModelOnImdb(model, opts.ensemble_options.training_method, initial_imdb_with_all_train_and_all_test, 1);

    [ ...
      validation_accuracy, ...
      validation_sensitivity, ...
      validation_specificity, ...
    ] = getAccSensSpecMultiClass(labels_train, top_validation_predictions, true);

    afprintf(sprintf('[INFO] Computing pseudo loss... '));
    loss = 0;
    switch opts.ensemble_options.loss_calculation_method
      case 'default_in_literature'
        for i = 1:data_train_count
          if labels_train(i) == top_validation_predictions(i)
            continue;
          else
            loss = loss + sample_weights(iteration, i);
          end
        end
      case 'class_normalized'
        loss = 0.5 * (1 - validation_sensitivity) + 0.5 * (1 - validation_specificity);
    end
    fprintf('Loss: %6.5f\n', loss);

    % If count exceeds a pre-defined threshold (5 in the current implementation)
    % the loop is broken and rolled back to the state where loss > 0.5 was not
    % encountered.
    if count > 5
      loss_per_iteration = loss_per_iteration(1:iteration-1);
      model_per_iteration = model_per_iteration(1:iteration-1);
      model_weight_per_iteration = model_weight_per_iteration(1:iteration-1);
      afprintf(sprintf('Too many iterations have loss > 0.5\n'));
      afprintf(sprintf('Aborting boosting...\n'));
      break;
    end

    % If the loss is greater than 1/2, it means that an inverted hypothesis
    % would perform better. In such cases, do not take that hypothesis into
    % consideration and repeat the same iteration. 'count' keeps counts of
    % the number of times the same boosting iteration have been repeated
    if loss > 0.5
      count = count + 1;
      continue;
    else
      count = 1;
    end

    model_per_iteration{iteration} = model; % Hypothesis function / trained model
    loss_per_iteration(iteration) = loss; % Pseudo-loss at each iteration
    beta = loss / (1 - loss); % Setting weight update parameter 'beta'.
    model_weight_per_iteration(iteration) = log(1 / beta); % Weight of the hypothesis

    % % At the final iteration there is no need to update the weights any
    % % further
    % if iteration == opts.ensemble_options.iteration_count
    %     break;
    % end

    % -------------------------------------------------------------------------
    %                                                          7. Update weight
    % -------------------------------------------------------------------------
    afprintf(sprintf('[INFO] Updating weights... '));
    for i = 1:data_train_count
      if labels_train(i) == top_validation_predictions(i)
        sample_weights(iteration + 1, i) = sample_weights(iteration, i) * beta;
      else
        sample_weights(iteration + 1, i) = sample_weights(iteration, i);
      end
    end
    fprintf('done!\n');

    % Normalizing the weight for the next iteration
    sum_W = sum(sample_weights(iteration + 1, :));
    for i = 1:data_train_count
      sample_weights(iteration + 1, i) = sample_weights(iteration + 1, i) / sum_W;
    end

    % -------------------------------------------------------------------------
    %                            8. Test on test set (single model of ensemble)
    % -------------------------------------------------------------------------
    % afprintf(sprintf('[INFO] Computing `test set` predictions (positive: %d, negative: %d)...\n', ...
    %   data_test_positive_count, ...
    %   data_test_negative_count));
    % NOTE: this asks for predictions on a cnn, whereas below we ask for predicitons on ensemble-{cnn, svm}
    [top_test_predictions, ~] = ...
      getPredictionsFromModelOnImdb(model, opts.ensemble_options.training_method, test_imdb, 3);
    [ ...
      test_accuracy, ...
      test_sensitivity, ...
      test_specificity, ...
    ] = getAccSensSpecMultiClass(labels_test, top_test_predictions, true);

    % -------------------------------------------------------------------------
    %                                          9. Save single model of ensemble
    % -------------------------------------------------------------------------
    afprintf(sprintf('[INFO] Saving model and info... '));
    ensemble.(sprintf('iteration_%d', iteration)).trained_model.model = model_per_iteration{iteration};
    ensemble.(sprintf('iteration_%d', iteration)).trained_model.loss = loss_per_iteration(iteration);
    ensemble.(sprintf('iteration_%d', iteration)).trained_model.weight_normalized = 0;
    ensemble.(sprintf('iteration_%d', iteration)).trained_model.weight_not_normalized = model_weight_per_iteration(iteration);
    % ensemble.(sprintf('iteration_%d', iteration)).trained_model.positive_count = numel(find(resampled_labels == 2));
    % ensemble.(sprintf('iteration_%d', iteration)).trained_model.negative_count = numel(find(resampled_labels == 1));
    % ensemble.(sprintf('iteration_%d', iteration)).validation.positive_count = data_train_positive_count;
    % ensemble.(sprintf('iteration_%d', iteration)).validation.negative_count = data_train_negative_count;
    ensemble.(sprintf('iteration_%d', iteration)).validation.top_predictions = top_validation_predictions;
    % ensemble.(sprintf('iteration_%d', iteration)).validation.all_predictions = all_validation_predictions;
    ensemble.(sprintf('iteration_%d', iteration)).validation.labels = labels_train;
    ensemble.(sprintf('iteration_%d', iteration)).validation.accuracy = validation_accuracy;
    ensemble.(sprintf('iteration_%d', iteration)).validation.sensitivity = validation_sensitivity;
    ensemble.(sprintf('iteration_%d', iteration)).validation.specificity = validation_specificity;
    % ensemble.(sprintf('iteration_%d', iteration)).test.positive_count = data_test_positive_count;
    % ensemble.(sprintf('iteration_%d', iteration)).test.negative_count = data_test_negative_count;
    ensemble.(sprintf('iteration_%d', iteration)).test.top_predictions = top_test_predictions;
    % ensemble.(sprintf('iteration_%d', iteration)).test.all_predictions = all_test_predictions;
    ensemble.(sprintf('iteration_%d', iteration)).test.labels = labels_test;
    ensemble.(sprintf('iteration_%d', iteration)).test.accuracy = test_accuracy;
    ensemble.(sprintf('iteration_%d', iteration)).test.sensitivity = test_sensitivity;
    ensemble.(sprintf('iteration_%d', iteration)).test.specificity = test_specificity;
    ensemble.(sprintf('iteration_%d', iteration)).samples.weights.pre_update = sample_weights(iteration,:);
    ensemble.(sprintf('iteration_%d', iteration)).samples.weights.post_update = sample_weights(iteration + 1,:);
    save(opts.paths.ensemble_models_file_path, 'ensemble');
    fprintf('done!\n');
    plotIncrementalEnsemblePerformance(ensemble, opts.paths.experiment_dir);
    % Incrementing loop counter
    iteration = iteration + 1;
  end
  % -------------------------------------------------------------------------
  %         10. All iterations are complete; normalize and save model weights
  % -------------------------------------------------------------------------
  very_high_number = 10e1;
  model_weight_per_iteration(model_weight_per_iteration > very_high_number) = very_high_number; % to replace Inf weight (when model has no loss)
  model_weight_per_iteration = model_weight_per_iteration / sum(model_weight_per_iteration);
  model_weight_per_iteration(model_weight_per_iteration < 1 / very_high_number) = 0; % to replace really small weights with 0
  for iteration = 1:length(model_weight_per_iteration)
    ensemble.(sprintf('iteration_%d', iteration)).trained_model.weight_normalized = model_weight_per_iteration(iteration);
  end

  % -------------------------------------------------------------------------
  %                                               11. Get performance summary
  % -------------------------------------------------------------------------
  training_method = sprintf('ensemble-%s', opts.ensemble_options.training_method);
  if opts.general.return_performance_summary
    afprintf(sprintf('[INFO] Getting model performance on `train` set...\n'));
    [top_train_predictions, ~] = getPredictionsFromModelOnImdb(ensemble, training_method, initial_imdb_with_all_train_and_all_test, 1);
    afprintf(sprintf('[INFO] Model performance on `train` set\n'));
    labels_train = imdb.images.labels(imdb.images.set == 1);
    [ ...
      train_accuracy, ...
      train_sensitivity, ...
      train_specificity, ...
    ] = getAccSensSpecMultiClass(labels_train, top_train_predictions, true);
    afprintf(sprintf('[INFO] Getting model performance on `test` set...\n'));
    [top_test_predictions, ~] = getPredictionsFromModelOnImdb(ensemble, training_method, initial_imdb_with_all_train_and_all_test, 3);
    afprintf(sprintf('[INFO] Model performance on `test` set\n'));
    labels_test = imdb.images.labels(imdb.images.set == 3);
    [ ...
      test_accuracy, ...
      test_sensitivity, ...
      test_specificity, ...
    ] = getAccSensSpecMultiClass(labels_test, top_test_predictions, true);
  else
    train_accuracy = -1;
    train_sensitivity = -1;
    train_specificity = -1;
    test_accuracy = -1;
    test_sensitivity = -1;
    test_specificity = -1;
  end

  % -------------------------------------------------------------------------
  %                                                         12. Assign output
  % -------------------------------------------------------------------------
  trained_model = ensemble;
  performance_summary.train.accuracy = train_accuracy;
  performance_summary.train.sensitivity = train_sensitivity;
  performance_summary.train.specificity = train_specificity;
  performance_summary.test.accuracy = test_accuracy;
  performance_summary.test.sensitivity = test_sensitivity;
  performance_summary.test.specificity = test_specificity;

  summarized_ensemble_info = getSummarizedEnsembleInfo(ensemble);
  performance_summary = mergeStructs(summarized_ensemble_info, performance_summary);

  % -------------------------------------------------------------------------
  %                                                           13. Save output
  % -------------------------------------------------------------------------
  saveStruct2File(performance_summary, opts.paths.results_file_path, 0);

% -------------------------------------------------------------------------
function summarized_ensemble_info = getSummarizedEnsembleInfo(ensemble)
% -------------------------------------------------------------------------
  if ~length(fieldnames(ensemble))
    summarized_ensemble_info.model_weight_normalized = -1;
    summarized_ensemble_info.train_positive_count = -1;
    summarized_ensemble_info.train_negative_count = -1;
    summarized_ensemble_info.validation_accuracy = -1;
    summarized_ensemble_info.validation_sensitivity = -1;
    summarized_ensemble_info.validation_specificity = -1;
    summarized_ensemble_info.test_accuracy = -1;
    summarized_ensemble_info.test_sensitivity = -1;
    summarized_ensemble_info.test_specificity = -1;
    return;
  end

  number_of_models_in_ensemble = length(fieldnames(ensemble));
  for iteration = 1:number_of_models_in_ensemble
    summarized_ensemble_info.model_weight_normalized(iteration) = ensemble.(sprintf('iteration_%d', iteration)).trained_model.weight_normalized;
    % summarized_ensemble_info.train_positive_count(iteration) = ensemble.(sprintf('iteration_%d', iteration)).trained_model.positive_count;
    % summarized_ensemble_info.train_negative_count(iteration) = ensemble.(sprintf('iteration_%d', iteration)).trained_model.negative_count;
    summarized_ensemble_info.validation_accuracy(iteration) = ensemble.(sprintf('iteration_%d', iteration)).validation.accuracy;
    summarized_ensemble_info.validation_sensitivity(iteration) = ensemble.(sprintf('iteration_%d', iteration)).validation.sensitivity;
    summarized_ensemble_info.validation_specificity(iteration) = ensemble.(sprintf('iteration_%d', iteration)).validation.specificity;
    summarized_ensemble_info.test_accuracy(iteration) = ensemble.(sprintf('iteration_%d', iteration)).test.accuracy;
    summarized_ensemble_info.test_sensitivity(iteration) = ensemble.(sprintf('iteration_%d', iteration)).test.sensitivity;
    summarized_ensemble_info.test_specificity(iteration) = ensemble.(sprintf('iteration_%d', iteration)).test.specificity;
  end

% -------------------------------------------------------------------------
function plotIncrementalEnsemblePerformance(ensemble, experiment_dir)
% -------------------------------------------------------------------------
  number_of_models_in_ensemble = length(fieldnames(ensemble));
  ensemble_models_validation_accuracy = zeros(1, number_of_models_in_ensemble);
  ensemble_models_validation_sensitivity = zeros(1, number_of_models_in_ensemble);
  ensemble_models_validation_specificity = zeros(1, number_of_models_in_ensemble);
  ensemble_models_test_accuracy = zeros(1, number_of_models_in_ensemble);
  ensemble_models_test_sensitivity = zeros(1, number_of_models_in_ensemble);
  ensemble_models_test_specificity = zeros(1, number_of_models_in_ensemble);
  for iteration = 1:number_of_models_in_ensemble
    ensemble_models_validation_accuracy(iteration) = ensemble.(sprintf('iteration_%d', iteration)).validation.accuracy;
    ensemble_models_validation_sensitivity(iteration) = ensemble.(sprintf('iteration_%d', iteration)).validation.sensitivity;
    ensemble_models_validation_specificity(iteration) = ensemble.(sprintf('iteration_%d', iteration)).validation.specificity;
    ensemble_models_test_accuracy(iteration) = ensemble.(sprintf('iteration_%d', iteration)).test.accuracy;
    ensemble_models_test_sensitivity(iteration) = ensemble.(sprintf('iteration_%d', iteration)).test.sensitivity;
    ensemble_models_test_specificity(iteration) = ensemble.(sprintf('iteration_%d', iteration)).test.specificity;
  end
  figure(2);
  clf;
  model_fig_path = fullfile(experiment_dir, 'incremental-performance.pdf');
  xlabel('iteration');
  title('individual model performance');
  grid on;
  hold on;
  plot(1:number_of_models_in_ensemble, ensemble_models_validation_accuracy, 'r.-', 'linewidth', 2);
  plot(1:number_of_models_in_ensemble, ensemble_models_validation_sensitivity, 'g.-', 'linewidth', 2);
  plot(1:number_of_models_in_ensemble, ensemble_models_validation_specificity, 'b.-', 'linewidth', 2);
  plot(1:number_of_models_in_ensemble, ensemble_models_test_accuracy, 'r.--', 'linewidth', 2);
  plot(1:number_of_models_in_ensemble, ensemble_models_test_sensitivity, 'g.--', 'linewidth', 2);
  plot(1:number_of_models_in_ensemble, ensemble_models_test_specificity, 'b.--', 'linewidth', 2);
  leg = { ...
    'val acc', ...
    'val sens', ...
    'val spec', ...
    'test acc', ...
    'test sens', ...
    'test spec', ...
  };
  set(legend(leg{:}),'color','none');
  drawnow;
  print(2, model_fig_path, '-dpdf');
