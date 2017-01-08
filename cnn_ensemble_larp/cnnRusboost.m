% -------------------------------------------------------------------------
function ensemble_performance_summary = cnnRusboost(single_ensemble_options)
% -------------------------------------------------------------------------
  % -------------------------------------------------------------------------
  %                                                              opts.general
  % -------------------------------------------------------------------------
  imdb = getValueFromFieldOrDefault(single_ensemble_options, 'imdb', struct());
  opts.general.dataset = getValueFromFieldOrDefault( ...
    single_ensemble_options, ...
    'dataset', ...
    'mnist-two-class-9-4');
  opts.general.network_arch = getValueFromFieldOrDefault( ...
    single_ensemble_options, ...
    'network_arch', ...
    'lenet');
  opts.general.iteration_count = getValueFromFieldOrDefault( ...
    single_ensemble_options, ...
    'iteration_count', ...
    5);

  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
    single_ensemble_options, ...
    'experiment_parent_dir', ...
    fullfile(vl_rootnn, 'experiment_results'));
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'rusboost-%s-%s-%s-max-iteration-count-%d', ...
    opts.general.dataset, ...
    opts.general.network_arch, ...
    opts.paths.time_string, ...
    opts.general.iteration_count));
  if ~exist(opts.paths.experiment_dir)
    mkdir(opts.paths.experiment_dir);
  end
  opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, 'options.txt');
  opts.paths.results_file_path = fullfile(opts.paths.experiment_dir, 'results.txt');
  opts.paths.ensemble_models_file_path = fullfile(opts.paths.experiment_dir, 'ensemble_models.mat');

  % -------------------------------------------------------------------------
  %                                                     opts.ensemble_options
  % -------------------------------------------------------------------------
  opts.ensemble_options.symmetric_weight_updates = getValueFromFieldOrDefault( ...
    single_ensemble_options, ...
    'symmetric_weight_updates', ...
    false);
  opts.ensemble_options.symmetric_loss_updates = getValueFromFieldOrDefault( ...
    single_ensemble_options, ...
    'symmetric_loss_updates', ...
    false);
  opts.ensemble_options.random_undersampling_ratio = (50/50);

  % -------------------------------------------------------------------------
  %                                                   opts.single_cnn_options
  % -------------------------------------------------------------------------
  opts.single_cnn_options.dataset = opts.general.dataset;
  opts.single_cnn_options.network_arch = opts.general.network_arch;
  opts.single_cnn_options.experiment_parent_dir = opts.paths.experiment_dir;
  opts.single_cnn_options.weight_init_source = 'gen';
  opts.single_cnn_options.weight_init_sequence = {'compRand', 'compRand', 'compRand'};
  opts.single_cnn_options.gpus = ifNotMacSetGpu(getValueFromFieldOrDefault(single_ensemble_options, 'gpus', 1));
  opts.single_cnn_options.backprop_depth = getValueFromFieldOrDefault(single_ensemble_options, 'backprop_depth', 4);
  opts.single_cnn_options.debug_flag = false;

  % -------------------------------------------------------------------------
  %                                                    save experiment setup!
  % -------------------------------------------------------------------------
  saveStruct2File(opts, opts.paths.options_file_path, 0);

  % -------------------------------------------------------------------------
  %                     2. process the imdb to separate positive and negative
  %                               samples (to be randomly-undersampled later)
  % -------------------------------------------------------------------------
  fh_imdb_utils = imdbTwoClassUtils;
  [ ...
    data, ...
    data_train, ...
    data_train_positive, ...
    data_train_negative, ...
    data_train_indices, ...
    data_train_positive_indices, ...
    data_train_negative_indices, ...
    data_train_count, ...
    data_train_positive_count, ...
    data_train_negative_count, ...
    labels_train, ...
    data_test, ...
    data_test_positive, ...
    data_test_negative, ...
    data_test_indices, ...
    data_test_positive_indices, ...
    data_test_negative_indices, ...
    data_test_count, ...
    data_test_positive_count, ...
    data_test_negative_count, ...
    labels_test, ...
  ] = fh_imdb_utils.getImdbInfo(imdb, 1);

  % -------------------------------------------------------------------------
  %                                     3. initialize training sample weights
  % -------------------------------------------------------------------------
  % W stores the weights of the instances in each row for every iteration of
  % boosting. Weights for all the instances are initialized by 1/m for the
  % first iteration.
  W = 1 / data_train_count * ones(1, data_train_count);

  % L stores pseudo loss values, H stores hypothesis, B stores (1/beta)
  % values that is used as the weight of the % hypothesis while forming the
  % final hypothesis. % All of the following are of length <=T and stores
  % values for every iteration of the boosting process.
  L = [];
  H = {};
  B = [];

  iteration = 1; % loop counter
  count = 1; % number of times the same boosting iteration have been repeated

  % -------------------------------------------------------------------------
  %                       4. create training (barebones) and validation imdbs
  % -------------------------------------------------------------------------
  training_resampled_imdb = fh_imdb_utils.constructPartialImdb([], [], 3); % barebones; filled in below
  validation_imdb = fh_imdb_utils.constructPartialImdb(data_train, labels_train, 3);
  test_imdb = fh_imdb_utils.constructPartialImdb(data_test, labels_test, 3);

  % -------------------------------------------------------------------------
  %                              5. go through T iterations of RUSBoost, each
  %                                              training a CNN over E epochs
  % -------------------------------------------------------------------------
  printConsoleOutputSeparator();
  ensemble_models = {};
  while iteration <= opts.general.iteration_count
    afprintf(sprintf('\n'));
    afprintf(sprintf('[INFO] Boosting iteration #%d (attempt %d)...\n', iteration, count));

    % Resampling NEG_DATA with weights of positive example
    afprintf(sprintf(...
      '[INFO] Resampling positive and negative data (ratio = %3.6f)... ', ...
      opts.ensemble_options.random_undersampling_ratio));

    try
      [resampled_data, resampled_labels] = fh_imdb_utils.resampleData( ...
        data_train, ...
        labels_train, ...
        W(iteration, :), ...
        opts.ensemble_options.random_undersampling_ratio);
      flag = true;
    catch
      afprintf(sprintf('[INFO] Weights no longer large enough to take sample from; terminating!\n'));
      break;
    end
    fprintf('done!\n');

    training_resampled_imdb.images.data = single(resampled_data);
    training_resampled_imdb.images.labels = single(resampled_labels);
    training_resampled_imdb.images.set = 1 * ones(length(resampled_labels), 1);

    afprintf(sprintf('[INFO] Training model (positive: %d, negative: %d)...\n', ...
      numel(find(resampled_labels == 2)), ...
      numel(find(resampled_labels == 1))));
    opts.single_cnn_options.imdb = training_resampled_imdb;
    [net, ~] = cnnAmir(opts.single_cnn_options);

    % IMPORTANT NOTE: we randomly undersample when training a model, but then,
    % we use all of the training samples (in their order) to update weights.
    afprintf(sprintf( ...
      '[INFO] Computing validation set predictions (positive: %d, negative: %d)...\n', ...
      data_train_positive_count, ...
      data_train_negative_count));
    validation_predictions = getPredictionsFromNetOnImdb(net, validation_imdb, 3);
    [ ...
      validation_accuracy, ...
      validation_sensitivity, ...
      validation_specificity, ...
    ] = getAccSensSpec(labels_train, validation_predictions, true);

    % -------------------------------------------------------------------------
    %                        6. Computing the pseudo loss of hypothesis 'model'
    % -------------------------------------------------------------------------
    afprintf(sprintf('[INFO] Computing pseudo loss... '));
    negative_to_positive_ratio = data_train_negative_count / data_train_positive_count;
    loss = 0;
    for i = 1:data_train_count
      if labels_train(i) == validation_predictions(i)
        continue;
      else
        if labels_train(i) == 2
          if opts.ensemble_options.symmetric_loss_updates
            loss = loss + W(iteration, i);
          else
            loss = loss + W(iteration, i) * min(negative_to_positive_ratio, 2);
          end
        else
          loss = loss + W(iteration, i);
        end
      end
    end
    fprintf('Loss: %6.5f\n', loss);

    % If count exceeds a pre-defined threshold (5 in the current implementation)
    % the loop is broken and rolled back to the state where loss > 0.5 was not
    % encountered.
    if count > 5
      L = L(1:iteration-1);
      H = H(1:iteration-1);
      B = B(1:iteration-1);
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

    H{iteration} = net; % Hypothesis function / Trained CNN Network
    L(iteration) = loss; % Pseudo-loss at each iteration
    beta = loss / (1 - loss); % Setting weight update parameter 'beta'.
    B(iteration) = log(1 / beta); % Weight of the hypothesis

    % % At the final iteration there is no need to update the weights any
    % % further
    % if iteration == opts.general.iteration_count
    %     break;
    % end

    % -------------------------------------------------------------------------
    %                                                        7. Updating weight
    % -------------------------------------------------------------------------
    afprintf(sprintf('[INFO] Updating weights... '));
    for i = 1:data_train_count
      if labels_train(i) == validation_predictions(i)
        W(iteration + 1, i) = W(iteration, i) * beta;
      else
        if labels_train(i) == 2
          if opts.ensemble_options.symmetric_weight_updates
            W(iteration + 1, i) = W(iteration, i);
          else
            W(iteration + 1, i) = W(iteration, i) * min(negative_to_positive_ratio, 2);
          end
        else
          W(iteration + 1, i) = W(iteration, i);
        end
      end
    end
    fprintf('done!\n');

    % Normalizing the weight for the next iteration
    sum_W = sum(W(iteration + 1, :));
    for i = 1:data_train_count
      W(iteration + 1, i) = W(iteration + 1, i) / sum_W;
    end

    % -------------------------------------------------------------------------
    %                                       8. test on single model of ensemble
    % -------------------------------------------------------------------------
    afprintf(sprintf('[INFO] Computing test set predictions (positive: %d, negative: %d)...\n', ...
      data_test_positive_count, ...
      data_test_negative_count));
    test_predictions = getPredictionsFromNetOnImdb(net, test_imdb, 3);
    [ ...
      test_accuracy, ...
      test_sensitivity, ...
      test_specificity, ...
    ] = getAccSensSpec(labels_test, test_predictions, true);

    % -------------------------------------------------------------------------
    %                                          9. save single model of ensemble
    % -------------------------------------------------------------------------
    afprintf(sprintf('[INFO] Saving model and info... '));
    ensemble_models{iteration}.model_net = H{iteration};
    ensemble_models{iteration}.model_loss = L(iteration);
    ensemble_models{iteration}.model_weight_normalized = 0;
    ensemble_models{iteration}.model_weight_not_normalized = B(iteration);
    ensemble_models{iteration}.train_positive_count = numel(find(resampled_labels == 2));
    ensemble_models{iteration}.train_negative_count = numel(find(resampled_labels == 1));
    ensemble_models{iteration}.validation_positive_count = data_train_positive_count;
    ensemble_models{iteration}.validation_negative_count = data_train_negative_count;
    ensemble_models{iteration}.validation_predictions = validation_predictions;
    ensemble_models{iteration}.validation_labels = labels_train;
    ensemble_models{iteration}.validation_accuracy = validation_accuracy;
    ensemble_models{iteration}.validation_sensitivity = validation_sensitivity;
    ensemble_models{iteration}.validation_specificity = validation_specificity;
    ensemble_models{iteration}.validation_weights_pre_update = W(iteration,:);
    ensemble_models{iteration}.validation_weights_post_update = W(iteration + 1,:);
    ensemble_models{iteration}.test_positive_count = data_test_positive_count;
    ensemble_models{iteration}.test_negative_count = data_test_negative_count;
    ensemble_models{iteration}.test_predictions = test_predictions;
    ensemble_models{iteration}.test_labels = labels_test;
    ensemble_models{iteration}.test_accuracy = test_accuracy;
    ensemble_models{iteration}.test_sensitivity = test_sensitivity;
    ensemble_models{iteration}.test_specificity = test_specificity;
    save(opts.paths.ensemble_models_file_path, 'ensemble_models');
    fprintf('done!\n');
    plotIncrementalEnsemblePerformance(ensemble_models, opts.paths.experiment_dir);
    % Incrementing loop counter
    iteration = iteration + 1;
  end
  % -------------------------------------------------------------------------
  % 10. now that all iterations are complete normalize and save model weights
  % -------------------------------------------------------------------------
  B = B / sum(B);
  for iteration = 1:length(B)
    ensemble_models{iteration}.model_weight_normalized = B(iteration);
  end

  % folds.(sprintf('fold_%d', i))

  % -------------------------------------------------------------------------
  %            11. test on test set, keeping in mind beta's between each mode
  % -------------------------------------------------------------------------
  % The final hypothesis is calculated and tested on the test set simulteneously
  printConsoleOutputSeparator();
  ensemble_performance_summary = getEnsemblePerformanceSummary(ensemble_models, test_imdb);
  saveStruct2File(ensemble_performance_summary, opts.paths.results_file_path, 0);
  printConsoleOutputSeparator();

% -------------------------------------------------------------------------
function plotIncrementalEnsemblePerformance(ensemble_models, experiment_dir)
% -------------------------------------------------------------------------
  num_models_in_ensemble = numel(ensemble_models);
  ensemble_models_validation_accuracy = zeros(1, num_models_in_ensemble);
  ensemble_models_validation_sensitivity = zeros(1, num_models_in_ensemble);
  ensemble_models_validation_specificity = zeros(1, num_models_in_ensemble);
  ensemble_models_test_accuracy = zeros(1, num_models_in_ensemble);
  ensemble_models_test_sensitivity = zeros(1, num_models_in_ensemble);
  ensemble_models_test_specificity = zeros(1, num_models_in_ensemble);
  for i = 1:num_models_in_ensemble
    ensemble_models_validation_accuracy(i) = ensemble_models{i}.validation_accuracy;
    ensemble_models_validation_sensitivity(i) = ensemble_models{i}.validation_sensitivity;
    ensemble_models_validation_specificity(i) = ensemble_models{i}.validation_specificity;
    ensemble_models_test_accuracy(i) = ensemble_models{i}.test_accuracy;
    ensemble_models_test_sensitivity(i) = ensemble_models{i}.test_sensitivity;
    ensemble_models_test_specificity(i) = ensemble_models{i}.test_specificity;
  end
  figure(2);
  clf;
  model_fig_path = fullfile(experiment_dir, 'incremental-performance.pdf');
  xlabel('training epoch'); ylabel('error');
  title('performance');
  grid on;
  hold on;
  plot(1:num_models_in_ensemble, ensemble_models_validation_accuracy, 'r.-', 'linewidth', 2);
  plot(1:num_models_in_ensemble, ensemble_models_validation_sensitivity, 'g.-', 'linewidth', 2);
  plot(1:num_models_in_ensemble, ensemble_models_validation_specificity, 'b.-', 'linewidth', 2);
  plot(1:num_models_in_ensemble, ensemble_models_test_accuracy, 'r.--', 'linewidth', 2);
  plot(1:num_models_in_ensemble, ensemble_models_test_sensitivity, 'g.--', 'linewidth', 2);
  plot(1:num_models_in_ensemble, ensemble_models_test_specificity, 'b.--', 'linewidth', 2);
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

% -------------------------------------------------------------------------
function ensemble_performance_summary = getEnsemblePerformanceSummary(ensemble_models, test_imdb)
% -------------------------------------------------------------------------
  ensemble_performance_summary.model_weight_normalized = 0;
  ensemble_performance_summary.train_positive_count = 0;
  ensemble_performance_summary.train_negative_count = 0;
  ensemble_performance_summary.validation_accuracy = 0;
  ensemble_performance_summary.validation_sensitivity = 0;
  ensemble_performance_summary.validation_specificity = 0;
  ensemble_performance_summary.test_accuracy = 0;
  ensemble_performance_summary.test_sensitivity = 0;
  ensemble_performance_summary.test_specificity = 0;
  ensemble_performance_summary.weighted_test_accuracy = 0;
  ensemble_performance_summary.weighted_test_sensitivity = 0;
  ensemble_performance_summary.weighted_test_specificity= 0;


  number_of_models_in_ensemble = numel(ensemble_models);
  for iteration = 1:number_of_models_in_ensemble
    ensemble_performance_summary.model_weight_normalized(iteration) = ensemble_models{iteration}.model_weight_normalized;
    ensemble_performance_summary.train_positive_count(iteration) = ensemble_models{iteration}.train_positive_count;
    ensemble_performance_summary.train_negative_count(iteration) = ensemble_models{iteration}.train_negative_count;
    ensemble_performance_summary.validation_accuracy(iteration) = ensemble_models{iteration}.validation_accuracy;
    ensemble_performance_summary.validation_sensitivity(iteration) = ensemble_models{iteration}.validation_sensitivity;
    ensemble_performance_summary.validation_specificity(iteration) = ensemble_models{iteration}.validation_specificity;
    ensemble_performance_summary.test_accuracy(iteration) = ensemble_models{iteration}.test_accuracy;
    ensemble_performance_summary.test_sensitivity(iteration) = ensemble_models{iteration}.test_sensitivity;
    ensemble_performance_summary.test_specificity(iteration) = ensemble_models{iteration}.test_specificity;
  end

  weighted_results = getWeightedEnsembleResultsOnTestSet(ensemble_models, test_imdb);
  ensemble_performance_summary.weighted_test_accuracy = weighted_results.test_accuracy;
  ensemble_performance_summary.weighted_test_sensitivity = weighted_results.test_sensitivity;
  ensemble_performance_summary.weighted_test_specificity = weighted_results.test_specificity;

% -------------------------------------------------------------------------
function weighted_results = getWeightedEnsembleResultsOnTestSet(ensemble_models, test_imdb)
% -------------------------------------------------------------------------
  fprintf('\n');
  afprintf(sprintf('[INFO] ENSEMBLE RESULTS ON TEST SET: \n'));
  printConsoleOutputSeparator();

  % -------------------------------------------------------------------------
  % Initial stuff
  % -------------------------------------------------------------------------
  fh_imdb_utils = imdbTwoClassUtils;
  [ ...
    data, ...
    data_train, ...
    data_train_positive, ...
    data_train_negative, ...
    data_train_indices, ...
    data_train_positive_indices, ...
    data_train_negative_indices, ...
    data_train_count, ...
    data_train_positive_count, ...
    data_train_negative_count, ...
    labels_train, ...
    data_test, ...
    data_test_positive, ...
    data_test_negative, ...
    data_test_indices, ...
    data_test_positive_indices, ...
    data_test_negative_indices, ...
    data_test_count, ...
    data_test_positive_count, ...
    data_test_negative_count, ...
    labels_test, ...
  ] = fh_imdb_utils.getImdbInfo(test_imdb, 1);

  number_of_models_in_ensemble = numel(ensemble_models);
  B = zeros(1, number_of_models_in_ensemble);
  for iteration = 1:numel(B)
    B(iteration) = ensemble_models{iteration}.model_weight_normalized;
  end

  weighted_ensemble_prediction = zeros(1, data_test_count);
  test_set_predictions_per_model = {};
  for iteration = 1:number_of_models_in_ensemble % looping through all trained models
    afprintf(sprintf('\n'));
    afprintf(sprintf('[INFO] Getting test set predictions for model #%d (positive: %d, negative: %d)...', ...
      iteration, ...
      data_test_positive_count, ...
      data_test_negative_count));
    test_set_predictions_per_model{iteration} = ensemble_models{iteration}.test_predictions;
    fprintf('done.\n');
  end

  for i = 1:data_test_count
    % Calculating the total weight of the class labels from all the models
    % produced during boosting
    wt_positive = 0; % class 2
    wt_negative = 0; % class 1
    for iteration = 1:number_of_models_in_ensemble % looping through all trained models
       p = test_set_predictions_per_model{iteration}(i);
       if p == 2 % if is positive
           wt_positive = wt_positive + B(iteration);
       else
           wt_negative = wt_negative + B(iteration);
       end
    end

    if (wt_positive > wt_negative)
        weighted_ensemble_prediction(i) = 2;
    else
        weighted_ensemble_prediction(i) = 1;
    end
  end

  % -------------------------------------------------------------------------
  % 7. done, go treat yourself to something sugary!
  % -------------------------------------------------------------------------
  printConsoleOutputSeparator();
  predictions_test = weighted_ensemble_prediction;
  afprintf(sprintf('Weighted results:\n'));
  afprintf(sprintf('Model Weights: '), 1);
  for i = 1:length(B)
    fprintf('%.2f,\t', B(i));
  end
  [weighted_accuracy, weighted_sensitivity, weighted_specificity] = getAccSensSpec(labels_test, predictions_test, true);
  weighted_results.test_accuracy = weighted_accuracy;
  weighted_results.test_sensitivity = weighted_sensitivity;
  weighted_results.test_specificity = weighted_specificity;
