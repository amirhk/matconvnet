function fh = cnnRusboost()
  % assign function handles so we can call these local functions from elsewhere
  fh.getInitialImdb = @getInitialImdb;
  fh.mainCNNRusboost = @mainCNNRusboost;
  fh.kFoldCNNRusboost = @kFoldCNNRusboost;
  fh.testAllEnsembleModelsOnTestImdb = @testAllEnsembleModelsOnTestImdb;
  fh.saveKFoldResults = @saveKFoldResults;
  fh.printKFoldResults = @printKFoldResults;

% -------------------------------------------------------------------------
function folds = kFoldCNNRusboost()
% -------------------------------------------------------------------------
  % -------------------------------------------------------------------------
  %                                                                   general
  % -------------------------------------------------------------------------
  opts.number_of_folds = 5;
  opts.max_number_of_models_in_each_ensemble = 10;
  opts.dataset = 'mnist-two-class-unbalanced';

  % -------------------------------------------------------------------------
  %                                                                      imdb
  % -------------------------------------------------------------------------
  opts.num_patients = 104;
  opts.leave_out_type = 'special';
  opts.contrast_normalization = true;
  % opts.whitenData = true;
  opts.train_balance = false;
  opts.train_augment_healthy = 'none';
  opts.train_augment_cancer = 'none';
  opts.test_balance = false;
  opts.test_augment_healthy = 'none';
  opts.test_augment_cancer = 'none';

  % -------------------------------------------------------------------------
  %                                                                     paths
  % -------------------------------------------------------------------------
  opts.data_dir = fullfile(getDevPath(), 'data', 'source', sprintf('%s', opts.dataset));
  opts.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.experiment_dir_parent_path = fullfile('experiment_results', sprintf('k-fold-rusboost-%s', opts.time_string));
  if ~exist(opts.experiment_dir_parent_path)
    mkdir(opts.experiment_dir_parent_path);
  end
  opts.folds_file_path = fullfile(opts.experiment_dir_parent_path, 'folds.mat');
  opts.options_file_path = fullfile(opts.experiment_dir_parent_path, 'options.txt');
  opts.results_file_path = fullfile(opts.experiment_dir_parent_path, 'results.txt');

  % -------------------------------------------------------------------------
  %                                                    save experiment setup!
  % -------------------------------------------------------------------------
  saveStruct2File(opts, opts.options_file_path, 0);
  afprintf(sprintf('[INFO] Running K-fold CNN Rusboost (K = %d)...\n', opts.number_of_folds), 1);

  % -------------------------------------------------------------------------
  %                                 randomly divide off patients into K folds
  % -------------------------------------------------------------------------
  folds = {};
  patients_per_fold = ceil(opts.num_patients / opts.number_of_folds);
  random_patient_indices = randperm(104);
  afprintf(sprintf('\n'));
  afprintf(sprintf('[INFO] Randomly dividing patients into K = %d folds...\n', opts.number_of_folds));
  for i = 1:opts.number_of_folds
    start_index = 1 + (i - 1) * patients_per_fold;
    end_index = min(104, i * patients_per_fold);
    folds.(sprintf('fold_%d', i)).patient_indices = random_patient_indices(start_index : end_index);
  end

  % -------------------------------------------------------------------------
  %                   create a non-balanced, non-augmented imdb for each fold
  % -------------------------------------------------------------------------
  imdbs = {}; % separate so don't have to save ~1.5 GB of imdbs!!!

  switch opts.dataset
    case 'mnist-two-class-unbalanced'
      for i = 1:opts.number_of_folds
        afprintf(sprintf('\n'));
        afprintf(sprintf('[INFO] Constructing imdb for fold #%d...\n', i));
        opts.network_arch = 'lenet';
        % imdb = constructMnistUnbalancedTwoClassImdb(opts.data_dir, opts.network_arch);
        tmp = load(fullfile(getDevPath(), 'data', 'saved-two-class-mnist.mat'));
        imdb = tmp.imdb;
        imdbs{i} = imdb;
        afprintf(sprintf('[INFO] done!\n'));
      end
      single_ensemble_options.dataset = 'mnist-two-class-unbalanced';
      single_ensemble_options.network_arch = 'lenet';
    case 'prostate'
      for i = 1:opts.number_of_folds
        afprintf(sprintf('\n'));
        afprintf(sprintf('[INFO] Constructing imdb for fold #%d...\n', i));
        opts.leave_out_indices = folds.(sprintf('fold_%d', i)).patient_indices;
        imdb = constructProstateImdb(opts);
        imdbs{i} = imdb;
        afprintf(sprintf('[INFO] done!\n'));
      end
      single_ensemble_options.dataset = 'prostate';
      single_ensemble_options.network_arch = 'prostatenet';
  end

  % -------------------------------------------------------------------------
  %                                        train ensemble larp for each fold!
  % -------------------------------------------------------------------------
  for i = 1:opts.number_of_folds
    afprintf(sprintf('[INFO] Running cnn_rusboost on fold #%d...\n', i));
    single_ensemble_options.imdb = imdbs{i};
    single_ensemble_options.experiment_dir_parent_path = opts.experiment_dir_parent_path;
    single_ensemble_options.iteration_count = opts.max_number_of_models_in_each_ensemble;
    [ ...
      folds.(sprintf('fold_%d', i)).ensemble_models_info, ...
      folds.(sprintf('fold_%d', i)).weighted_results, ...
    ] = mainCNNRusboost(single_ensemble_options);
    % overwrite and save results so far
    save(opts.folds_file_path, 'folds');
  end

  % -------------------------------------------------------------------------
  %                                                    save and print results
  % -------------------------------------------------------------------------
  saveKFoldResults(folds, opts.results_file_path);
  printKFoldResults(folds);

% -------------------------------------------------------------------------
function [ensemble_models_info, weighted_results] = mainCNNRusboost(single_ensemble_options)
% -------------------------------------------------------------------------
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 0. take as input a pre-processed IMDB (augment cancer in training set, that's it!), say
  %   train: 94 patients
  %   test: 10 patients, ~1000 health, ~20 cancer
  % TODO: this can be extended to be say 10-fold ensemble larp, then average the folds
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  imdb = getValueFromFieldOrDefault(single_ensemble_options, 'imdb', getInitialImdb());
  experiment_dir_parent_path = getValueFromFieldOrDefault(single_ensemble_options, 'experiment_dir_parent_path', 'data_rusboost');
  iteration_count = getValueFromFieldOrDefault(single_ensemble_options, 'iteration_count', 5);
  dataset = getValueFromFieldOrDefault(single_ensemble_options, 'dataset', 'prostate');
  network_arch = getValueFromFieldOrDefault(single_ensemble_options, 'network_arch', 'prostatenet');

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 1. some important parameter definition
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  opts.iteration_count = iteration_count; % number of boosting iterations
  opts.dataset = dataset;
  opts.network_arch = network_arch;
  opts.backprop_depth = 4;
  opts.weight_init_source = 'gen';
  opts.weight_init_sequence = {'compRand', 'compRand', 'compRand'};
  opts.random_undersampling_ratio = (50/50);

  opts.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.experiment_dir_path = fullfile( ...
    experiment_dir_parent_path, ...
    sprintf('rusboost-%s-%s-%s', opts.dataset, opts.network_arch, opts.time_string));
  opts.all_model_infos_path = fullfile(opts.experiment_dir_path, 'ensemble_models_info.mat');
  if ~exist(opts.experiment_dir_path)
    mkdir(opts.experiment_dir_path);
  end

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 2. process the imdb to separate positive and negative samples (to be
  % randomly-undersampled later)
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  data_train = imdb.images.data(:,:,:,imdb.images.set == 1);
  labels_train = imdb.images.labels(imdb.images.set == 1);
  data_train_healthy = data_train(:,:,:,labels_train == 1);
  data_train_cancer = data_train(:,:,:,labels_train == 2);
  data_train_count = size(data_train, 4);
  data_train_healthy_count = size(data_train_healthy, 4);
  data_train_cancer_count = size(data_train_cancer, 4);

  data_test = imdb.images.data(:,:,:,imdb.images.set == 3);
  labels_test = imdb.images.labels(imdb.images.set == 3);
  data_test_healthy = data_test(:,:,:,labels_test == 1);
  data_test_cancer = data_test(:,:,:,labels_test == 2);
  data_test_count = size(data_test, 4);
  data_test_healthy_count = size(data_test_healthy, 4);
  data_test_cancer_count = size(data_test_cancer, 4);

  afprintf(sprintf('[INFO] TRAINING SET: total: %d, healthy: %d, cancer: %d\n', ...
    data_train_count, ...
    data_train_healthy_count, ...
    data_train_cancer_count));
  afprintf(sprintf('[INFO] TESTING SET: total: %d, healthy: %d, cancer: %d\n', ...
    data_test_count, ...
    data_test_healthy_count, ...
    data_test_cancer_count));

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 3. initialize training sample weights
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % W stores the weights of the instances in each row for every iteration of
  % boosting. Weights for all the instances are initialized by 1/m for the
  % first iteration.
  W = zeros(1, data_train_count);
  for i = 1 : data_train_count
    W(1, i) = 1 / data_train_count;
  end

  % L stores pseudo loss values, H stores hypothesis, B stores (1/beta)
  % values that is used as the weight of the % hypothesis while forming the
  % final hypothesis. % All of the following are of length <=T and stores
  % values for every iteration of the boosting process.
  L = [];
  H = {};
  B = [];

  t = 1; % loop counter
  count = 1; % number of times the same boosting iteration have been repeated

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 4. create training (barebones) and validation imdbs
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  training_resampled_imdb = constructPartialImdb([], [], 3); % barebones; filled in below
  validation_imdb = constructPartialImdb(data_train, labels_train, 3);
  test_imdb = constructPartialImdb(data_test, labels_test, 3);

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 5. go through T iterations of RUSBoost, each of which trains a CNN over E epochs
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  printConsoleOutputSeparator();
  ensemble_models_info = {};
  while t <= opts.iteration_count
    afprintf(sprintf('\n'));
    afprintf(sprintf('[INFO] Boosting iteration #%d (attempt %d)...\n', t, count));

    % Resampling NEG_DATA with weights of positive example
    afprintf(sprintf('[INFO] Resampling healthy and cancer data (ratio = %3.6f)... ', opts.random_undersampling_ratio));
    [resampled_data, resampled_labels] = ...
      resampleData(data_train, labels_train, W(t, :), opts.random_undersampling_ratio);
    afprintf(sprintf('done!\n'));

    training_resampled_imdb.images.data = single(resampled_data);
    training_resampled_imdb.images.labels = single(resampled_labels);
    training_resampled_imdb.images.set = 1 * ones(length(resampled_labels), 1);

    % Weird. Need at least 1 test sample for cnn_train to work. TODO: this is because of TP stuff in cnn_train
    training_resampled_imdb.images.data = cat(4, training_resampled_imdb.images.data, resampled_data(:,:,:, end));
    training_resampled_imdb.images.labels = cat(2,training_resampled_imdb.images.labels, resampled_labels(end));
    training_resampled_imdb.images.set = cat(1, training_resampled_imdb.images.set, 3);

    afprintf(sprintf('[INFO] Training model (healthy: %d, cancer: %d)...\n', ...
      numel(find(resampled_labels == 1)), ...
      numel(find(resampled_labels == 2))));
    train_opts.imdb = training_resampled_imdb;
    train_opts.dataset = opts.dataset;
    train_opts.network_arch = opts.network_arch;
    train_opts.backprop_depth = opts.backprop_depth;
    train_opts.weight_init_source = opts.weight_init_source;
    train_opts.weight_init_sequence = opts.weight_init_sequence;
    train_opts.debug_flag = false;
    train_opts.experiment_parent_dir = opts.experiment_dir_path;
    [net, info] = cnnAmir(train_opts);

    % IMPORTANT NOTE: we randomly undersample when training a model, but then,
    % we use all of the training samples (in their order) to update weights.
    afprintf(sprintf('[INFO] Computing validation set predictions (healthy: %d, cancer: %d)...\n', ...
      data_train_healthy_count, ...
      data_train_cancer_count));
    validation_predictions = getPredictionsFromNetOnImdb(net, validation_imdb, 3);
    [ ...
      validation_acc, ...
      validation_sens, ...
      validation_spec, ...
    ] = getAccSensSpec(labels_train, validation_predictions, true);

    % Computing the pseudo loss of hypothesis 'model'
    afprintf(sprintf('[INFO] Computing pseudo loss... '));
    healthy_to_cancer_ratio = data_train_healthy_count / data_train_cancer_count;
    loss = 0;
    for i = 1:data_train_count
      if labels_train(i) == validation_predictions(i)
        continue;
      else
        loss = loss + W(t, i);
      end
    end
    fprintf('Loss: %6.5f\n', loss);

    % If count exceeds a pre-defined threshold (5 in the current
    % implementation), the loop is broken and rolled back to the state
    % where loss > 0.5 was not encountered.
    if count > 5
      L = L(1:t-1);
      H = H(1:t-1);
      B = B(1:t-1);
      afprintf(sprintf('Too many iterations have loss > 0.5\n'));
      afprintf(sprintf('Aborting boosting...\n'));
      break;
    end

    % If the loss is greater than 1/2, it means that an inverted
    % hypothesis would perform better. In such cases, do not take that
    % hypothesis into consideration and repeat the same iteration. 'count'
    % keeps counts of the number of times the same boosting iteration have
    % been repeated
    if loss > 0.5
      count = count + 1;
      continue;
    else
      count = 1;
    end

    H{t} = net; % Hypothesis function / Trained CNN Network
    L(t) = loss; % Pseudo-loss at each iteration
    beta = loss / (1 - loss); % Setting weight update parameter 'beta'.
    B(t) = log(1 / beta); % Weight of the hypothesis

    % % At the final iteration there is no need to update the weights any
    % % further
    % if t == opts.iteration_count
    %     break;
    % end

    % Updating weight
    afprintf(sprintf('[INFO] Updating weights... '));
    for i = 1:data_train_count
      if labels_train(i) == validation_predictions(i)
        W(t + 1, i) = W(t, i) * beta;
      else
        if labels_train(i) == 2
          W(t + 1, i) = min(healthy_to_cancer_ratio, 5) * W(t, i);
          % W(t + 1, i) = W(t, i);
        else
          W(t + 1, i) = W(t, i);
        end
      end
    end
    fprintf('done!\n');

    % Normalizing the weight for the next iteration
    sum_W = sum(W(t + 1, :));
    for i = 1:data_train_count
      W(t + 1, i) = W(t + 1, i) / sum_W;
    end

    %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    % 6. test on single model of ensemble
    %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    afprintf(sprintf('[INFO] Computing test set predictions (healthy: %d, cancer: %d)...\n', ...
      data_test_healthy_count, ...
      data_test_cancer_count));
    test_predictions = getPredictionsFromNetOnImdb(net, test_imdb, 3);
    [ ...
      test_acc, ...
      test_sens, ...
      test_spec, ...
    ] = getAccSensSpec(labels_test, test_predictions, true);

    %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    % 7. save single model of ensemble
    %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    afprintf(sprintf('[INFO] Saving model and info... '));
    ensemble_models_info{t}.model_net = H{t};
    ensemble_models_info{t}.model_loss = L(t);
    ensemble_models_info{t}.model_weight = B(t);
    ensemble_models_info{t}.train_healthy_count = numel(find(resampled_labels == 1));
    ensemble_models_info{t}.train_cancer_count = numel(find(resampled_labels == 2));
    ensemble_models_info{t}.validation_healthy_count = data_train_healthy_count;
    ensemble_models_info{t}.validation_cancer_count = data_train_cancer_count;
    ensemble_models_info{t}.validation_predictions = validation_predictions;
    ensemble_models_info{t}.validation_labels = labels_train;
    ensemble_models_info{t}.validation_accuracy = validation_acc;
    ensemble_models_info{t}.validation_sensitivity = validation_sens;
    ensemble_models_info{t}.validation_specificity = validation_spec;
    ensemble_models_info{t}.validation_weights_pre_update = W(t,:);
    ensemble_models_info{t}.validation_weights_post_update = W(t + 1,:);
    ensemble_models_info{t}.test_healthy_count = data_test_healthy_count;
    ensemble_models_info{t}.test_cancer_count = data_test_cancer_count;
    ensemble_models_info{t}.test_predictions = test_predictions;
    ensemble_models_info{t}.test_labels = labels_test;
    ensemble_models_info{t}.test_accuracy = test_acc;
    ensemble_models_info{t}.test_sensitivity = test_sens;
    ensemble_models_info{t}.test_specificity = test_spec;
    save(opts.all_model_infos_path, 'ensemble_models_info');
    fprintf('done!\n');
    plotThisShit(ensemble_models_info, opts.experiment_dir_path);
    % Incrementing loop counter
    t = t + 1;
  end

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 8. test on test set, keeping in mind beta's between each mode
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % The final hypothesis is calculated and tested on the test set simulteneously
  printConsoleOutputSeparator();
  weighted_results = testAllEnsembleModelsOnTestImdb(ensemble_models_info, imdb);
  printConsoleOutputSeparator();

% -------------------------------------------------------------------------
function [resampled_data, resampled_labels] = resampleData(data, labels, weights, ratio)
% -------------------------------------------------------------------------
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % Initial stuff
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  data_healthy = data(:,:,:,labels == 1);
  data_cancer = data(:,:,:,labels == 2);
  data_count = size(data, 4);
  data_healthy_count = size(data_healthy, 4);
  data_cancer_count = size(data_cancer, 4);
  data_healthy_indices = find(labels == 1);
  data_cancer_indices = find(labels == 2);

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % Random Under-sampling (RUS): Healthy Data
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  downsampled_data_healthy_count = round(data_cancer_count * ratio);
  downsampled_data_healthy_indices = randsample(data_healthy_indices, downsampled_data_healthy_count, false);
  downsampled_data_healthy = data(:,:,:, downsampled_data_healthy_indices);

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % Weighted Upsampling (more weight -> more repeat): Healthy & Cancer Data
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  max_repeat_healthy = 25;
  max_repeat_cancer = 200;
  normalized_weights = weights / min(weights);
  repeat_counts = ceil(normalized_weights);
  for j = data_healthy_indices
    if repeat_counts(j) > max_repeat_healthy
      repeat_counts(j) = max_repeat_healthy;
    end
  end
  for j = data_cancer_indices
    if repeat_counts(j) > max_repeat_cancer
      repeat_counts(j) = max_repeat_cancer;
    end
  end

  healthy_repeat_counts = repeat_counts(downsampled_data_healthy_indices);
  cancer_repeat_counts = repeat_counts(data_cancer_indices);

  upsampled_data_healthy = upsample(downsampled_data_healthy, healthy_repeat_counts);
  upsampled_data_cancer = upsample(data_cancer, cancer_repeat_counts);

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % Putting it all together
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  resampled_data_healthy_count = size(upsampled_data_healthy, 4);
  resampled_data_cancer_count = size(upsampled_data_cancer, 4);
  resampled_data_all = cat(4, upsampled_data_healthy, upsampled_data_cancer);
  resampled_labels_all = cat( ...
    2, ...
    1 * ones(1, resampled_data_healthy_count), ...
    2 * ones(1, resampled_data_cancer_count));

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % Shuffle this to mixup order of healthy and cancer in imdb so we don't
  % have the CNN overtrain in 1 particular direction. Only shuffling for
  % training; later weights are calculated and updated for all training data.
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  ix = randperm(size(resampled_data_all, 4));
  resampled_data = resampled_data_all(:,:,:,ix);
  resampled_labels = resampled_labels_all(ix);

% -------------------------------------------------------------------------
function [upsampled_data] = upsample(data, repeat_counts)
  % remember, data is 4D, with N 3D samples
% -------------------------------------------------------------------------
  assert(size(data, 4) == length(repeat_counts));
  total_repeat_count = sum(repeat_counts);
  upsampled_data = zeros(size(data, 1), size(data, 2), size(data, 3), total_repeat_count);
  counter = 1;
  for i = 1:length(repeat_counts)
    sample_repeat_count = repeat_counts(i);
    % repeated_sample = repmat(data(:,:,:,i), [1,1,1,sample_repeat_count]);
    repeated_sample_4D_matrix = augmentSample(data(:,:,:,i), sample_repeat_count, 'rotate-flip');
    upsampled_data(:,:,:, counter : counter + sample_repeat_count - 1) = repeated_sample_4D_matrix;
    counter = counter + sample_repeat_count;
  end

% -------------------------------------------------------------------------
function [repeated_sample_4D_matrix] = augmentSample(sample, repeat_count, augment_type)
  % augment_type = {'repmat', 'rotate', 'flip', 'rotate-flip'}
% -------------------------------------------------------------------------
  repeated_sample_4D_matrix = zeros(size(sample, 1), size(sample, 2), size(sample, 3), repeat_count);
  switch augment_type
    case 'repmat'
      repeated_sample_4D_matrix = repmat(sample, [1,1,1,repeat_count]);
    case 'rotate'
      degrees = linspace(0, 360, repeat_count);
      index = 1;
      for degree = degrees
        rotated_3D_image = imrotate(sample, degree, 'crop');
        repeated_sample_4D_matrix(:,:,:,index) = rotated_3D_image;
        index = index + 1;
      end
    case 'rotate-flip'
      degrees = linspace(0, 360, floor(repeat_count / 2));
      index = 1;
      for degree = degrees
        rotated_3D_image = imrotate(sample, degree, 'crop');
        repeated_sample_4D_matrix(:,:,:,index) = rotated_3D_image;
        repeated_sample_4D_matrix(:,:,:,index + 1) = fliplr(rotated_3D_image);
        index = index + 2;
      end
      if mod(repeat_count, 2)
        % because of the `floor()` above, the last index of
        % repeated_sample_4D_matrixhas not been augmented yet...
        % just make it a simple copy of sample.
        repeated_sample_4D_matrix(:,:,:,end) = sample;
      end
  end

% -------------------------------------------------------------------------
function imdb = constructPartialImdb(data, labels, set_number)
% -------------------------------------------------------------------------
  imdb.images.data = data;
  imdb.images.labels = labels;
  imdb.images.set = set_number * ones(length(labels), 1);
  imdb.meta.sets = {'train', 'val', 'test'};

% -------------------------------------------------------------------------
function imdb = getInitialImdb()
% -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Loading saved imdb... '));
  imdbPath = fullfile(getDevPath(), '/matconvnet/data_1/_prostate/_saved_prostate_imdb.mat');
  imdb = load(imdbPath);
  imdb = imdb.imdb;
  afprintf(sprintf('done!\n'));

% -------------------------------------------------------------------------
function plotThisShit(ensemble_models_info, experiment_dir_path)
% -------------------------------------------------------------------------
  num_models_in_ensemble = numel(ensemble_models_info);
  ensemble_models_validation_accuracy = zeros(1, num_models_in_ensemble);
  ensemble_models_validation_sensitivity = zeros(1, num_models_in_ensemble);
  ensemble_models_validation_specificity = zeros(1, num_models_in_ensemble);
  ensemble_models_test_accuracy = zeros(1, num_models_in_ensemble);
  ensemble_models_test_sensitivity = zeros(1, num_models_in_ensemble);
  ensemble_models_test_specificity = zeros(1, num_models_in_ensemble);
  for i = num_models_in_ensemble
    ensemble_models_validation_accuracy(i) = ensemble_models_info{i}.validation_accuracy;
    ensemble_models_validation_sensitivity(i) = ensemble_models_info{i}.validation_sensitivity;
    ensemble_models_validation_specificity(i) = ensemble_models_info{i}.validation_specificity;
    ensemble_models_test_accuracy(i) = ensemble_models_info{i}.test_accuracy;
    ensemble_models_test_sensitivity(i) = ensemble_models_info{i}.test_sensitivity;
    ensemble_models_test_specificity(i) = ensemble_models_info{i}.test_specificity;
  end
  figure(2);
  clf;
  model_fig_path = fullfile(experiment_dir_path, 'incremental-performance.pdf');
  xlabel('training epoch'); ylabel('error');
  title('performance');
  grid on;
  hold on;
  plot(1:num_models_in_ensemble, ensemble_models_validation_accuracy, 'r-');
  plot(1:num_models_in_ensemble, ensemble_models_validation_sensitivity, 'g-');
  plot(1:num_models_in_ensemble, ensemble_models_validation_specificity, 'b-');
  plot(1:num_models_in_ensemble, ensemble_models_test_accuracy, 'r.--');
  plot(1:num_models_in_ensemble, ensemble_models_test_sensitivity, 'g.--');
  plot(1:num_models_in_ensemble, ensemble_models_test_specificity, 'b.--');
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
%_p-------------------------------------------------------------------------
function weighted_results = testAllEnsembleModelsOnTestImdb(ensemble_models_info, imdb)
% -------------------------------------------------------------------------
  fprintf('\n');
  afprintf(sprintf('[INFO] ENSEMBLE RESULTS ON TEST SET: \n'));
  printConsoleOutputSeparator();
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % Initial stuff
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  data_test = imdb.images.data(:,:,:,imdb.images.set == 3);
  labels_test = imdb.images.labels(imdb.images.set == 3);
  data_test_healthy = data_test(:,:,:,labels_test == 1);
  data_test_cancer = data_test(:,:,:,labels_test == 2);
  data_test_count = size(data_test, 4);
  data_test_healthy_count = size(data_test_healthy, 4);
  data_test_cancer_count = size(data_test_cancer, 4);

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % Construct IMDB
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  test_imdb = constructPartialImdb(data_test, labels_test, 3);

  H = {};
  B = zeros(1, numel(ensemble_models_info));
  for i = 1:numel(B)
    H{i} = ensemble_models_info{i}.model_net;
    B(i) = ensemble_models_info{i}.model_weight;
  end
  assert(numel(H) == numel(B))
  B = B / sum(B);

  weighted_test_set_predictions = zeros(data_test_count, 2);
  test_set_predictions_per_model = {};
  for i = 1:size(H, 2) % looping through all trained networks
    afprintf(sprintf('\n'));
    afprintf(sprintf('[INFO] Computing test set predictions for model #%d (healthy: %d, cancer: %d)...\n', ...
      i, ...
      data_test_healthy_count, ...
      data_test_cancer_count));
    net = H{i};
    test_set_predictions_per_model{i} = getPredictionsFromNetOnImdb(net, test_imdb, 3);
    [acc, sens, spec] = getAccSensSpec(labels_test, test_set_predictions_per_model{i}, true);
  end

  for i = 1:data_test_count
    % Calculating the total weight of the class labels from all the models
    % produced during boosting
    wt_healthy = 0; % class 1
    wt_cancer = 0; % class 2
    for j = 1:size(H, 2) % looping through all trained networks
       p = test_set_predictions_per_model{j}(i);
       if p == 2 % if is cancer
           wt_cancer = wt_cancer + B(j);
       else
           wt_healthy = wt_healthy + B(j);
       end
    end

    if (wt_cancer > wt_healthy)
        weighted_test_set_predictions(i,:) = [2 wt_cancer];
    else
        weighted_test_set_predictions(i,:) = [1 wt_healthy];
    end
  end

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 7. done, go treat yourself to something sugary!
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  printConsoleOutputSeparator();
  predictions_test = weighted_test_set_predictions(:, 1)';
  [weighted_acc, weighted_sens, weighted_spec] = getAccSensSpec(labels_test, predictions_test, false);
  afprintf(sprintf('Model weights: '))
  disp(B);
  afprintf(sprintf('[INFO] Weighted Acc: %3.6f\n', weighted_acc));
  afprintf(sprintf('[INFO] Weighted Sens: %3.6f\n', weighted_sens));
  afprintf(sprintf('[INFO] Weighted Spec: %3.6f\n', weighted_spec));
  weighted_results.acc = weighted_acc;
  weighted_results.sens = weighted_sens;
  weighted_results.spec = weighted_spec;

% -------------------------------------------------------------------------
function results = getKFoldResults(folds)
% -------------------------------------------------------------------------
  all_folds_acc = [];
  all_folds_sens = [];
  all_folds_spec = [];
  all_folds_ensemble_count = [];
  number_of_folds = numel(fields(folds));
  for i = 1:number_of_folds
    for j = 1:numel(folds.(sprintf('fold_%d', i)).ensemble_models_info)
      % results.(sprintf('fold_%d', i)).weight(j) = ...
      %   folds.(sprintf('fold_%d', i)).ensemble_models_info{j}.model_weight; % weight is normalized a bunch of times after each iter...
      results.(sprintf('fold_%d', i)).acc(j) = ...
        folds.(sprintf('fold_%d', i)).ensemble_models_info{j}.test_accuracy;
      results.(sprintf('fold_%d', i)).sens(j) = ...
        folds.(sprintf('fold_%d', i)).ensemble_models_info{j}.test_sensitivity;
      results.(sprintf('fold_%d', i)).spec(j) = ...
        folds.(sprintf('fold_%d', i)).ensemble_models_info{j}.test_specificity;
    end
    results.(sprintf('fold_%d', i)).weighted_acc = ...
      folds.(sprintf('fold_%d', i)).weighted_results.acc;
    results.(sprintf('fold_%d', i)).weighted_sens = ...
      folds.(sprintf('fold_%d', i)).weighted_results.sens;
    results.(sprintf('fold_%d', i)).weighted_spec = ...
      folds.(sprintf('fold_%d', i)).weighted_results.spec;
  end

  for i = 1:number_of_folds
    all_folds_acc(i) = folds.(sprintf('fold_%d', i)).weighted_results.acc;
    all_folds_sens(i) = folds.(sprintf('fold_%d', i)).weighted_results.sens;
    all_folds_spec(i) = folds.(sprintf('fold_%d', i)).weighted_results.spec;
    all_folds_ensemble_count(i) = numel(folds.(sprintf('fold_%d', i)).ensemble_models_info);
  end
  results.kfold_acc_avg = mean(all_folds_acc);
  results.kfold_sens_avg = mean(all_folds_sens);
  results.kfold_spec_avg = mean(all_folds_spec);
  results.kfold_ensemble_count_avg = mean(all_folds_ensemble_count);

  results.kfold_acc_std = std(all_folds_acc);
  results.kfold_sens_std = std(all_folds_sens);
  results.kfold_spec_std = std(all_folds_spec);
  results.kfold_ensemble_count_std = std(all_folds_ensemble_count);

% -------------------------------------------------------------------------
function saveKFoldResults(folds, results_file_path)
% -------------------------------------------------------------------------
  results = getKFoldResults(folds);
  saveStruct2File(results, results_file_path, 0);
% _p------------------------------------------------------------------------
function printKFoldResults(folds)
% -------------------------------------------------------------------------
  format shortG
  % for i = 1:numel(folds)
  %   afprintf(sprintf('Fold #%d Weighted RusBoost Performance:\n', i));
  %   disp(folds.(sprintf('fold_%d', i)).weighted_results);
  % end
  results = getKFoldResults(folds);
  afprintf(sprintf(' -- -- -- -- -- -- -- -- -- ALL FOLDS -- -- -- -- -- -- -- -- -- \n'));
  afprintf(sprintf(' -- -- -- -- -- -- -- -- -- TODO AMIR! -- -- -- -- -- -- -- -- -- \n'));
  % afprintf(sprintf('acc: %3.6f, std: %3.6f\n', mean(results.all_folds_acc), std(results.all_folds_acc)));
  % afprintf(sprintf('sens: %3.6f, std: %3.6f\n', mean(results.all_folds_sens), std(results.all_folds_sens)));
  % afprintf(sprintf('spec: %3.6f, std: %3.6f\n', mean(results.all_folds_spec), std(results.all_folds_spec)));
  % afprintf(sprintf('ensemble count: %3.6f, std: %3.6f\n', mean(results.all_folds_ensemble_count), std(results.all_folds_ensemble_count)));
