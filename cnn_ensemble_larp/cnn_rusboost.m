function fh = cnnRusboost()
  % assign function handles so we can call these local functions from elsewhere
  fh.getInitialImdb = @getInitialImdb;
  fh.mainCNNRusboost = @mainCNNRusboost;
  fh.kFoldCNNRusboost = @kFoldCNNRusboost;
  fh.printWeightedRepeats = @printWeightedRepeats;
  fh.testAllModelsOnTestImdb = @testAllModelsOnTestImdb;

% -------------------------------------------------------------------------
function folds = kFoldCNNRusboost()
% -------------------------------------------------------------------------
  opts.numPatients = 104;
  opts.numberOfFolds = 10;
  afprintf(sprintf('[INFO] Running K-fold CNN Rusboost (K = %d)...\n', opts.numberOfFolds), 1);

  patients_per_fold = ceil(opts.numPatients / opts.numberOfFolds);
  random_patient_indices = randperm(104);
  folds = {};

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 1. randomly divide off patients into K folds
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  afprintf(sprintf('\n'));
  afprintf(sprintf('[INFO] Randomly dividing patients into K folds...\n'));
  for i = 1:opts.numberOfFolds
    start_index = 1 + (i - 1) * patients_per_fold;
    end_index = min(104, i * patients_per_fold);
    folds.(sprintf('fold_%d', i)).patient_indices = random_patient_indices(start_index : end_index);
  end

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 2. create a non-balanced, non-augmented imdb for each fold
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  opts.dataDir = fullfile(getDevPath(), 'matconvnet/data_1/_prostate');
  opts.imdbBalancedDir = fullfile(getDevPath(), 'matconvnet/data_1/balanced-prostate-prostatenet');
  opts.imdbBalancedPath = fullfile(getDevPath(), 'matconvnet/data_1/balanced-prostate-prostatenet/imdb.mat');
  opts.leaveOutType = 'special';
  opts.contrastNormalization = true;
  opts.whitenData = true;
  imdbs = {}; % separate so don't have to save ~1.5 GB of imdbs!!!
  for i = 1:opts.numberOfFolds
    afprintf(sprintf('\n'));
    afprintf(sprintf('[INFO] Constructing imdb for fold #%d...\n', i));
    opts.leaveOutIndices = folds.(sprintf('fold_%d', i)).patient_indices;
    imdb = constructProstateImdb(opts);
    imdbs{i} = imdb;
    % folds.(sprintf('fold_%d', i)).imdb = imdb;
    afprintf(sprintf('[INFO] done!\n'));
  end

  all_folds_acc = [];
  all_folds_sens = [];
  all_folds_spec = [];

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 2. train ensemble larp for each fold!
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  opts.timeString = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.experimentDirParentPath = fullfile('data_rusboost', sprintf('k-fold-rusboost-%s', opts.timeString));
  for i = 1:opts.numberOfFolds
    afprintf(sprintf('[INFO] Running cnn_rusboost on fold #%d...\n', i));
    [ ...
      folds.(sprintf('fold_%d', i)).ensemble_models_info, ...
      folds.(sprintf('fold_%d', i)).weighted_results, ...
    ] = mainCNNRusboost(imdbs{i}, opts.experimentDirParentPath);
    % ] = mainCNNRusboost(folds.(sprintf('fold_%d', i)).imdb, opts.experimentDirParentPath);
    all_folds_acc(i) = folds.(sprintf('fold_%d', i)).weighted_results.acc;
    all_folds_sens(i) = folds.(sprintf('fold_%d', i)).weighted_results.sens;
    all_folds_spec(i) = folds.(sprintf('fold_%d', i)).weighted_results.spec;

    folds.all_folds_acc = all_folds_acc;  % overwrite
    folds.all_folds_sens = all_folds_sens;  % overwrite
    folds.all_folds_spec = all_folds_spec;  % overwrite
    save(opts.experimentDirParentPath, 'folds'); % overwrite and save
  end

  afprintf(sprintf('[INFO] Finished running K-fold CNN Rusboost (K = %d)...\n', opts.numberOfFolds), 1);
  afprintf(sprintf('[INFO] k-fold acc avg: %3.2f std: %3.2f\n', mean(all_folds_acc), std(all_folds_acc)));
  afprintf(sprintf('[INFO] k-fold sens avg: %3.2f std: %3.2f\n', mean(all_folds_sens), std(all_folds_sens)));
  afprintf(sprintf('[INFO] k-fold spec avg: %3.2f std: %3.2f\n', mean(all_folds_spec), std(all_folds_spec)));

% -------------------------------------------------------------------------
function [ensemble_models_info, weighted_results] = mainCNNRusboost(imdb, experimentDirParentPath)
% -------------------------------------------------------------------------
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 0. take as input a pre-processed IMDB (augment cancer in training set, that's it!), say
  %   train: 94 patients
  %   test: 10 patients, ~1000 health, ~20 cancer
  % TODO: this can be extended to be say 10-fold ensemble larp, then average the folds
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  if nargin == 0
    imdb = getInitialImdb();
    experimentDirParentPath = 'data_rusboost';
  else
    % TODO: only input currently designed for is imdb (for k-fold)!
    imdb = imdb;
    experimentDirParentPath = experimentDirParentPath;
  end

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 1. some important parameter definition
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  opts.iteration_count = 5; % number of boosting iterations
  opts.dataset = 'prostate';
  opts.networkArch = 'prostatenet';
  opts.backpropDepth = 4;
  opts.weightInitSource = 'gen';
  opts.weightInitSequence = {'compRand', 'compRand', 'compRand'};
  opts.random_undersampling_ratio = (65/35);

  opts.timeString = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.experimentDirPath = fullfile(experimentDirParentPath, sprintf('rusboost-%s-%s-%s', opts.dataset, opts.networkArch, opts.timeString));
  opts.allModelInfosPath = fullfile(opts.experimentDirPath, 'ensemble_models_info.mat');
  if ~exist(opts.experimentDirPath)
    mkdir(opts.experimentDirPath);
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

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 5. go through T iterations of RUSBoost, each of which trains a CNN over E epochs
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  printOutputSeparator();
  ensemble_models_info = {};
  while t <= opts.iteration_count
    afprintf(sprintf('\n'));
    afprintf(sprintf('[INFO] Boosting iteration #%d (attempt %d)...\n', t, count));

    % Resampling NEG_DATA with weights of positive example
    afprintf(sprintf('[INFO] Resampling healthy and cancer data (ratio = 65/35)... '));
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
    [net, info] = cnn_amir( ...
      'imdb', training_resampled_imdb, ...
      'dataset', opts.dataset, ...
      'networkArch', opts.networkArch, ...
      'backpropDepth', opts.backpropDepth, ...
      'weightInitSource', opts.weightInitSource, ...
      'weightInitSequence', opts.weightInitSequence, ...
      'debugFlag', false);

    afprintf(sprintf('[INFO] Computing validation set predictions (healthy: %d, cancer: %d)...\n', ...
      data_train_healthy_count, ...
      data_train_cancer_count));
    % IMPORTANT NOTE: we randomly undersample when training a model, but then,
    % we use all of the training samples (in their order) to update weights.
    predictions = getPredictionsFromNetOnImdb(net, validation_imdb);

    % Computing the pseudo loss of hypothesis 'model'
    afprintf(sprintf('[INFO] Computing pseudo loss... '));
    cancer_to_healthy_ratio = 1 / (data_train_cancer_count / data_train_healthy_count);
    loss = 0;
    for i = 1:data_train_count
      if labels_train(i) == predictions(i)
        continue;
      else
        loss = loss + W(t, i);
      end
    end
    % for i = 1:data_train_count
    %   if labels_train(i) == predictions(i)
    %     continue;
    %   else
    %     if labels_train(i) == 2
    %       loss = loss + 10 * cancer_to_healthy_ratio * W(t, i);
    %     else
    %       loss = loss + W(t, i);
    %     end
    %   end
    % end
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
    % for i = 1:data_train_count
    %   if labels_train(i) == predictions(i)
    %     W(t + 1, i) = W(t, i) * beta;
    %   else
    %     W(t + 1, i) = W(t, i);
    %   end
    % end
    for i = 1:data_train_count
      if labels_train(i) == predictions(i)
        W(t + 1, i) = W(t, i) * beta;
      else
        if labels_train(i) == 2
          W(t + 1, i) = cancer_to_healthy_ratio * W(t, i);
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

    afprintf(sprintf('[INFO] Saving model and info... '));
    [acc, sens, spec] = getAccSensSpec(labels_train, predictions);
    ensemble_models_info{t}.model_net = H{t};
    ensemble_models_info{t}.model_loss = L(t);
    ensemble_models_info{t}.model_weight = B(t);
    ensemble_models_info{t}.perf_accuracy = acc;
    ensemble_models_info{t}.perf_sensitivity = sens;
    ensemble_models_info{t}.perf_specificity = spec;
    ensemble_models_info{t}.train_healthy_count = numel(find(resampled_labels == 1));
    ensemble_models_info{t}.train_cancer_count = numel(find(resampled_labels == 2));
    ensemble_models_info{t}.validation_healthy_count = data_train_healthy_count;
    ensemble_models_info{t}.validation_cancer_count = data_train_cancer_count;
    ensemble_models_info{t}.validation_predictions = labels_train;
    ensemble_models_info{t}.validation_labels = labels_train;
    ensemble_models_info{t}.validation_weights_pre_update = W(t,:);
    ensemble_models_info{t}.validation_weights_post_update = W(t + 1,:);
    save(opts.allModelInfosPath, 'ensemble_models_info');
    fprintf('done!\n');
    afprintf(sprintf('[INFO] Acc: %3.2f Sens: %3.2f Spec: %3.2f\n', ...
      ensemble_models_info{t}.perf_accuracy, ...
      ensemble_models_info{t}.perf_sensitivity, ...
      ensemble_models_info{t}.perf_specificity));

    % Incrementing loop counter
    t = t + 1;
  end

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 6. test on test set, keeping in mind beta's between each mode
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % The final hypothesis is calculated and tested on the test set simulteneously
  printOutputSeparator();
  weighted_results = testAllModelsOnTestImdb(ensemble_models_info, imdb);
  printOutputSeparator();

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

% TODO: copy to central file
% -------------------------------------------------------------------------
function fn = getBatch()
% -------------------------------------------------------------------------
  fn = @(x,y) getSimpleNNBatch(x,y);

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
  images = imdb.images.data(:,:,:,batch);
  labels = imdb.images.labels(1,batch);
  if rand > 0.5, images=fliplr(images); end

% -------------------------------------------------------------------------
function predictions = getPredictionsFromNetOnImdb(net, imdb)
% -------------------------------------------------------------------------
  [net, info] = cnn_train(net, imdb, getBatch(), ...
    'debugFlag', false, ...
    'continue', false, ...
    'numEpochs', 1, ...
    'val', find(imdb.images.set == 3));
  predictions = info.predictions;

% -------------------------------------------------------------------------
function [acc, sens, spec] = getAccSensSpec(labels, predictions)
% -------------------------------------------------------------------------
  positive_class_num = 2;
  negative_class_num = 1;
  TP = sum((labels == predictions) .* (predictions == positive_class_num)); % TP
  TN = sum((labels == predictions) .* (predictions == negative_class_num)); % TN
  FP = sum((labels ~= predictions) .* (predictions == positive_class_num)); % FP
  FN = sum((labels ~= predictions) .* (predictions == negative_class_num)); % FN
  acc = (TP + TN) / (TP + TN + FP + FN);
  sens = TP / (TP + FN);
  spec = TN / (TN + FP);

% -------------------------------------------------------------------------
function printOutputSeparator()
% -------------------------------------------------------------------------
  afprintf(sprintf('\n'));
  afprintf(sprintf('-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- \n'), 1);

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
function weighted_results = testAllModelsOnTestImdb(ensemble_models_info, imdb)
% -------------------------------------------------------------------------
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
    test_set_predictions_per_model{i} = getPredictionsFromNetOnImdb(net, test_imdb);
    [acc, sens, spec] = getAccSensSpec(labels_test, test_set_predictions_per_model{i});
    % afprintf(sprintf('[INFO] Acc: %3.2f Sens: %3.2f Spec: %3.2f\n', acc, sens, spec));
    afprintf(sprintf('[INFO] Acc: %3.2f\n', acc));
    afprintf(sprintf('[INFO] Sens: %3.2f\n', sens));
    afprintf(sprintf('[INFO] Spec: %3.2f\n', spec));
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
  printOutputSeparator();
  predictions_test = weighted_test_set_predictions(:, 1)';
  [weighted_acc, weighted_sens, weighted_spec] = getAccSensSpec(labels_test, predictions_test);
  afprintf(sprintf('Model weights: '))
  disp(B);
  afprintf(sprintf('[INFO] Weighted Acc: %3.2f\n', weighted_acc));
  afprintf(sprintf('[INFO] Weighted Sens: %3.2f\n', weighted_sens));
  afprintf(sprintf('[INFO] Weighted Spec: %3.2f\n', weighted_spec));
  weighted_results.acc = weighted_acc;
  weighted_results.sens = weighted_sens;
  weighted_results.spec = weighted_spec;

% -------------------------------------------------------------------------
function printWeightedRepeats(ensemble_models_info)
% -------------------------------------------------------------------------
  format shortG
  for i = 1:numel(ensemble_models_info)
    vw = ensemble_models_info{i}.validation_weights_post_update;
    vl = ensemble_models_info{i}.validation_labels;
    healthy_repeats = ceil(vw(vl == 1) / min(vw));
    cancer_repeats = ceil(vw(vl == 2) / min(vw));
    tmp = tabulate(healthy_repeats);
    healthy_occurances = tmp(tmp(:,2) > 0, :);
    disp(healthy_occurances);
    tmp = sum(healthy_occurances(:,1) .* healthy_occurances(:,2));
    fprintf('weighted healthy repeats: %d (really %d)\n', tmp, round(tmp * (1856 / 11380) * (65 / 35)));
    % tmp = healthy_occurances;
    % healthy_occurances(healthy_occurances(:,1) > 25,1) = 25;
    % max_limited_healthy_occurances = healthy_occurances;
    % tmp = sum(max_limited_healthy_occurances(:,1) .* max_limited_healthy_occurances(:,2));
    % fprintf('max allowed weighted healthy repeats: %d (really %d)\n', tmp, round(tmp * (1856 / 11380) * (65 / 35)));
    fprintf('\n-- -- -- -- --\n');
    tmp = tabulate(cancer_repeats);
    cancer_occurances = tmp(tmp(:,2) > 0, :);
    disp(cancer_occurances);
    tmp = sum(cancer_occurances(:,1) .* cancer_occurances(:,2));
    fprintf('weighted cancer repeats: %d\n', tmp);
    fprintf('\n\n== == == == == == == == == == == == == == == == == == == == == ==\n\n\n');
  end



% fh = cnn_rusboost();
% imdb  = fh.getInitialImdb();
% fh.testAllModelsOnTestImdb(ensemble_models_info, imdb)

