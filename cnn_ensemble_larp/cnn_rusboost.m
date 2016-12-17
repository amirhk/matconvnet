function [B, H] = main_cnn_rusboost()

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 0. some important parameter definition
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  opts.iteration_count = 50; % number of boosting iterations
  opts.dataset = 'prostate';
  opts.networkArch = 'prostatenet';
  opts.backpropDepth = 4;
  opts.weightInitSource = 'gen';
  opts.weightInitSequence = {'compRand', 'compRand', 'compRand'};
  opts.random_undersampling_ratio = (65/35);

  opts.imdbPath = fullfile(getDevPath(), '/matconvnet/data_1/_prostate/_saved_prostate_imdb.mat');
  opts.timeString = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.experimentDirPath = sprintf('data_rusboost/rusboost-%s-%s-%s', opts.dataset, opts.networkArch, opts.timeString);
  opts.allModelInfosPath = fullfile(opts.experimentDirPath, 'all_model_infos.mat');
  if ~exist(opts.experimentDirPath)
    mkdir(opts.experimentDirPath);
  end

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 1. take as input a pre-processed IMDB (augment cancer in training set, that's it!), say
  %   train: 94 patients
  %   test: 10 patients, ~1000 health, ~20 cancer
  % TODO: this can be extended to be say 10-fold ensemble larp, then average the folds
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  fprintf('[INFO] Loading saved imdb... ');
  imdb = load(opts.imdbPath);
  imdb = imdb.imdb;
  fprintf('done!\n');

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

  fprintf('\t[INFO] TRAINING SET: total: %d, healthy: %d, cancer: %d\n', ...
    data_train_count, ...
    data_train_healthy_count, ...
    data_train_cancer_count);
  fprintf('\t[INFO] TESTING SET: total: %d, healthy: %d, cancer: %d\n', ...
    data_test_count, ...
    data_test_healthy_count, ...
    data_test_cancer_count);

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

  training_resampled_imdb.images.data = [];   % filled in below
  training_resampled_imdb.images.labels = []; % filled in below
  training_resampled_imdb.images.set = [];    % filled in below
  training_resampled_imdb.meta.sets = {'train', 'val', 'test'};

  validation_imdb.images.data = data_train;
  validation_imdb.images.labels = labels_train;
  validation_imdb.images.set = 3 * ones(length(labels_train), 1);
  validation_imdb.meta.sets = {'train', 'val', 'test'};

  test_imdb.images.data = data_test;
  test_imdb.images.labels = labels_test;
  test_imdb.images.set = 3 * ones(length(labels_test), 1);
  test_imdb.meta.sets = {'train', 'val', 'test'};

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 4. go through T iterations of RUSBoost, each of which trains a CNN over E epochs
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  printOutputSeparator();
  all_model_infos = {};
  while t <= opts.iteration_count

    fprintf('\n[INFO] Boosting iteration #%d (attempt %d)...\n', t, count);

    % Resampling NEG_DATA with weights of positive example
    fprintf('\t[INFO] Resampling healthy and cancer data (ratio = 65/35)... ');
    [resampled_data, resampled_labels] = ...
      resampleData(data_train, labels_train, W(t, :), opts.random_undersampling_ratio);
    fprintf('done!\n');

    training_resampled_imdb.images.data = single(resampled_data);
    training_resampled_imdb.images.labels = single(resampled_labels);
    training_resampled_imdb.images.set = 1 * ones(length(resampled_labels), 1);

    % Weird. Need at least 1 test sample for cnn_train to work. TODO: this is because of TP stuff in cnn_train
    training_resampled_imdb.images.data = cat(4, training_resampled_imdb.images.data, resampled_data(:,:,:, end));
    training_resampled_imdb.images.labels = cat(2,training_resampled_imdb.images.labels, resampled_labels(end));
    training_resampled_imdb.images.set = cat(1, training_resampled_imdb.images.set, 3);

    fprintf('\t[INFO] Training model (healthy: %d, cancer: %d)...\n', ...
     numel(find(resampled_labels == 1)), ...
     numel(find(resampled_labels == 2)));
    [net, info] = cnn_amir( ...
      'imdb', training_resampled_imdb, ...
      'dataset', opts.dataset, ...
      'networkArch', opts.networkArch, ...
      'backpropDepth', opts.backpropDepth, ...
      'weightInitSource', opts.weightInitSource, ...
      'weightInitSequence', opts.weightInitSequence, ...
      'debugFlag', false);
    % fprintf('\tdone!\n');

    fprintf('\t[INFO] Computing predictions (healthy: %d, cancer: %d)...\n', ...
     data_train_healthy_count, ...
     data_train_cancer_count);
    % IMPORTANT NOTE: we randomly undersample when training a model, but then,
    % we use all of the training samples (in their order) to update weights.
    predictions = getPredictionsFromNetOnImdb(net, validation_imdb);
    % fprintf('done!\n');

    % Computing the pseudo loss of hypothesis 'model'
    fprintf('\t[INFO] Computing pseudo loss... ');
    cancer_to_healthy_ratio = data_train_cancer_count / data_train_healthy_count;
    loss = 0;
    for i = 1:data_train_count
        if labels_train(i) == predictions(i)
          continue;
        else
          if labels_train(i) == 2
            loss = loss + cancer_to_healthy_ratio * W(t, i);
          else
            loss = loss + W(t, i);
          end
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
       fprintf('\tToo many iterations have loss > 0.5\n');
       fprintf('\tAborting boosting...\n');
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

    % At the final iteration there is no need to update the weights any
    % further
    if t == opts.iteration_count
        break;
    end

    % Updating weight
    fprintf('\t[INFO] Updating weights... ');
    for i = 1:data_train_count
        if labels_train(i) == predictions(i)
            W(t + 1, i) = W(t, i) * beta;
        else
            W(t + 1, i) = W(t, i);
        end
    end
    fprintf('done!\n');

    % Normalizing the weight for the next iteration
    sum_W = sum(W(t + 1, :));
    for i = 1:data_train_count
        W(t + 1, i) = W(t + 1, i) / sum_W;
    end

    fprintf('\t[INFO] Saving model and info... ');
    [acc, sens, spec] = getAccSensSpec(labels_train, predictions);
    all_model_infos{t}.model_net = H{t};
    all_model_infos{t}.model_loss = L(t);
    all_model_infos{t}.model_weight = B(t);
    all_model_infos{t}.perf_accuracy = acc;
    all_model_infos{t}.perf_sensitivity = sens;
    all_model_infos{t}.perf_specificity = spec;
    all_model_infos{t}.train_healthy_count = numel(find(resampled_labels == 1));
    all_model_infos{t}.train_cancer_count = numel(find(resampled_labels == 2));
    all_model_infos{t}.validation_healthy_count = data_train_healthy_count;
    all_model_infos{t}.validation_cancer_count = data_train_cancer_count;
    all_model_infos{t}.validation_predictions = labels_train;
    all_model_infos{t}.validation_labels = labels_train;
    save(opts.allModelInfosPath, 'all_model_infos');
    fprintf('done!\n');
    fprintf('\t[INFO] Acc: %3.2f Sens: %3.2f Spec: %3.2f\n', ...
      all_model_infos{i}.perf_accuracy, ...
      all_model_infos{i}.perf_sensitivity, ...
      all_model_infos{i}.perf_specificity);


    % Incrementing loop counter
    t = t + 1;
  end

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 5. test on test set, keeping in mind beta's between each mode
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  printOutputSeparator();
  % The final hypothesis is calculated and tested on the test set
  % simulteneously.

  B = B / sum(B);

  test_set_prediction_overall = zeros(data_test_count, 2);
  test_set_predictions_per_model = {};
  for i = 1:size(H, 2) % looping through all trained networks
    fprintf('\n[INFO] Getting test set predictions for model #%d (healthy: %d, cancer: %d)...\n', ...
      i, ...
      data_test_healthy_count, ...
      data_test_cancer_count);
    net = H{i};
    test_set_predictions_per_model{i} = getPredictionsFromNetOnImdb(net, test_imdb);
    [acc, sens, spec] = getAccSensSpec(labels_test, test_set_predictions_per_model{i});
    fprintf('\t[INFO] Acc: %3.2f\n', acc);
    fprintf('\t[INFO] Sens: %3.2f\n', sens);
    fprintf('\t[INFO] Spec: %3.2f\n', spec);
  end

  for i = 1:data_test_count
    % Calculating the total weight of the class labels from all the models
    % produced during boosting
    wt_healthy = 0; % class 1
    wt_cancer = 0; % class 2
    for j = 1:size(H, 2) % looping through all trained networks
       p = test_set_predictions_per_model{j}(1);
       if p == 2 % if is cancer
           wt_cancer = wt_cancer + B(j);
       else
           wt_healthy = wt_healthy + B(j);
       end
    end

    if (wt_cancer > wt_healthy)
        test_set_prediction_overall(i,:) = [2 wt_cancer];
    else
        test_set_prediction_overall(i,:) = [1 wt_cancer]; % TODO: should this not be wt_healthy?!!?!?!
    end
  end

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % 6. done, go treat yourself to something sugary!
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  printOutputSeparator();
  predictions_test = test_set_prediction_overall(:, 1)';
  [acc, sens, spec] = getAccSensSpec(labels_test, predictions_test);
  fprintf('[INFO] Overall Acc: %3.2f\n', acc);
  fprintf('[INFO] Overall Sens: %3.2f\n', sens);
  fprintf('[INFO] Overall Spec: %3.2f\n', spec);
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
  normalized_weights = weights / min(weights);
  repeat_counts = ceil(normalized_weights);

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
    tmp = repmat(data(:,:,:,i), [1,1,1,sample_repeat_count]);
    upsampled_data(:,:,:, counter : counter + sample_repeat_count - 1) = tmp;
    counter = counter + sample_repeat_count;
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
  fprintf('\n-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- \n');

