function fh = imdbTwoClassUtils()
  % assign function handles so we can call these local functions from elsewhere
  fh.getImdbInfo = @getImdbInfo;
  fh.balanceImdb = @balanceImdb;
  fh.resampleData = @resampleData;
  fh.constructPartialImdb = @constructPartialImdb;
  fh.constructTwoClassUnbalancedImdb = @constructTwoClassUnbalancedImdb;

% -------------------------------------------------------------------------
function [ ...
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
  labels_test] = getImdbInfo(imdb, print_info)
% -------------------------------------------------------------------------
  % enforce row vector before doing bsxfun
  imdb.images.labels = reshape(imdb.images.labels, 1, prod(size(imdb.images.labels)));
  imdb.images.set = reshape(imdb.images.set, 1, prod(size(imdb.images.set)));

  % indices (WARNING: these are indices in imdb.images.data, not in data_train or data_test)
  data = imdb.images.data;
  data_train_indices = imdb.images.set == 1;
  data_train_positive_indices = bsxfun(@and, imdb.images.labels == 2, imdb.images.set == 1);
  data_train_negative_indices = bsxfun(@and, imdb.images.labels == 1, imdb.images.set == 1);
  data_test_indices = imdb.images.set == 3;
  data_test_positive_indices = bsxfun(@and, imdb.images.labels == 2, imdb.images.set == 3);
  data_test_negative_indices = bsxfun(@and, imdb.images.labels == 1, imdb.images.set == 3);

  % train set
  data_train = imdb.images.data(:,:,:,data_train_indices);
  data_train_positive = imdb.images.data(:,:,:,data_train_positive_indices);
  data_train_negative = imdb.images.data(:,:,:,data_train_negative_indices);
  data_train_count = size(data_train, 4);
  data_train_positive_count = size(data_train_positive, 4);
  data_train_negative_count = size(data_train_negative, 4);
  labels_train = imdb.images.labels(data_train_indices);

  % test set
  data_test = imdb.images.data(:,:,:,data_test_indices);
  data_test_positive = imdb.images.data(:,:,:,data_test_positive_indices);
  data_test_negative = imdb.images.data(:,:,:,data_test_negative_indices);
  data_test_count = size(data_test, 4);
  data_test_positive_count = size(data_test_positive, 4);
  data_test_negative_count = size(data_test_negative, 4);
  labels_test = imdb.images.labels(data_test_indices);

  if print_info
    afprintf(sprintf('[INFO] imdb info:\n'));
    afprintf(sprintf('[INFO] TRAINING SET: total: %d, positive: %d, negative: %d\n', ...
      data_train_count, ...
      data_train_positive_count, ...
      data_train_negative_count));
    afprintf(sprintf('[INFO] TESTING SET: total: %d, positive: %d, negative: %d\n', ...
      data_test_count, ...
      data_test_positive_count, ...
      data_test_negative_count));
  end

% TODO: this should really be resampleImdb w/ a bunch of options to resample
% an input class {pos, neg} in an input set {1,3}....
% -------------------------------------------------------------------------
function [resampled_data, resampled_labels] = resampleData(data, labels, weights, ratio)
% -------------------------------------------------------------------------
  % -------------------------------------------------------------------------
  % Initial stuff
  % -------------------------------------------------------------------------
  data_positive = data(:,:,:,labels == 2);
  data_negative = data(:,:,:,labels == 1);
  data_count = size(data, 4);
  data_positive_count = size(data_positive, 4);
  data_negative_count = size(data_negative, 4);
  data_positive_indices = find(labels == 2);
  data_negative_indices = find(labels == 1);

  % -------------------------------------------------------------------------
  % Random Under-sampling (RUS): Negative Data
  % -------------------------------------------------------------------------
  weights_negative_indices = weights(1, data_negative_indices);
  downsampled_data_negative_count = round(data_positive_count * ratio);
  % just random sampling.... horrendously wrong
  % downsampled_data_negative_indices = randsample(data_negative_indices, downsampled_data_negative_count, false);
  % TODO: the line below is weighted sampling w/ replacement; the most accurate
  % way is to weighted randsample w/o replacement
  downsampled_data_negative_indices = randsample( ...
    data_negative_indices, ...
    downsampled_data_negative_count, ...
    true, ...
    weights_negative_indices);
  downsampled_data_negative = data(:,:,:, downsampled_data_negative_indices);


  % -------------------------------------------------------------------------
  % Weighted Upsampling (more weight -> more repeat): Negative & Positive Data
  % -------------------------------------------------------------------------
  max_repeat_positive = 200;
  max_repeat_negative = 25;
  normalized_weights = weights / min(weights);
  repeat_counts = ceil(normalized_weights);
  for j = data_positive_indices
    if repeat_counts(j) > max_repeat_positive
      repeat_counts(j) = max_repeat_positive;
    end
  end
  for j = data_negative_indices
    if repeat_counts(j) > max_repeat_negative
      repeat_counts(j) = max_repeat_negative;
    end
  end

  positive_repeat_counts = repeat_counts(data_positive_indices);
  negative_repeat_counts = repeat_counts(downsampled_data_negative_indices);

  upsampled_data_positive = upsample(data_positive, positive_repeat_counts);
  upsampled_data_negative = upsample(downsampled_data_negative, negative_repeat_counts);

  % -------------------------------------------------------------------------
  % Putting it all together
  % -------------------------------------------------------------------------
  resampled_data_positive_count = size(upsampled_data_positive, 4);
  resampled_data_negative_count = size(upsampled_data_negative, 4);
  resampled_data_all = cat(4, upsampled_data_positive, upsampled_data_negative);
  resampled_labels_all = cat( ...
    2, ...
    2 * ones(1, resampled_data_positive_count), ...
    1 * ones(1, resampled_data_negative_count));

  % -------------------------------------------------------------------------
  % Shuffle this to mixup order of negative and positive in imdb so we don't
  % have the CNN overtrain in 1 particular direction. Only shuffling for
  % training; later weights are calculated and updated for all training data.
  % -------------------------------------------------------------------------
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
function imdb = balanceImdb(imdb, set_name, balance_type)
  % set_name = {'train', 'test'}
  % balance_type = {'downsample', 'augment'}
% -------------------------------------------------------------------------

  % -------------------------------------------------------------------------
  %                                                                  get info
  % -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Initial imdb info...\n'));
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
  ] = getImdbInfo(imdb, true);

  switch set_name
    case 'train'
      assert(data_train_positive_count < data_train_negative_count);
    case 'test'
      assert(data_test_positive_count < data_test_negative_count);
  end

  data_train_indices = find(data_train_indices);
  data_train_positive_indices = find(data_train_positive_indices);
  data_train_negative_indices = find(data_train_negative_indices);
  data_test_indices = find(data_test_indices);
  data_test_positive_indices = find(data_test_positive_indices);
  data_test_negative_indices = find(data_test_negative_indices);

  % -------------------------------------------------------------------------
  %                                                                      beef
  % -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Balancing `%s` set in imdb (balance type: `%s`)...\n', set_name, balance_type));
  switch balance_type
    case 'downsample'
      switch set_name
        case 'train'
          downsampled_data_train_negative_indices = randsample( ...
            data_train_negative_indices, ...
            data_train_positive_count, ...
            false);
          downsampled_data_train_negative = data(:,:,:,downsampled_data_train_negative_indices);
          new_data_train_positive = data_train_positive;
          new_data_train_negative = downsampled_data_train_negative;
          new_data_test_positive = data_test_positive;
          new_data_test_negative = data_test_negative;
        case 'test'
          downsampled_data_test_negative_indices = randsample( ...
            data_test_negative_indices, ...
            data_test_positive_count, ...
            false);
          downsampled_data_test_negative = data(:,:,:,downsampled_data_test_negative_indices);
          new_data_train_positive = data_train_positive;
          new_data_train_negative = data_train_negative;
          new_data_test_positive = data_test_positive;
          new_data_test_negative = downsampled_data_test_negative;
      end
    case 'augment'
      % TODO
  end

  % -------------------------------------------------------------------------
  %                                                   putting it all together
  % -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Putting it all together...\n'));
  new_data = cat( ...
    4, ...
    new_data_train_positive, ...
    new_data_train_negative, ...
    new_data_test_positive, ...
    new_data_test_negative);
  new_labels = cat( ...
    2, ...
    2 * ones(1, size(new_data_train_positive, 4)), ...
    1 * ones(1, size(new_data_train_negative, 4)), ...
    2 * ones(1, size(new_data_test_positive, 4)), ...
    1 * ones(1, size(new_data_test_negative, 4)));
  new_set = cat( ...
    2, ...
    1 * ones(1, size(new_data_train_positive, 4)), ...
    1 * ones(1, size(new_data_train_negative, 4)), ...
    3 * ones(1, size(new_data_test_positive, 4)), ...
    3 * ones(1, size(new_data_test_negative, 4)));

  % -------------------------------------------------------------------------
  %                                                                   shuffle
  % -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Shuffling data...\n'));
  ix = randperm(size(new_data, 4));
  new_data = new_data(:,:,:,ix);
  new_labels = new_labels(ix);
  new_set = new_set(ix);


  % -------------------------------------------------------------------------
  %                                                                   finally
  % -------------------------------------------------------------------------
  imdb.images.data = single(new_data);
  imdb.images.labels = single(new_labels);
  imdb.images.set = single(new_set);
  afprintf(sprintf('[INFO] Final imdb info...\n'));
  [~] = getImdbInfo(imdb, true);
  afprintf(sprintf('done!\n'));

% -------------------------------------------------------------------------
function imdb = constructTwoClassUnbalancedImdb(imdb, positive_class_number, negative_class_number, unbalance_inverse_ratio)
% -------------------------------------------------------------------------
  % enforce row vector before doing bsxfun
  imdb.images.labels = reshape(imdb.images.labels, 1, prod(size(imdb.images.labels)));
  imdb.images.set = reshape(imdb.images.set, 1, prod(size(imdb.images.set)));

  % indices
  train_positive_indices = bsxfun(@and, imdb.images.labels == positive_class_number, imdb.images.set == 1);
  train_negative_indices = bsxfun(@and, imdb.images.labels == negative_class_number, imdb.images.set == 1);
  test_positive_indices = bsxfun(@and, imdb.images.labels == positive_class_number, imdb.images.set == 3);
  test_negative_indices = bsxfun(@and, imdb.images.labels == negative_class_number, imdb.images.set == 3);

  % train set
  data_train_positive = imdb.images.data(:,:,:,train_positive_indices);
  data_train_negative = imdb.images.data(:,:,:,train_negative_indices);
  downsampled_data_train_positive_indices = randsample(size(data_train_positive, 4), floor(size(data_train_positive, 4) / unbalance_inverse_ratio));
  downsampled_data_train_positive = data_train_positive(:,:,:, downsampled_data_train_positive_indices);

  data_train_positive = downsampled_data_train_positive;
  data_train_negative = data_train_negative;
  labels_train_positive = 2 * ones(1, size(data_train_positive, 4));
  labels_train_negative = 1 * ones(1, size(data_train_negative, 4));

  data_train = cat(4, data_train_positive, data_train_negative);
  labels_train = cat(2, labels_train_positive, labels_train_negative);

  % shuffle
  ix = randperm(size(data_train, 4));
  data_train = data_train(:,:,:,ix);
  labels_train = labels_train(ix);

  % test set
  data_test_positive = imdb.images.data(:,:,:,test_positive_indices);
  data_test_negative = imdb.images.data(:,:,:,test_negative_indices);
  labels_test_positive = 2 * ones(1, size(data_test_positive, 4));
  labels_test_negative = 1 * ones(1, size(data_test_negative, 4));

  data_test = cat(4, data_test_positive, data_test_negative);
  labels_test = cat(2, labels_test_positive, labels_test_negative);

  % put it all together
  imdb.images.data = single(cat(4, data_train, data_test));
  imdb.images.labels = single(cat(2, labels_train, labels_test));
  imdb.images.set = single(cat(2, 1 * ones(1, size(labels_train, 2)), 3 * ones(1, size(labels_test, 2))));
  % imdb.images.set = (round(rand(1,length(labels_train))) * 2) + 1; % randomly assign to either set 1 or set 3
  imdb.meta.sets = {'train', 'val', 'test'};
  imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false);
