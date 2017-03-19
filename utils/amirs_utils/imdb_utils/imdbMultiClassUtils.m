% -------------------------------------------------------------------------
function fh = imdbMultiClassUtils()
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

  % assign function handles so we can call these local functions from elsewhere
  fh.saveImdb = @saveImdb;
  fh.getImdbInfo = @getImdbInfo;
  fh.subsampleImdb = @subsampleImdb;
  fh.balanceAllClassesInImdb = @balanceAllClassesInImdb;

% -------------------------------------------------------------------------
function [ ...
  data_train_per_class, ...
  data_train_count_per_class, ...
  data_train_indices_per_class, ...
  data_test_per_class, ...
  data_test_count_per_class, ...
  data_test_indices_per_class] = getImdbInfo(imdb, debug_flag)
% -------------------------------------------------------------------------
  % enforce row vector before doing bsxfun
  imdb.images.labels = reshape(imdb.images.labels, 1, prod(size(imdb.images.labels)));
  imdb.images.set = reshape(imdb.images.set, 1, prod(size(imdb.images.set)));

  unique_classes = unique(imdb.images.labels);

  for class_number = unique_classes
    % train
    data_train_indices_per_class{class_number} = bsxfun(@and, imdb.images.labels == class_number, imdb.images.set == 1);
    data_train_count_per_class{class_number} = sum(data_train_indices_per_class{class_number});
    data_train_per_class{class_number} = imdb.images.data(:,:,:,data_train_indices_per_class{class_number});
    % test
    data_test_indices_per_class{class_number} = bsxfun(@and, imdb.images.labels == class_number, imdb.images.set == 3);
    data_test_count_per_class{class_number} = sum(data_test_indices_per_class{class_number});
    data_test_per_class{class_number} = imdb.images.data(:,:,:,data_test_indices_per_class{class_number});
  end

  if debug_flag
    afprintf(sprintf('[INFO] imdb info:\n'));
    afprintf(sprintf('[INFO] TRAINING SET:\n'));
    afprintf(sprintf('[INFO] total: %d\n', sum([data_train_count_per_class{:}], 2)), 1);
    for class_number = unique_classes
      afprintf(sprintf('[INFO] class #%d: %d\n', class_number, data_train_count_per_class{class_number}), 1);
    end
    afprintf(sprintf('[INFO] TESTING SET:\n'));
    afprintf(sprintf('[INFO] total: %d\n', sum([data_test_count_per_class{:}], 2)), 1);
    for class_number = unique_classes
      afprintf(sprintf('[INFO] class #%d: %d\n', class_number, data_test_count_per_class{class_number}), 1);
    end
  end

% -------------------------------------------------------------------------
function imdb = balanceAllClassesInImdb(imdb, set_name, balance_count)
  % set_name = {'train', 'test'}
  % balance_count = {38, 100, 277, 707, 1880, 5000}
% -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Balancing all `%sing` classes of imdb to #%d samples...\n', set_name, balance_count));
  unique_classes_count = numel(unique(imdb.images.labels));
  % Balance train sets as desired
  for class_number = 1:unique_classes_count
    imdb = subsampleImdb(imdb, set_name, class_number, balance_count);
  end
  afprintf(sprintf('[INFO] Done.\n'));


% -------------------------------------------------------------------------
function imdb = subsampleImdb(imdb, set_name, class_number, subsample_count)
  % set_name = {'train', 'test'}
  % class_number = {1, 2, 3, ...}
  % subsample_count = {38, 100, 277, 707, 1880, 5000}
% -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Subsampling imdb (`%sing` samples from class #%d)...\n', set_name, class_number));
  [ ...
    data_train_per_class, ...
    ~, ... % data_train_count_per_class, ...
    ~, ... % data_train_indices_per_class, ...
    data_test_per_class, ...
    ~, ... % data_test_count_per_class, ...
    ~, ... % data_test_indices_per_class, ...
  ] = getImdbInfo(imdb, 0);

  switch set_name
    case 'train'
      data_for_set_and_class = data_train_per_class{class_number};
      subsampled_data_indices = randsample(size(data_for_set_and_class, 4), subsample_count);
      subsampled_data = data_for_set_and_class(:,:,:,subsampled_data_indices);
      data_train_per_class{class_number} = subsampled_data;
    case 'test'
      data_for_set_and_class = data_test_per_class{class_number};
      subsampled_data_indices = randsample(size(data_for_set_and_class, 4), subsample_count);
      subsampled_data = data_for_set_and_class(:,:,:,subsampled_data_indices);
      data_test_per_class{class_number} = subsampled_data;
  end
  imdb = constructImdbHelper(data_train_per_class, data_test_per_class);


% -------------------------------------------------------------------------
function imdb = constructImdbHelper(data_train_per_class, data_test_per_class)
% -------------------------------------------------------------------------
  % train set
  data_train = cat(4, data_train_per_class{:});
  labels_train = zeros(1, size(data_train, 4));
  index = 1;
  for i = 1:numel(data_train_per_class)
    for j = 1:size(data_train_per_class{i}, 4)
      labels_train(index) = i;
      index = index + 1;
    end
  end

  % test set
  data_test = cat(4, data_test_per_class{:});
  labels_test = zeros(1, size(data_test, 4));
  index = 1;
  for i = 1:numel(data_test_per_class)
    for j = 1:size(data_test_per_class{i}, 4)
      labels_test(index) = i;
      index = index + 1;
    end
  end

  % put it all together
  data = single(cat(4, data_train, data_test));
  labels = single(cat(2, labels_train, labels_test));
  set = single(cat(2, 1 * ones(1, size(labels_train, 2)), 3 * ones(1, size(labels_test, 2))));

  % shuffle
  ix = randperm(size(data, 4));
  data = data(:,:,:,ix);
  labels = labels(ix);
  set = set(ix);

  % put it all together
  imdb.images.data = data;
  imdb.images.labels = labels;
  imdb.images.set = set;
  imdb.meta.sets = {'train', 'val', 'test'};


%-------------------------------------------------------------------------
function saveImdb(dataset, imdb, train_balance_count, test_balance_count)
%-------------------------------------------------------------------------
afprintf(sprintf('[INFO] Saving imdb...\n'));
  save_file_name = sprintf( ...
    ... % 'saved-multi-class-%s-train-balance-%d-test-balance-%d', ...
    'saved-multi-class-%s-train-balance-%d-test-balance-%s', ...
    dataset, ...
    train_balance_count, ...
    test_balance_count);
  save(save_file_name, 'imdb');
