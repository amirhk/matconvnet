% --------------------------------------------------------------------
function imdb = constructImageNetTinyImdb(input_opts)
% --------------------------------------------------------------------
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

  afprintf(sprintf('[INFO] Constructing ImageNet Tiny imdb...\n'));

  class_name_to_class_label_mapping = getClassNameToClassLabelMapping(input_opts.imdb.data_dir);
  % keyboard

  [data_train, labels_train, set_train] = getTrainingDataAndLabels(input_opts.imdb.data_dir, class_name_to_class_label_mapping);
  [data_test, labels_test, set_test] = getValidationDataAndLabels(input_opts.imdb.data_dir, class_name_to_class_label_mapping);

  afprintf(sprintf('[INFO] Concatinating training data and validation data...\n'));
  data = cat(4, data_train, data_test);
  labels = cat(1, labels_train, labels_test);
  set = cat(1, set_train, set_test);
  afprintf(sprintf('[INFO] done!\n'));

  assert(length(labels) == length(set));
  total_number_of_samples = length(labels);

  % remove mean in any case
  afprintf(sprintf('[INFO] Mean substracting...\n'));
  data_mean = mean(data(:,:,:,set == 1), 4);
  data = bsxfun(@minus, data, data_mean);
  afprintf(sprintf('[INFO] done!\n'));

  % shuffle
  afprintf(sprintf('[INFO] Shuffling samples...\n'));
  ix = randperm(total_number_of_samples);
  imdb.images.data = data(:,:,:,ix);
  imdb.images.labels = labels(ix);
  imdb.images.set = set(ix);
  afprintf(sprintf('[INFO] done!\n'));

  % fh = imdbMultiClassUtils;
  % fh.getImdbInfo(imdb, 1);
  afprintf(sprintf('done!\n\n'));


% --------------------------------------------------------------------
function [data, labels, set] = getTrainingDataAndLabels(data_dir, class_name_to_class_label_mapping)
% --------------------------------------------------------------------
  afprintf(sprintf('[INFO] Retrieving training data...\n'));

  expected_number_of_classes = 200;
  expected_number_of_samples_per_class = 500;
  expeced_total_number_of_samples = expected_number_of_classes * expected_number_of_samples_per_class;

  % max_number_of_samples_per_class = 500;
  % expeced_total_number_of_samples = expected_number_of_classes * max_number_of_samples_per_class;

  fileID = fopen(fullfile(data_dir, 'wnids.txt'), 'r');
  formatSpec = '%s';
  tmp = textscan(fileID, formatSpec);
  class_names_cell_array = tmp{1};
  number_of_classes = numel(class_names_cell_array);
  assert(number_of_classes == expected_number_of_classes);

  target_sample_size = 32;

  tmp_data = zeros(target_sample_size, target_sample_size, 3, expeced_total_number_of_samples);
  tmp_labels = zeros(expeced_total_number_of_samples, 1);

  sample_index = 1;
  for i = 1 : number_of_classes
    afprintf(sprintf('[INFO] Loading samples from class # %d / %d\n', i, number_of_classes),  1);
    class_name = class_names_cell_array{i};
    class_label = i;
    all_training_sample_files_for_class = dir(fullfile(data_dir, 'train', class_name, 'images', '*.JPEG'));
    assert(numel(all_training_sample_files_for_class) == expected_number_of_samples_per_class);
    for j = 1 : numel(all_training_sample_files_for_class)
    % for j = 1 : max_number_of_samples_per_class
      single_training_sample_file_name = all_training_sample_files_for_class(i).name;
      single_training_sample_label = getLabelForClassName(class_name,class_name_to_class_label_mapping);
      tmp_data(:,:,:,sample_index) = getProcessedSample(fullfile(data_dir, 'train', class_name, 'images', single_training_sample_file_name));
      tmp_labels(sample_index) = single_training_sample_label;
      sample_index = sample_index + 1;
    end
  end

  runSanityCheckOnData(tmp_data, expeced_total_number_of_samples);
  total_number_of_samples = expeced_total_number_of_samples;

  data = tmp_data;
  labels = tmp_labels;
  set = ones(expeced_total_number_of_samples, 1);

  afprintf(sprintf('[INFO] done!\n'));


% --------------------------------------------------------------------
function [data, labels, set] = getValidationDataAndLabels(data_dir, class_name_to_class_label_mapping)
% --------------------------------------------------------------------
  afprintf(sprintf('[INFO] Retrieving validation data...\n'));

  % expected_number_of_classes = 200;
  % expected_number_of_samples_per_class = 500;
  expeced_total_number_of_samples = 10000;

  fileID = fopen(fullfile(data_dir, 'val', 'val_annotations.txt'), 'r');
  formatSpec = '%s %s %d %d %d %d';
  tmp = textscan(fileID, formatSpec);
  validation_sample_file_names = tmp{1};
  validation_sample_class_names = tmp{2};

  target_sample_size = 32;

  tmp_data = zeros(target_sample_size, target_sample_size, 3, expeced_total_number_of_samples);
  tmp_labels = zeros(expeced_total_number_of_samples, 1);

  for i = 1 : numel(validation_sample_file_names)
    if mod(i, 250) == 0
      afprintf(sprintf('[INFO] Loaded %d samples from total %d\n', floor(i / 250) * 250, expeced_total_number_of_samples),  1);
    end
    single_training_sample_file_name = validation_sample_file_names{i};
    single_training_sample_label = getLabelForClassName(validation_sample_class_names{i}, class_name_to_class_label_mapping);
    tmp_data(:,:,:,i) = getProcessedSample(fullfile(data_dir, 'val', 'images', single_training_sample_file_name));
    tmp_labels(i) = single_training_sample_label;
  end

  runSanityCheckOnData(tmp_data, expeced_total_number_of_samples);
  total_number_of_samples = expeced_total_number_of_samples;

  data = tmp_data;
  labels = tmp_labels;
  set = 3 * ones(expeced_total_number_of_samples, 1);
  afprintf(sprintf('[INFO] done!\n'));


% --------------------------------------------------------------------
function runSanityCheckOnData(data, expeced_total_number_of_samples)
% --------------------------------------------------------------------
  target_sample_size = 32;
  for i = 1 : expeced_total_number_of_samples
    if isequal(data, zeros(target_sample_size, target_sample_size, 3));
      throwException('[ERROR] wtf yo! should not have a sample be all zeros!');
    end
  end


% --------------------------------------------------------------------
function sample = getProcessedSample(file_name)
% --------------------------------------------------------------------
  target_sample_size = 32;
  sample = imread(file_name);
  sample = imresize(sample, [target_sample_size, target_sample_size]);
  if size(sample, 3) == 3
    % do nothing
  elseif size(sample, 3) == 1
    sample = repmat(sample, [1,1,3]);
  else
    throwException('[ERROR] wtf yo! get your dimensions straight already, Karpathy!!');
  end


% --------------------------------------------------------------------
function class_label = getLabelForClassName(file_name, class_name_to_class_label_mapping)
% --------------------------------------------------------------------
  class_label = class_name_to_class_label_mapping.(file_name);


% --------------------------------------------------------------------
function class_name_to_class_label_mapping = getClassNameToClassLabelMapping(data_dir)
% --------------------------------------------------------------------
  % we assume that the validation set contains at least 1 sample from each of
  % the 200 sample classes
  fileID = fopen(fullfile(data_dir, 'val', 'val_annotations.txt'), 'r');
  formatSpec = '%s %s %d %d %d %d';
  tmp = textscan(fileID, formatSpec);
  validation_sample_class_names = tmp{2};

  mapping = {};
  unique_classes_counter = 1;
  for i = 1 : numel(validation_sample_class_names)
    class_name = validation_sample_class_names{i};
    if ~isfield(mapping, class_name)
      mapping.(class_name) = unique_classes_counter;
      unique_classes_counter = unique_classes_counter + 1;
    end
  end

  assert(numel(fieldnames(mapping)) == 200);

  class_name_to_class_label_mapping = mapping;


































