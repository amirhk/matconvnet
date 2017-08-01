% --------------------------------------------------------------------
function imdb = constructPathologyImdb(input_opts)
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

  afprintf(sprintf('[INFO] Constructing Pathology imdb...\n'));

  % dataset = 'pathology';
  % input_opts.imdb.data_dir = fullfile('/Volumes/Amir', 'data', 'source', dataset);

  [data_train, labels_train, set_train] = getDataAndLabels(input_opts.imdb.data_dir, 'BreaKHis_v1', 'fold1', 'train', '40X');
  [data_test, labels_test, set_test] = getDataAndLabels(input_opts.imdb.data_dir, 'BreaKHis_v1', 'fold1', 'test', '40X');

  afprintf(sprintf('[INFO] Concatinating training data and testing data...\n'));
  data = single(cat(4, data_train, data_test));
  labels = single(cat(1, labels_train, labels_test));
  set = single(cat(1, set_train, set_test));
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
function [data, labels, set] = getDataAndLabels(data_dir, sub_folder_1, sub_folder_2, sub_folder_3, sub_folder_4)
% --------------------------------------------------------------------
  afprintf(sprintf('[INFO] Retrieving training data...\n'));

  % notation: 1 image contains >= 1 patches (patches, crops, augments, whatever... )
  all_image_file_names = dir(fullfile(data_dir, sub_folder_1, sub_folder_2, sub_folder_3, sub_folder_4, 'SOB_*.png'));
  total_number_of_images = numel(all_image_file_names);
  expected_image_size = [460,700,3];
  expected_number_of_patches_per_image = 126;
  target_patch_size = [64 64 3];
  expected_total_number_of_images = total_number_of_images * expected_number_of_patches_per_image;

  tmp_data = zeros(cat(2, target_patch_size, expected_total_number_of_images));
  tmp_labels = zeros(expected_total_number_of_images, 1);

  patch_counter = 1;
  for i = 1150 : total_number_of_images
    afprintf(sprintf('[INFO] Loading image # %d / %d\t\t', i, total_number_of_images));
    single_image_file_name = all_image_file_names(i).name;
    single_image_file_name_with_path = fullfile(all_image_file_names(i).folder, all_image_file_names(i).name);
    single_training_image_label = getLabelForClassName(single_image_file_name);
    tmp_image = getProcessedImage(single_image_file_name_with_path, expected_image_size);
    patches = getPatchesFromImage(tmp_image);
    fprintf('Extracting patches\t\t');
    assert(numel(patches) == expected_number_of_patches_per_image, 'incorrect number of patches for image');
    for j = 1 : numel(patches)
      tmp_data(:,:,:, patch_counter) = patches{j};
      tmp_labels(patch_counter) = single_training_image_label; % all patches for the same image share that image's label
    end
    fprintf('Done!\n');
  end

  keyboard


% --------------------------------------------------------------------
function sample = getProcessedImage(file_name, expected_image_size)
% --------------------------------------------------------------------
  sample = im2double(imread(file_name));
  if ~isequal(size(sample), expected_image_size)
    keyboard
  end
  % assert(isequal(size(sample), expected_image_size), 'file size incorrect');


% --------------------------------------------------------------------
function patches = getPatchesFromImage(sample)
% --------------------------------------------------------------------
  window_size_y = 64;
  window_size_x = 64;
  stride_y = 48;
  stride_x = 48;
  sample_size_1 = size(sample, 1);

  sample_size_2 = size(sample, 2);
  sample_size_3 = size(sample, 3);

  patches = {};
  y = 1;
  x = 1;

  while y + window_size_y <= sample_size_1
    y_start = y;
    y_end = y + window_size_y - 1;
    x = 1;
    while x + window_size_x <= sample_size_2
      x_start = x;
      x_end = x + window_size_x - 1;
      patches{end+1} = sample(y_start:y_end, x_start:x_end, :);

      % increment
      x = x + stride_x;
    end

    % increment
    y = y + stride_y;
  end


% --------------------------------------------------------------------
function class_label = getLabelForClassName(file_name)
% --------------------------------------------------------------------
  if strfind(file_name, '_A-')
    class_label = 1;
  elseif strfind(file_name, '_F-')
    class_label = 2;
  elseif strfind(file_name, '_PT-')
    class_label = 3;
  elseif strfind(file_name, '_TA-')
    class_label = 4;
  elseif strfind(file_name, '_DC-')
    class_label = 5;
  elseif strfind(file_name, '_LC-')
    class_label = 6;
  elseif strfind(file_name, '_MC-')
    class_label = 7;
  elseif strfind(file_name, '_PC-')
    class_label = 8;
  else
    throwException('[ERROR] cannot assign class label correctly');
  end


































