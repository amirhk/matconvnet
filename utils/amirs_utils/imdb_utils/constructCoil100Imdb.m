% -------------------------------------------------------------------------
function imdb = constructCOIL100Imdb(opts)
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

  afprintf(sprintf('[INFO] Constructing STL-10 imdb...\n'));

  num_of_objects = 100; % numOfClasses
  num_of_angles = 72;
  num_of_images = num_of_objects * num_of_angles;
  image_size_y = 128;
  image_size_x = 128;
  image_size_z = 3;

  train_file = fullfile(opts.imdb.data_dir, 'train.mat');
  test_file = fullfile(opts.imdb.data_dir, 'test.mat');

  train_images_indices = 1:2:num_of_images; % every other angle of the image
  test_images_indices = 2:2:num_of_images; % every other angle of the image

  labels = zeros(1, num_of_images);
  for i = 1:num_of_objects
    labels((i-1) * num_of_angles + 1:i * num_of_angles) = i;
  end

  if ~exist(train_file) || ~exist(test_file)
    afprintf(sprintf('\t[INFO] no `images.mat` file found; generating a new one from image files...\n'));
    images = zeros(num_of_images, image_size_y * image_size_x * image_size_z); % [100 * 72, 49152]
    for object_num = 1:1:100
      for angle_num = 0:5:355
        image_name = sprintf('obj%d__%d.png', object_num, angle_num);
        image = imread(fullfile(opts.imdb.data_dir, image_name));
        image = reshape(image, 1, []); % [1, 49152]
        images((object_num - 1) * num_of_angles + (angle_num + 5) / 5, :) = image;
      end
      if ~mod(object_num, 5)
        afprintf(sprintf('\t\t[INFO] finished processing %d files.\n', object_num));
      end
    end
    fprintf('\tdone\n');

    meta_train.data = images(train_images_indices,:);
    meta_train.labels = labels(train_images_indices);
    meta_test.data = images(test_images_indices,:);
    meta_test.labels = labels(test_images_indices);

    afprintf(sprintf('\t[INFO] Saving train meta data (large file ~25MB)...'));
    save(fullfile(opts.imdb.data_dir, 'train.mat'), 'meta_train');
    fprintf('done\n');
    afprintf(sprintf('\t[INFO] Saving test meta data (large file ~25MB)...'));
    save(fullfile(opts.imdb.data_dir, 'test.mat'), 'meta_test');
    fprintf('done\n');
  else
    afprintf(sprintf('\t[INFO] Found pre-existing train and test meta files. Loading... '));
    meta_train = load(train_file);
    meta_train = meta_train.meta_train;
    meta_test = load(test_file);
    meta_test = meta_test.meta_test;
    fprintf('done.\n');
  end

  % disp(meta_train);
  data_train = meta_train.data;
  labels_train = meta_train.labels;
  afprintf(sprintf('\t[INFO] Processing and resizing `train` images... '));
  data_train = imresize(reshape(data_train', image_size_y,image_size_x,image_size_z,[]), [32,32]);
  fprintf('done.\n');
  labels_train = single(labels_train);
  set_train = 1 * ones(1, num_of_images / 2);

  data_test = meta_test.data;
  labels_test = meta_test.labels;
  afprintf(sprintf('\t[INFO] Processing and resizing `test` images... '));
  data_test = imresize(reshape(data_test', image_size_y,image_size_x,image_size_z,[]), [32,32]);
  fprintf('done.\n');
  labels_test = single(labels_test);
  set_test = 3 * ones(1, num_of_images / 2);

  data = single(cat(4, data_train, data_test));
  % data = cat(4, data_train, data_test);
  labels = single(cat(2, labels_train, labels_test));
  set = cat(2, set_train, set_test);

  % remove mean in any case
  dataMean = mean(data(:,:,:,set == 1), 4);
  data = bsxfun(@minus, data, dataMean);

  imdb.images.data = data;
  imdb.images.labels = labels;
  imdb.images.set = set;
  imdb.meta.sets = {'train', 'val', 'test'};
  % imdb.meta.classes = ... i just made 100 classes myself, 1 per object (that
  % has 72 angles)... so we don't really have names for these classes. Note, we
  % can most definitely come up with fewer classes, say 'cup', 'car', 'fruit'...
  fprintf('done!\n\n');
