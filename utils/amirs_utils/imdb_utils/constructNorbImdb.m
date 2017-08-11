% -------------------------------------------------------------------------
function imdb = constructNorbImdb(opts)
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

  afprintf(sprintf('[INFO] Constructing Norb imdb...\n'));

  data_train_fid = fopen('/Users/a6karimi/dev/data/source/norb/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat', 'r');
  data_test_fid = fopen('/Users/a6karimi/dev/data/source/norb/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat', 'r');
  labels_train_fid = fopen('/Users/a6karimi/dev/data/source/norb/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat', 'r');
  labels_test_fid = fopen('/Users/a6karimi/dev/data/source/norb/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat', 'r');

  number_of_train_samples = 24300;
  number_of_test_samples = 24300;
  data_train = zeros(96, 96, 1, number_of_train_samples);
  data_test = zeros(96, 96, 1, number_of_test_samples);
  labels_train = zeros(number_of_train_samples, 1);
  labels_test = zeros(number_of_test_samples, 1);

  data_train = loadData('train', data_train_fid, data_train, number_of_train_samples);
  data_test = loadData('test', data_test_fid, data_test, number_of_test_samples);
  labels_train = loadLabels('train', labels_train_fid, labels_train, number_of_train_samples);
  labels_test = loadLabels('test', labels_test_fid, labels_test, number_of_test_samples);

  assert(size(data_train, 4) == number_of_train_samples);
  assert(size(data_test, 4) == number_of_test_samples);
  assert(size(labels_train, 1) == number_of_train_samples);
  assert(size(labels_test, 1) == number_of_test_samples);

  afprintf(sprintf('[INFO] Concatinating data and labels...\n'), 1);
  data = cat(4, data_train, data_test); clear data_train; clear data_test;
  labels = cat(1, labels_train, labels_test);
  set = cat(1, 1 * ones(number_of_train_samples, 1), 3 * ones(number_of_test_samples, 1));
  afprintf(sprintf('[INFO] done!\n'), 1);

  assert(length(labels) == length(set));

  % shuffle
  ix = randperm(number_of_train_samples + number_of_test_samples);
  imdb.images.data = single(data(:,:,:,ix));
  imdb.images.labels = single(labels(ix)');
  imdb.images.set = single(set(ix)');
  imdb.name = 'norb';

  afprintf(sprintf('done!\n\n'));
  % fh = imdbMultiClassUtils;
  % fh.getImdbInfo(imdb, 1);
  % save(sprintf('%s.mat', imdb.name), 'imdb');


% -------------------------------------------------------------------------
function tmp_data = loadData(set_string, fid, tmp_data, number_of_samples)
% -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Loading %s data...\n', set_string), 1);
  fread(fid, 4, 'uchar'); % result = [85 76 61 30], byte matrix(in base 16: [55 4C 3D 1E])
  fread(fid, 4, 'uchar'); % result = [4 0 0 0], ndim = 4
  fread(fid, 4, 'uchar'); % result = [236 94 0 0], dim0 = 24300 (=94*256+236)
  fread(fid, 4, 'uchar'); % result = [2 0 0 0], dim1 = 2
  fread(fid, 4, 'uchar'); % result = [96 0 0 0], dim2 = 96
  fread(fid, 4, 'uchar'); % result = [96 0 0 0], dim3 = 96
  for i = 1 : number_of_samples
    tmp_data(:,:,:,i) = transpose(reshape(fread(fid,96*96),96,96));
    % ignore this second image from another camera
    garbage = transpose(reshape(fread(fid,96*96),96,96));
  end
  afprintf(sprintf('[INFO] done!\n'), 1);


% -------------------------------------------------------------------------
function tmp_labels = loadLabels(set_string, fid, tmp_labels, number_of_samples)
% -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Loading %s data...\n', set_string), 1);
  fread(fid, 4, 'uchar'); % result = [84 76 61 30], int matrix (54 4C 3D 1E)
  fread(fid, 4, 'uchar'); % result = [1 0 0 0], ndim = 1
  fread(fid, 4, 'uchar'); % result = [236 94 0 0], dim0 = 24300
  fread(fid, 4, 'uchar'); % result = [1 0 0 0] (ignore this integer)
  fread(fid, 4, 'uchar'); % result = [1 0 0 0] (ignore this integer)
  tmp_labels = fread(fid, number_of_samples, 'int') + 1; % result = [0 1 2 3 4 0 1 2 3 4] (only on little-endian)
  % added +1 to all entries above because they started at 0.
  afprintf(sprintf('[INFO] done!\n'), 1);






















