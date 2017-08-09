% -------------------------------------------------------------------------
function imdb = constructUCIGisetteImdb(opts)
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

  afprintf(sprintf('[INFO] Constructing UCI Gisette imdb...\n'));

  data_train_file = fullfile(opts.imdb.data_dir, 'gisette_train.data');
  data_test_file = fullfile(opts.imdb.data_dir, 'gisette_valid.data');
  labels_train_file = fullfile(opts.imdb.data_dir, 'gisette_train.labels');
  labels_test_file = fullfile(opts.imdb.data_dir, 'gisette_valid.labels');

  afprintf(sprintf('[INFO] Loading data & labels...\n'), 1);
  data_train = load(data_train_file);
  data_test = load(data_test_file);
  labels_train = load(labels_train_file);
  labels_test = load(labels_test_file);
  afprintf(sprintf('[INFO] done!\n'), 1);

  % convert labels from {-1, +1} to {1, 2}
  labels_train = ceil(2.^labels_train);
  labels_test = ceil(2.^labels_test);

  sample_dim = 5000;
  number_of_training_samples = 6000;
  number_of_testing_samples = 1000;
  total_number_of_samples = number_of_training_samples + number_of_testing_samples;
  assert(size(data_train, 1) == number_of_training_samples);
  assert(size(data_train, 2) == sample_dim);
  assert(size(data_test, 1) == number_of_testing_samples);
  assert(size(data_test, 2) == sample_dim);

  data = cat(1, data_train, data_test);
  labels = cat(1, labels_train, labels_test);
  set = cat(1, 1 * ones(number_of_training_samples, 1), 3 * ones(number_of_testing_samples, 1));

  assert(length(labels) == length(set));

  % shuffle
  ix = randperm(total_number_of_samples);
  imdb.images.data = single(data(ix,:));
  imdb.images.labels = single(labels(ix)');
  imdb.images.set = single(set(ix)'); % NOT set(ix).... that way you won't have any of your first class samples in the test set!
  imdb.name = 'uci-gisette';

  imdb = get4DImdb(imdb, sample_dim, 1, 1, total_number_of_samples)
  afprintf(sprintf('done!\n\n'));
  fh = imdbMultiClassUtils;
  fh.getImdbInfo(imdb, 1);
  save(sprintf('%s.mat', imdb.name), 'imdb');


% -------------------------------------------------------------------------
function one_two_labels = convertGoodBadLabelsToOneTwoLabels(good_bad_labels)
% -------------------------------------------------------------------------
  a = reshape(good_bad_labels, 1, []);
  b = replace(a, 'g', '1');
  c = replace(b, 'b', '2');
  d = c';
  e = str2num(d);
  one_two_labels = e;
