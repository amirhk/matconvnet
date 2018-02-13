% --------------------------------------------------------------------
function imdb = constructMnistFashionImdb(opts)
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

  afprintf(sprintf('[INFO] Constructing MNIST Fashion imdb...\n'));

  file_name_train = fullfile(opts.imdb.data_dir, 'fashion-mnist_train.csv');
  file_name_test = fullfile(opts.imdb.data_dir, 'fashion-mnist_test.csv');

  data_and_labels_train = csvread(file_name_train);
  data_and_labels_test = csvread(file_name_test);

  data_train = data_and_labels_train(:, 2:end);
  data_test = data_and_labels_test(:, 2:end);

  labels_train = data_and_labels_train(:, 1);
  labels_test = data_and_labels_test(:, 1);

  vectorized_data = cat(1, data_train, data_test);
  labels = cat(1, labels_train, labels_test);
  set = cat(1, 1 * ones(length(labels_train), 1), 3 * ones(length(labels_test), 1));

  assert(length(labels) == length(set));
  number_of_samples = length(labels);

  % shuffle
  ix = randperm(number_of_samples);
  imdb.images.data = single(vectorized_data(ix,:));
  imdb.images.labels = single(labels(ix)') + 1;
  imdb.images.set = single(set'); % NOT set(ix).... that way you won't have any of your first class samples in the test set!
  imdb.name = 'mnist-fashion';

  imdb = get4DImdb(imdb, 28, 28, 1, number_of_samples);
  afprintf(sprintf('done!\n\n'));
  fh = imdbMultiClassUtils;
  fh.getImdbInfo(imdb, 1);
  % save(sprintf('%s.mat', imdb.name), 'imdb');
