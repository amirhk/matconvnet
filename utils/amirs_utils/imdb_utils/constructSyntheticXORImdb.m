% -------------------------------------------------------------------------
function imdb = constructSyntheticXORImdb(number_of_samples_per_group, sample_dim)
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

  % afprintf(sprintf('[INFO] Constructing synthetic XOR imdb...\n'));

  group_1_samples = mvnrnd(5*[+1,+1], eye(2), number_of_samples_per_group);
  group_2_samples = mvnrnd(5*[+1,-1], eye(2), number_of_samples_per_group);
  group_3_samples = mvnrnd(5*[-1,+1], eye(2), number_of_samples_per_group);
  group_4_samples = mvnrnd(5*[-1,-1], eye(2), number_of_samples_per_group);

  % add random noise to each sample to make a d-dimensional sample (so add d-2 random features)
  group_1_samples = [group_1_samples, randn(number_of_samples_per_group, sample_dim - 2)];
  group_2_samples = [group_2_samples, randn(number_of_samples_per_group, sample_dim - 2)];
  group_3_samples = [group_3_samples, randn(number_of_samples_per_group, sample_dim - 2)];
  group_4_samples = [group_4_samples, randn(number_of_samples_per_group, sample_dim - 2)];

  data_class_1 = cat(1, group_1_samples, group_4_samples);
  data_class_2 = cat(1, group_2_samples, group_3_samples);

  number_of_samples_class_1 = size(data_class_1, 1);
  number_of_samples_class_2 = size(data_class_2, 1);
  total_number_of_samples = number_of_samples_class_1 + number_of_samples_class_2;
  number_of_training_samples = round(0.7 * total_number_of_samples);
  number_of_testing_samples = total_number_of_samples - number_of_training_samples;


  data = cat(1, data_class_1, data_class_2);
  labels = cat(1, 1 * ones(number_of_samples_class_1, 1), 2 * ones(number_of_samples_class_2, 1));
  set = cat(1, 1 * ones(number_of_training_samples, 1), 3 * ones(number_of_testing_samples, 1));

  assert(length(labels) == length(set));

  % shuffle
  ix = randperm(total_number_of_samples);
  imdb.images.data = data(ix,:);
  imdb.images.labels = labels(ix)';
  imdb.images.set = set'; % NOT set(ix).... that way you won't have any of your first class samples in the test set!
  % imdb.name = sprintf('circles-%dD-%d-train-%d-test-%.1f-var', sample_dim, number_of_training_samples, number_of_testing_samples, sample_variance_multiplier);

  % get the data into 4D format to be compatible with code built for all other imdbs.
  imdb = get4DImdb(imdb, sample_dim, 1, 1, total_number_of_samples);
  % afprintf(sprintf('done!\n\n'));
  % fh = imdbMultiClassUtils;
  % fh.getImdbInfo(imdb, 1);
  % save(sprintf('%s.mat', imdb.name), 'imdb');

