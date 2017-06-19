% NOTES:
% 1) Using im2double will bring all pixel values between [-1,+1] and hence need
%    higher LR. Note, that constructing CIFAR imdb in matconvnet does not use
%    im2doube by default, but it was recommended by Javad.
% 2) Subtract the mean of the training data from both the training and test data
% 3) STL-10 does NOT require contrast normalization or whitening
% -------------------------------------------------------------------------
function imdb = constructSyntheticGaussianImdb(samples_per_class, sample_dim, sample_mean, sample_variance)
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

  afprintf(sprintf('[INFO] Constructing synthetic Gaussian imdb...'));
  % data_m = mvnrnd(- sample_mean * ones(sample_dim, 1), sample_variance * eye(sample_dim), samples_per_class);
  % data_p = mvnrnd(+ sample_mean * ones(sample_dim, 1), sample_variance * eye(sample_dim), samples_per_class);
  data_m = mvnrnd(+ 9 * ones(sample_dim, 1), sample_variance * eye(sample_dim), samples_per_class);
  data_p = mvnrnd(+ 11 * ones(sample_dim, 1), sample_variance * eye(sample_dim), samples_per_class);
  labels_m = 1 * ones(1, samples_per_class);
  labels_p = 2 * ones(1, samples_per_class);

  number_of_training_samples = .5 * samples_per_class * 2;
  number_of_testing_samples = .5 * samples_per_class * 2;

  data = cat(1, data_m, data_p);
  labels = cat(2, labels_m, labels_p)';
  set = cat(1, 1 * ones(number_of_training_samples, 1), 3 * ones(number_of_testing_samples, 1));

  assert(length(labels) == length(set));

  % shuffle
  ix = randperm(samples_per_class * 2);
  imdb.images.data = data(ix,:);
  imdb.images.labels = labels(ix);
  imdb.images.set = set; % NOT set(ix).... that way you won't have any of your first class samples in the test set!
  imdb.name = sprintf('gaussian-%dD-%d-train-%d-test-%.1f-var', sample_dim, number_of_training_samples, number_of_testing_samples, sample_variance);

  % get the data into 4D format to be compatible with code built for all other imdbs.
  imdb.images.data = reshape(imdb.images.data', sample_dim, 1, 1, []);
  afprintf(sprintf('done!\n\n'));
  fh = imdbMultiClassUtils;
  fh.getImdbInfo(imdb, 1);
  % save(sprintf('%s.mat', imdb.name), 'imdb');
