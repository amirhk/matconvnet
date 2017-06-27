% -------------------------------------------------------------------------
function imdb = constructSyntheticCirclesImdb(total_number_of_samples, sample_dim, sample_mean, sample_variance_multiplier)
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

  afprintf(sprintf('[INFO] Constructing synthetic circles imdb...\n'));

  covariance_matrix = eye(sample_dim);
  all_samples = mvnrnd(zeros(sample_dim, 1), sample_variance_multiplier * covariance_matrix, total_number_of_samples);

  radius = sample_variance_multiplier;
  data_m = [];
  data_p = [];
  for i = 1 : size(all_samples, 1)
    sample = all_samples(i, :);
    if norm(sample) < radius * sqrt(sample_dim)
      data_m(end+1,:) = sample;
    else
      data_p(end+1,:) = sample;
    end
  end

  number_of_m_samples = size(data_m, 1);
  number_of_p_samples = size(data_p, 1);
  assert(number_of_m_samples + number_of_p_samples == total_number_of_samples);

  number_of_training_samples = .5 * total_number_of_samples;
  number_of_testing_samples = .5 * total_number_of_samples;

  data = cat(1, data_m, data_p);
  labels = cat(1, 1 * ones(number_of_m_samples, 1), 2 * ones(number_of_p_samples, 1));
  set = cat(1, 1 * ones(number_of_training_samples, 1), 3 * ones(number_of_testing_samples, 1));

  assert(length(labels) == length(set));

  % shuffle
  ix = randperm(total_number_of_samples);
  imdb.images.data = data(ix,:);
  imdb.images.labels = labels(ix);
  imdb.images.set = set; % NOT set(ix).... that way you won't have any of your first class samples in the test set!
  imdb.name = sprintf('circles-%dD-%d-train-%d-test-%.1f-var', sample_dim, number_of_training_samples, number_of_testing_samples, sample_variance_multiplier);

  % get the data into 4D format to be compatible with code built for all other imdbs.
  imdb = get4DImdb(imdb, sample_dim, 1, 1, total_number_of_samples);
  afprintf(sprintf('done!\n\n'));
  fh = imdbMultiClassUtils;
  fh.getImdbInfo(imdb, 1);
  save(sprintf('%s.mat', imdb.name), 'imdb');

