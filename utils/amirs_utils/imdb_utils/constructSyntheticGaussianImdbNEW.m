% -------------------------------------------------------------------------
function imdb = constructSyntheticGaussianImdbNEW(samples_per_class, sample_dim, c_separation, eccentricity)
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

  afprintf(sprintf('[INFO] Constructing synthetic Gaussian imdb...\n'));
  % diag - we want the means in every dimension to be +/- 1, and the variance of
  % the two classes (again in each dimension) to be say 5 so the classes have some
  % amount of overlap, but not too much. At the same time, we don't want all dimensions
  % to have the same variance (i.e., we don't want the original imdb to be spherical),
  % because we'd like to assert that after angle-separating the original imdb, then
  % we obtain a spherical imdb. So, we choose means to be +/- 1 and choose variances
  % to be var = diag([25, 9, 1, 25, 9, 1, ...])

  switch eccentricity
    case 1
      variance_basis = [1,1];
    case 1000
      variance_basis = [1000000, 1, 1];
    otherwise
      variance_basis = [25, 9, 1];
  end

  repeated_variance_basis = repmat(variance_basis, 1, ceil(sample_dim / length(variance_basis)));
  % now we must choose the first sample_dim elements of this (say sample_dim = 25, this matrix above will give 27 variance values...)
  covariance_matrix = diag(repeated_variance_basis(1:sample_dim));

  % note we are generating 2 classes of Gaussians, with the same covariance matrix...
  % so now we have 2 covariance matrices, an input c-separation, we can get the means
  tmp = c_separation * sqrt(trace(covariance_matrix));
  % tmp is norm(mu_1 - mu_2) from the c-separation formula
  % ==> tmp^2 = sum_{i=1}^{sample_dim} ( mu_1i - mu_2i )^2 = sample_dim * ( mu_1i - mu_2i )^2
  % we know that mu_2i = - mu_1i
  % ==> tmp^2 = sample_dim * ( 2 * mu_1i )^2
  % ==> tmp^2 = 4 * sample_dim * mu_1i^2
  % ==> mu_1i = tmp / \sqrt(4 * sample_dim) \forall i = [1, sample_dim]
  mu_1i = tmp / sqrt(4 * sample_dim);

  data_m = mvnrnd(- mu_1i * ones(sample_dim, 1), covariance_matrix, samples_per_class);
  data_p = mvnrnd(+ mu_1i * ones(sample_dim, 1), covariance_matrix, samples_per_class);

  number_of_training_samples = .5 * samples_per_class * 2;
  number_of_testing_samples = .5 * samples_per_class * 2;

  data = cat(1, data_m, data_p);
  labels = cat(1, 1 * ones(samples_per_class, 1), 2 * ones(samples_per_class, 1));
  set = cat(1, 1 * ones(number_of_training_samples, 1), 3 * ones(number_of_testing_samples, 1));


  assert(length(labels) == length(set));

  % shuffle
  ix = randperm(samples_per_class * 2);
  imdb.images.data = data(ix,:);
  imdb.images.labels = labels(ix);
  imdb.images.set = set; % NOT set(ix).... that way you won't have any of your first class samples in the test set!
  imdb.name = sprintf('gaussian-%dD-%d-train-%d-test-%.1f-var', sample_dim, number_of_training_samples, number_of_testing_samples);

  % get the data into 4D format to be compatible with code built for all other imdbs.
  imdb = get4DImdb(imdb, sample_dim, 1, 1, samples_per_class * 2);
  afprintf(sprintf('done!\n\n'));
  fh = imdbMultiClassUtils;
  % fh.getImdbInfo(imdb, 1);
  % save(sprintf('%s.mat', imdb.name), 'imdb');













