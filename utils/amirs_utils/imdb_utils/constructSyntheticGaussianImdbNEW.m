% -------------------------------------------------------------------------
function imdb = constructSyntheticGaussianImdbNEW(number_of_classes, samples_per_class, sample_dim, c_separation, eccentricity, shared_covariance)
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

  covariance_per_class = {};
  mean_per_class = {};
  data_per_class = {};
  labels_per_class = {};

  % ASSUMPTION: I'm going to enforce the same trace for the covariance matrix,
  %             even if the matrices themselves are different. This is required because
  %             later we use sqrt(trace(covariance_matrix)) for generating class means
  tmp_cov_matrix = getCovarianceMatrix(eccentricity, sample_dim);
  for i = 1 : number_of_classes
    if shared_covariance
      covariance_per_class{i} = tmp_cov_matrix;
    else
      tmp_cov_diag_vector = diag(tmp_cov_matrix);
      random_permutaion_of_indices = randperm(length(tmp_cov_diag_vector));
      tmp_cov_diag_vector = tmp_cov_diag_vector(random_permutaion_of_indices);
      covariance_per_class{i} = diag(tmp_cov_diag_vector);
    end
    assert(sqrt(trace(covariance_per_class{i})) == sqrt(trace(covariance_per_class{1})));
  end

  % ASSUMPTION: assuming we want tightly-packed gaussians, any pair of 2 guassians must
  %             be c-separated. So we want a n x n matrix with all entries = dist, where
  %               n = number_of_classes;
  %               dist = c_separation * sqrt(trace(covariance_matrix));
  tmp_dist = c_separation * sqrt(trace(covariance_per_class{1}));
  p_dist_matrix = tmp_dist * ( ones(number_of_classes, number_of_classes) - diag(ones(1, number_of_classes)) );
  [Y, ~] = cmdscale(p_dist_matrix);
  % Now Y is n x n (where n = number_of_classes); but sample_dim > n most likely!
  % Remember each row of Y is to be used as the mean of a Gaussian in number_of_classes dimensions
  % but we want our Gaussian to be sample_dim dimensions! Hence, in order to keep the pairwise distances
  % between the means the same (as what we computed in Y), we append the remaining dimensions in all
  % class means w/ 1 (this won't change the norm if all class means have the 1 added to their dimensions!)
  tmp_class_means = cat(2, Y, ones(size(Y, 1), sample_dim - number_of_classes + 1));

  for i = 1 : number_of_classes
    mean_per_class{i} = tmp_class_means(i,:);
  end

  % Now that we have means and covariances for all classes, we can generate data
  for i = 1 : number_of_classes
    data_per_class{i} = mvnrnd(mean_per_class{i}, covariance_per_class{i}, samples_per_class);
    labels_per_class{i} = i * ones(samples_per_class, 1);
  end

  number_of_total_samples = number_of_classes * samples_per_class;
  number_of_training_samples = ceil(0.7 * number_of_total_samples);
  number_of_testing_samples = number_of_total_samples - number_of_training_samples;


  data = cat(1, data_per_class{:});
  labels = cat(1, labels_per_class{:});
  set = cat(1, 1 * ones(number_of_training_samples, 1), 3 * ones(number_of_testing_samples, 1));

  % keyboard

  % return

  % % note we are generating 2 classes of Gaussians, with the same covariance matrix...
  % % so now we have 2 covariance matrices, an input c-separation, we can get the means
  % tmp = c_separation * sqrt(trace(covariance_matrix));
  % % tmp is norm(mu_1 - mu_2) from the c-separation formula
  % % ==> tmp^2 = sum_{i=1}^{sample_dim} ( mu_1i - mu_2i )^2 = sample_dim * ( mu_1i - mu_2i )^2
  % % we know that mu_2i = - mu_1i
  % % ==> tmp^2 = sample_dim * ( 2 * mu_1i )^2
  % % ==> tmp^2 = 4 * sample_dim * mu_1i^2
  % % ==> mu_1i = tmp / \sqrt(4 * sample_dim) \forall i = [1, sample_dim]
  % mu_1i = tmp / sqrt(4 * sample_dim);

  % data_m = mvnrnd(- mu_1i * ones(sample_dim, 1), covariance_matrix, samples_per_class);
  % data_p = mvnrnd(+ mu_1i * ones(sample_dim, 1), covariance_matrix, samples_per_class);

  % number_of_training_samples = .5 * samples_per_class * 2;
  % number_of_testing_samples = .5 * samples_per_class * 2;

  % data = cat(1, data_m, data_p);
  % labels = cat(1, 1 * ones(samples_per_class, 1), 2 * ones(samples_per_class, 1));
  % set = cat(1, 1 * ones(number_of_training_samples, 1), 3 * ones(number_of_testing_samples, 1));


  assert(length(labels) == length(set));

  % shuffle
  ix = randperm(samples_per_class * number_of_classes);
  imdb.images.data = data(ix,:);
  imdb.images.labels = labels(ix);
  imdb.images.set = set; % NOT set(ix).... that way you won't have any of your first class samples in the test set!
  imdb.name = sprintf('%d-gaussian-%dD-train-%d-test-%d', number_of_classes, sample_dim, number_of_training_samples, number_of_testing_samples);

  % get the data into 4D format to be compatible with code built for all other imdbs.
  imdb = get4DImdb(imdb, sample_dim, 1, 1, samples_per_class * number_of_classes);
  afprintf(sprintf('done!\n\n'));
  fh = imdbMultiClassUtils;
  % fh.getImdbInfo(imdb, 1);
  % save(sprintf('%s.mat', imdb.name), 'imdb');

% -------------------------------------------------------------------------
function covariance_matrix = getCovarianceMatrix(eccentricity, sample_dim);
% -------------------------------------------------------------------------
  % variance_basis = [1000000, 1];
  % for e.g., eccentricity 1000 in d-dimensions generates a diagonal matrix
  % whose diagonal entries are the power-2 of numbers uniformly sampled from
  % [1, 1000], and we make sure to include 1 and 1000 (inspired by Dasgupta)
  covariance_matrix = ceil(rand(1, sample_dim) * eccentricity);
  % now take 2 random indices (not the same one) and set them to 1 and 1000
  % for corner case eccentricities
  tmp = datasample(1:sample_dim, 2, 'Replace', false);
  covariance_matrix(tmp(1)) = 1;
  covariance_matrix(tmp(2)) = eccentricity;
  % now we must take the power of 2 of each elem, bc we want eigne-values
  covariance_matrix = diag(covariance_matrix .* covariance_matrix);


  % switch eccentricity
  %   case 1
  %     covariance_matrix = diag(ones(1, sample_dim));
  %   case 1000
  %     % variance_basis = [1000000, 1];
  %     % for e.g., eccentricity 1000 in d-dimensions generates a diagonal matrix
  %     % whose diagonal entries are the power-2 of numbers uniformly sampled from
  %     % [1, 1000], and we make sure to include 1 and 1000 (inspired by Dasgupta)
  %     covariance_matrix = ceil(rand(1, sample_dim) * eccentricity);
  %     % now take 2 random indices (not the same one) and set them to 1 and 1000
  %     % for corner case eccentricities
  %     tmp = datasample(1:sample_dim, 2, 'Replace', false);
  %     covariance_matrix(tmp(1)) = 1;
  %     covariance_matrix(tmp(2)) = eccentricity;
  %     % now we must take the power of 2 of each elem, bc we want eigne-values
  %     covariance_matrix = diag(covariance_matrix .* covariance_matrix);
  %   otherwise
  %     % diag - we want the means in every dimension to be +/- 1, and the variance of
  %     % the two classes (again in each dimension) to be say 5 so the classes have some
  %     % amount of overlap, but not too much. At the same time, we don't want all dimensions
  %     % to have the same variance (i.e., we don't want the original imdb to be spherical),
  %     % because we'd like to assert that after angle-separating the original imdb, then
  %     % we obtain a spherical imdb. So, we choose means to be +/- 1 and choose variances
  %     % to be var = diag([25, 9, 1, 25, 9, 1, ...])
  %     variance_basis = [25, 9, 1];
  %     repeated_variance_basis = repmat(variance_basis, 1, ceil(sample_dim / length(variance_basis)));
  %     % now we must choose the first sample_dim elements of this (say sample_dim = 25, this matrix above will give 27 variance values...)
  %     covariance_matrix = diag(repeated_variance_basis(1:sample_dim));
  % end













