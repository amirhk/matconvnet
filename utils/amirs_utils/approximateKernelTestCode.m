% -------------------------------------------------------------------------
function [accuracy_actual_eigen, accuracy_approx_eigen, accuracy_approx_direct] = approximateKernelTestCode(debug_flag)
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


  % -----------------------------------------------------------------------------
  % Data setup
  % -----------------------------------------------------------------------------

  n = 100;
  d = 25;

  X = [randn(d, n/2), randn(d, n/2) + 0.5];
  X_test = [randn(d, n/2), randn(d, n/2) + 0.5];
  Y = [ones(1,n/2) * 1, ones(1,n/2) * 2];
  Y_test = [ones(1,n/2) * 1, ones(1,n/2) * 2];

  % X = randn(d, n);
  % X_test = randn(d, n);
  % Y = randn(1,n);
  % Y_test = randn(1,n);

  % % [imdb, tmp] = setupExperimentsUsingProjectedImbds('uci-ion', 'whatever', false, false);
  % [imdb, tmp] = setupExperimentsUsingProjectedImbds('uci-spam', 'whatever', false, false);
  % % [imdb, tmp] = setupExperimentsUsingProjectedImbds('mnist-fashion-multi-class-subsampled', 'balanced-10', false, false);
  % vectorized_imdb = getVectorizedImdb(imdb);
  % indices_train = imdb.images.set == 1;
  % indices_test = imdb.images.set == 3;
  % X = vectorized_imdb.images.data(indices_train,:)';
  % Y = vectorized_imdb.images.labels(indices_train);
  % X_test = vectorized_imdb.images.data(indices_test,:)';
  % Y_test = vectorized_imdb.images.labels(indices_test);



  % -----------------------------------------------------------------------------
  % Meta-params
  % -----------------------------------------------------------------------------
  [label_dim, n] = size(Y);
  data_dim = size(X,1);
  if debug_flag; fprintf('Training on #%d data samples with %d dimensions...\n', n, data_dim); end;
  H = eye(n) - 1 / n * (ones(n,n));
  label_rbf_variance = 10e-5; % extremely small variance because we are approximating delta kernel
  projected_dim = 100000;



  % -----------------------------------------------------------------------------
  % Get Actual and Approx Kernels
  % -----------------------------------------------------------------------------
  L_actual = getActualKernel(Y, Y, label_rbf_variance);
  number_of_random_bases = projected_dim;
  [L_approx, psi] = getApproxKernel(Y, label_rbf_variance, number_of_random_bases);



  % -----------------------------------------------------------------------------
  % Compare projections
  % -----------------------------------------------------------------------------

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % U_actual_eigen
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  tmp_actual_eigen = X * H * L_actual * H * X'; % + eye(data_dim);
  [U D V] = svd(tmp_actual_eigen);
  projected_X = U' * X;
  projected_X_test = U' * X_test;
  accuracy_actual_eigen = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % U_approx_eigen
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  tmp_approx_eigen = X * H * L_approx * H * X'; % + eye(data_dim);
  [U D V] = svd(tmp_approx_eigen);
  projected_X = U' * X;
  projected_X_test = U' * X_test;
  accuracy_approx_eigen = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % U_approx_direct
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % U = X * H * psi';
  % % U = X * H * L_actual';
  % projected_X = U' * X;
  % projected_X_test = U' * X_test;

  projected_X = psi * H * getActualKernel(X, X, 10e+4);
  projected_X_test = psi * H * getActualKernel(X, X_test, 10e+4);
  accuracy_approx_direct = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);



% -------------------------------------------------------------------------
function L_actual = getActualKernel(data_1, data_2, rbf_variance)
  % data_* consists of 1 sample per column
% -------------------------------------------------------------------------
  assert(size(data_1, 1) == size(data_2, 1));
  L = zeros(size(data_1, 2), size(data_2, 2));
  for i = 1 : size(data_1, 2)
      for j = 1 : size(data_2, 2)
          u = data_1(:,i)';
          v = data_2(:,j)';
          L(i,j) = exp( - (u - v) * (u - v)' / (2 * rbf_variance ^ 2));
      end
  end
  L_actual = L;


% -------------------------------------------------------------------------
function [L_approx, psi] = getApproxKernel(data, rbf_variance, number_of_random_bases);
  % data consists of 1 sample per column
% -------------------------------------------------------------------------
  [d, n] = size(data);

  gamma = 1 / (2 * rbf_variance ^ 2);
  w = randn(number_of_random_bases, d);
  b = 2 * pi * rand(number_of_random_bases, 1);

  projected_data = cos(gamma * w * data + b * ones(1, n));
  psi = sqrt(2 / number_of_random_bases) * projected_data;

  L = psi' * psi;
  L_approx = L;


% -------------------------------------------------------------------------
function accuracy = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test)
% -------------------------------------------------------------------------
  fprintf('Computing 1-NN in %d-dims \t', size(projected_X',2));
  nearest_neighbor_model = fitcknn(projected_X', Y', 'NumNeighbors', 1);
  test_predictions = predict(nearest_neighbor_model, projected_X_test');
  test_labels = Y_test';
  accuracy = sum(test_predictions == test_labels) / length(test_labels);


% -------------------------------------------------------------------------
function assertPSD(matrix)
% -------------------------------------------------------------------------
  assert(all(eig(matrix) > eps), '[ERROR] expect all eig-values of L_actual to be +ve')

























