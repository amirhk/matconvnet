% -------------------------------------------------------------------------
function output = approximateKernelTestCode(debug_flag)
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

  % n = 100;
  % d = 25;

  % X = [randn(d, n/2), randn(d, n/2) + 0.5];
  % X_test = [randn(d, n/2), randn(d, n/2) + 0.5];
  % Y = [ones(1,n/2) * 1, ones(1,n/2) * 2];
  % Y_test = [ones(1,n/2) * 1, ones(1,n/2) * 2];

  % [imdb, tmp] = setupExperimentsUsingProjectedImbds('uci-ion', 'whatever', false, false);
  [imdb, tmp] = setupExperimentsUsingProjectedImbds('uci-sonar', 'whatever', false, false);
  % [imdb, tmp] = setupExperimentsUsingProjectedImbds('uci-spam', 'whatever', false, false); % TODO: create IMDB with 500 random samples... then test on that!??
  % [imdb, tmp] = setupExperimentsUsingProjectedImbds('uci-balance', 'whatever', false, false); imdb = imdb.imdb;
  % [imdb, tmp] = setupExperimentsUsingProjectedImbds('mnist-fashion-multi-class-subsampled', 'balanced-10', false, false);
  vectorized_imdb = getVectorizedImdb(imdb);
  indices_train = imdb.images.set == 1;
  indices_test = imdb.images.set == 3;
  X = vectorized_imdb.images.data(indices_train,:)';
  Y = vectorized_imdb.images.labels(indices_train);
  X_test = vectorized_imdb.images.data(indices_test,:)';
  Y_test = vectorized_imdb.images.labels(indices_test);



  % -----------------------------------------------------------------------------
  % Meta-params
  % -----------------------------------------------------------------------------
  output = {};
  [label_dim, n] = size(Y);
  data_dim = size(X,1);
  if debug_flag; fprintf('Training on #%d data samples with %d dimensions...\n', n, data_dim); end;
  H = eye(n) - 1 / n * (ones(n,n));
  projected_dim = 1;
  data_rbf_variance = 10e+1;
  label_rbf_variance = 10e-10; % extremely small variance because we are approximating delta kernel
  number_of_random_bases_for_data = 1000;
  number_of_random_bases_for_labels = projected_dim;



  % -----------------------------------------------------------------------------
  % Get Actual and Approx Kernels
  % -----------------------------------------------------------------------------
  L_actual = getActualKernel(Y, Y, label_rbf_variance);
  K_train_actual = getActualKernel(X, X, data_rbf_variance);
  K_test_actual = getActualKernel(X, X_test, data_rbf_variance);

  [L_approx, psi, ignore] = getApproxKernel(Y, Y, label_rbf_variance, number_of_random_bases_for_labels);
  [K_train_approx, ignore, ignore] = getApproxKernel(X, X, data_rbf_variance, number_of_random_bases_for_data);
  [K_test_approx, ignore, ignore] = getApproxKernel(X, X_test, data_rbf_variance, number_of_random_bases_for_data);

  % keyboard

  % % -----------------------------------------------------------------------------
  % % Compare Hierarchies
  % % -----------------------------------------------------------------------------

  % % number_of_mlp_nodes = 50;
  % tmpp = 2 * number_of_random_bases_for_labels;

  % [L_approx, psi, ignore] = getApproxKernel(Y, Y, label_rbf_variance, number_of_random_bases_for_labels);

  % X_0 = X;
  % X_test_0 = X_test;

  % X_1 = relu(psi * H * getActualKernel(X_0, X_0, data_rbf_variance));
  % X_test_1 = relu(psi * H * getActualKernel(X_0, X_test_0, data_rbf_variance));

  % X_2 = relu(psi * H * getActualKernel(X_1, X_1, data_rbf_variance));
  % X_test_2 = relu(psi * H * getActualKernel(X_1, X_test_1, data_rbf_variance));

  % X_3 = relu(psi * H * getActualKernel(X_2, X_2, data_rbf_variance));
  % X_test_3 = relu(psi * H * getActualKernel(X_2, X_test_2, data_rbf_variance));

  % X_4 = relu(psi * H * getActualKernel(X_3, X_3, data_rbf_variance));
  % X_test_4 = relu(psi * H * getActualKernel(X_3, X_test_3, data_rbf_variance));


  % % output.test_accuracy_proposed_0 = getTestAccuracyFromMLP(X_0, Y, X_test_0, Y_test, [number_of_mlp_nodes]);
  % % output.test_accuracy_proposed_1 = getTestAccuracyFromMLP(X_1, Y, X_test_1, Y_test, [number_of_mlp_nodes]);
  % % output.test_accuracy_proposed_2 = getTestAccuracyFromMLP(X_2, Y, X_test_2, Y_test, [number_of_mlp_nodes]);
  % % output.test_accuracy_proposed_3 = getTestAccuracyFromMLP(X_3, Y, X_test_3, Y_test, [number_of_mlp_nodes]);
  % % output.test_accuracy_proposed_4 = getTestAccuracyFromMLP(X_4, Y, X_test_4, Y_test, [number_of_mlp_nodes]);
  % output.test_accuracy_proposed_0 = getTestAccuracyFrom1NN(X_0, Y, X_test_0, Y_test);
  % output.test_accuracy_proposed_1 = getTestAccuracyFrom1NN(X_1, Y, X_test_1, Y_test);
  % output.test_accuracy_proposed_2 = getTestAccuracyFrom1NN(X_2, Y, X_test_2, Y_test);
  % output.test_accuracy_proposed_3 = getTestAccuracyFrom1NN(X_3, Y, X_test_3, Y_test);
  % output.test_accuracy_proposed_4 = getTestAccuracyFrom1NN(X_4, Y, X_test_4, Y_test);

  % X_0 = X;
  % X_test_0 = X_test;

  % tmp_matrix = randn(tmpp, size(X_0, 1));
  % X_1 = relu(tmp_matrix * X_0);
  % X_test_1 = relu(tmp_matrix * X_test_0);

  % tmp_matrix = randn(tmpp, size(X_1, 1));
  % X_2 = relu(tmp_matrix * X_1);
  % X_test_2 = relu(tmp_matrix * X_test_1);

  % tmp_matrix = randn(tmpp, size(X_2, 1));
  % X_3 = relu(tmp_matrix * X_2);
  % X_test_3 = relu(tmp_matrix * X_test_2);

  % tmp_matrix = randn(tmpp, size(X_3, 1));
  % X_4 = relu(tmp_matrix * X_3);
  % X_test_4 = relu(tmp_matrix * X_test_3);


  % % output.test_accuracy_rp_0 = getTestAccuracyFromMLP(X_0, Y, X_test_0, Y_test, [number_of_mlp_nodes]);
  % % output.test_accuracy_rp_1 = getTestAccuracyFromMLP(X_1, Y, X_test_1, Y_test, [number_of_mlp_nodes]);
  % % output.test_accuracy_rp_2 = getTestAccuracyFromMLP(X_2, Y, X_test_2, Y_test, [number_of_mlp_nodes]);
  % % output.test_accuracy_rp_3 = getTestAccuracyFromMLP(X_3, Y, X_test_3, Y_test, [number_of_mlp_nodes]);
  % % output.test_accuracy_rp_4 = getTestAccuracyFromMLP(X_4, Y, X_test_4, Y_test, [number_of_mlp_nodes]);
  % output.test_accuracy_rp_0 = getTestAccuracyFrom1NN(X_0, Y, X_test_0, Y_test);
  % output.test_accuracy_rp_1 = getTestAccuracyFrom1NN(X_1, Y, X_test_1, Y_test);
  % output.test_accuracy_rp_2 = getTestAccuracyFrom1NN(X_2, Y, X_test_2, Y_test);
  % output.test_accuracy_rp_3 = getTestAccuracyFrom1NN(X_3, Y, X_test_3, Y_test);
  % output.test_accuracy_rp_4 = getTestAccuracyFrom1NN(X_4, Y, X_test_4, Y_test);

  % % output.test_accuracy_bp_trained_0 = getTestAccuracyFromMLP(X_0, Y, X_test_0, Y_test, [number_of_mlp_nodes]);
  % % output.test_accuracy_bp_trained_1 = getTestAccuracyFromMLP(X_0, Y, X_test_0, Y_test, [tmpp, number_of_mlp_nodes]);
  % % output.test_accuracy_bp_trained_2 = getTestAccuracyFromMLP(X_0, Y, X_test_0, Y_test, [tmpp, tmpp, number_of_mlp_nodes]);
  % % output.test_accuracy_bp_trained_3 = getTestAccuracyFromMLP(X_0, Y, X_test_0, Y_test, [tmpp, tmpp, tmpp, number_of_mlp_nodes]);
  % % output.test_accuracy_bp_trained_4 = getTestAccuracyFromMLP(X_0, Y, X_test_0, Y_test, [tmpp, tmpp, tmpp, tmpp, number_of_mlp_nodes]);



  % -----------------------------------------------------------------------------
  % Compare projections
  % -----------------------------------------------------------------------------

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % SPCA U_actual_eigen
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  tmp = X * H * L_actual * H * X';
  [U D V] = svd(tmp);
  U = U(:,1:projected_dim);
  projected_X = U' * X;
  projected_X_test = U' * X_test;
  output.accuracy_spca_actual_eigen = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % SPCA U_approx_eigen
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  tmp = X * H * L_approx * H * X';
  [U D V] = svd(tmp);
  U = U(:,1:projected_dim);
  projected_X = U' * X;
  projected_X_test = U' * X_test;
  output.accuracy_spca_approx_eigen = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % k-SPCA U_actual_eigen
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  tmp = H * L_actual * H * K_train_actual';
  [U D V] = svd(tmp); % TODO: is it OK to use SVD? or should I use eigendec which is broken??
  U = U(:,1:projected_dim);
  projected_X = U' * K_train_actual;
  projected_X_test = U' * K_test_actual;
  output.accuracy_kspca_actual_eigen = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % k-SPCA U_approx_eigen
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  tmp = H * L_approx * H * K_train_approx';
  [U D V] = svd(tmp); % TODO: is it OK to use SVD? or should I use eigendec which is broken??
  U = U(:,1:projected_dim);
  projected_X = U' * K_train_approx;
  projected_X_test = U' * K_test_approx;
  output.accuracy_kspca_approx_eigen = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % U_approx_direct
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  U = X * H * psi';
  % % U = X * H * L_actual';
  projected_X = U' * X;
  projected_X_test = U' * X_test;
  output.accuracy_spca_approx_direct = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);

  projected_X = psi * H * K_train_approx;
  projected_X_test = psi * H * K_test_approx;
  output.accuracy_kspca_approx_direct = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);

  projected_X = psi * H * K_train_actual;
  projected_X_test = psi * H * K_test_actual;
  output.accuracy_kspca_actual_direct = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);



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
function [L_approx, psi_data_1, psi_data_2] = getApproxKernel(data_1, data_2, rbf_variance, number_of_random_bases);
  % data consists of 1 sample per column
% -------------------------------------------------------------------------
  assert(size(data_1, 1) == size(data_2, 1));
  d = size(data_1, 1);

  gamma = 1 / (2 * rbf_variance ^ 2);
  w = randn(number_of_random_bases, d);
  b = 2 * pi * rand(number_of_random_bases, 1);

  tmp_1 = gamma * w * data_1 + b * ones(1, size(data_1, 2));
  tmp_2 = gamma * w * data_2 + b * ones(1, size(data_2, 2));

  % projected_data_1 = cos(tmp_1);
  % projected_data_2 = cos(tmp_2);

  % psi_data_1 = sqrt(2 / number_of_random_bases) * projected_data_1;
  % psi_data_2 = sqrt(2 / number_of_random_bases) * projected_data_2;

  projected_data_1 = [cos(tmp_1)];
  projected_data_2 = [cos(tmp_2)];

  % projected_data_1 = [cos(tmp_1); sin(tmp_1)];
  % projected_data_2 = [cos(tmp_2); sin(tmp_2)];

  psi_data_1 = sqrt(1 / number_of_random_bases) * projected_data_1;
  psi_data_2 = sqrt(1 / number_of_random_bases) * projected_data_2;

  L_approx = psi_data_1' * psi_data_2;
  % L_approx = psi_data_1 * psi_data_2';


  % psi_data_1 = exp(i * w * data_1);
  % psi_data_2 = exp(i * w * data_2);
  % L_approx = psi_data_1' * psi_data_2;
  % keyboard



% -------------------------------------------------------------------------
function test_accuracy = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test)
% -------------------------------------------------------------------------
  fprintf('Computing 1-NN in %d-dims \t', size(projected_X',2));
  nearest_neighbor_model = fitcknn(projected_X', Y', 'NumNeighbors', 1);
  test_predictions = predict(nearest_neighbor_model, projected_X_test');
  test_labels = Y_test';
  test_accuracy = sum(test_predictions == test_labels) / length(test_labels);



% -------------------------------------------------------------------------
function test_accuracy = getTestAccuracyFromMLP(projected_X, Y, projected_X_test, Y_test, layers)
% -------------------------------------------------------------------------
  Y_one_hot = full(ind2vec(double(Y)));
  net = patternnet(layers);
  % change activation functions from default 'tansig' to 'relu/poslin'
  for i = 1:numel(net.layers)-1
    net.layers{i}.transferFcn = 'poslin';
  end
  net = train(net, projected_X, Y_one_hot, 'useGPU', 'no', 'showResources', 'no');

  all_data = [projected_X, projected_X_test];
  top_predictions_matrix_all_classes_softmax = net(all_data);

  [~, tmp] = sort(top_predictions_matrix_all_classes_softmax, 1, 'descend');
  top_predictions = tmp(1,:);

  train_predictions = top_predictions(1:size(projected_X,2));
  test_predictions = top_predictions(size(projected_X,2)+1:end);

  test_labels = Y_test;

  test_accuracy = sum(test_predictions == test_labels) / length(test_labels);



% -------------------------------------------------------------------------
function x = relu(x)
% -------------------------------------------------------------------------
  % x(x<0) = 0;
  x = x;









