% -------------------------------------------------------------------------
function output = approximateKernelTestCode(debug_flag, projected_dim, dataset)
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
  % [imdb, tmp] = setupExperimentsUsingProjectedImbds('uci-sonar', 'whatever', false, false);
  % [imdb, tmp] = setupExperimentsUsingProjectedImbds('uci-spam', 'whatever', false, false); % TODO: create IMDB with 500 random samples... then test on that!??
  % [imdb, tmp] = setupExperimentsUsingProjectedImbds('uci-balance', 'whatever', false, false); imdb = imdb.imdb;
  % [imdb, tmp] = setupExperimentsUsingProjectedImbds('mnist-fashion-multi-class-subsampled', 'balanced-10', false, false);
  % [imdb, tmp] = setupExperimentsUsingProjectedImbds('usps-multi-class-subsampled', 'whatever', false, false);

  % dataset = 'uci-ion';
  % dataset = 'uci-sonar';
  % dataset = 'uci-balance';
  if strcmp(dataset, 'xor-10D-350-train-150-test') || ...
     strcmp(dataset, 'rings-10D-350-train-150-test') || ...
     strcmp(dataset, 'spirals-10D-350-train-150-test')
    tmp_opts.dataset = dataset;
    imdb = loadSavedImdb(tmp_opts, false);
  else
    imdb = constructMultiClassImdbs(dataset, false);
    if strcmp(dataset, 'usps')
      imdb = createImdbWithBalance(dataset, imdb, 25, 25, false, false);
      % imdb = createImdbWithBalance(dataset, imdb, 100, 100, false, false);
    elseif strcmp(dataset, 'uci-spam')
      imdb = createImdbWithBalance(dataset, imdb, 1000, 250, false, false);
    end
  end
  vectorized_imdb = getVectorizedImdb(imdb);
  indices_train = imdb.images.set == 1;
  indices_test = imdb.images.set == 3;
  X = vectorized_imdb.images.data(indices_train,:)';
  Y = vectorized_imdb.images.labels(indices_train);
  X_test = vectorized_imdb.images.data(indices_test,:)';
  Y_test = vectorized_imdb.images.labels(indices_test);

  % normalize between 0-1
  min_x_train = min(X')';
  max_x_train = max(X')';
  X = (X - min_x_train) ./ (max_x_train - min_x_train);
  X_test = (X_test - min_x_train) ./ (max_x_train - min_x_train);
  X(isnan(X)) = 0;
  X_test(isnan(X_test)) = 0;
  % X = normc(X);
  % X_test = normc(X_test);



  % -----------------------------------------------------------------------------
  % Meta-params
  % -----------------------------------------------------------------------------
  output = {};
  [label_dim, n] = size(Y);
  data_dim = size(X,1);
  if debug_flag; fprintf('Training on #%d data samples with %d dimensions...\n', n, data_dim); end;
  H = eye(n) - 1 / n * (ones(n,n));
  projected_dim = projected_dim;
  data_rbf_variance = 10e-1;
  label_rbf_variance = 10e-10; % extremely small variance because we are approximating delta kernel
  number_of_random_bases_for_data = 1000;
  % number_of_random_bases_for_labels = 2;
  number_of_random_bases_for_labels = projected_dim;

  if strcmp(dataset, 'usps')
    data_rbf_variance = 10e+0;
  elseif strcmp(dataset, 'uci-spam')
    data_rbf_variance = 10e+0;
  elseif strcmp(dataset, 'xor-10D-350-train-150-test')
    data_rbf_variance = 10e-1;
  elseif strcmp(dataset, 'rings-10D-350-train-150-test')
    data_rbf_variance = 3*10e-2;
  elseif strcmp(dataset, 'spirals-10D-350-train-150-test')
    data_rbf_variance = 3*10e-2;
  end


  % % -----------------------------------------------------------------------------
  % % Compare projections
  % % -----------------------------------------------------------------------------

  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % % SPCA-eigen
  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % time_start = tic;
  % L_actual = getActualKernel(Y, Y, label_rbf_variance);
  % tmp = X * H * L_actual * H * X';
  % [U D V] = svd(tmp);
  % U = U(:,1:projected_dim);
  % projected_X = U' * X;
  % projected_X_test = U' * X_test;
  % output.accuracy_spca_actual_eigen = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);
  % output.duration_spca_actual_eigen = toc(time_start);

  % projected_X_spca_eigen = projected_X;
  % projected_X_test_spca_eigen = projected_X_test;


  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % % k-SPCA-eigen
  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % time_start = tic;
  % L_actual = getActualKernel(Y, Y, label_rbf_variance);
  % K_train_actual = getActualKernel(X, X, data_rbf_variance);
  % K_test_actual = getActualKernel(X, X_test, data_rbf_variance);
  % tmp = H * L_actual * H * K_train_actual';
  % [U D V] = svd(tmp); % TODO: is it OK to use SVD? or should I use eigendec which is broken??
  % U = U(:,1:projected_dim);
  % projected_X = U' * K_train_actual;
  % projected_X_test = U' * K_test_actual;
  % output.accuracy_kspca_actual_eigen = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);
  % output.duration_kspca_actual_eigen = toc(time_start);

  % projected_X_kspca_eigen = projected_X;
  % projected_X_test_kspca_eigen = projected_X_test;


  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % % SPCA-direct
  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % Y_plus_noise = Y;
  % % Y_plus_noise = Y + randn(1, size(Y, 2)) / 10000;

  % time_start = tic;
  % [L_approx, psi, ~, ~] = getApproxKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance, number_of_random_bases_for_labels, projected_dim);
  % U = X * H * psi';
  % % U = X * psi';
  % projected_X = U' * X;
  % projected_X_test = U' * X_test;
  % output.accuracy_spca_approx_direct = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);
  % output.duration_spca_approx_direct = toc(time_start);

  % projected_X_spca_direct = projected_X;
  % projected_X_test_spca_direct = projected_X_test;


  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % % k-SPCA-eigen
  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % Y_plus_noise = Y;
  % % Y_plus_noise = Y + randn(1, size(Y, 2)) / 10000;

  % time_start = tic;
  % [L_approx, psi, ~, ~] = getApproxKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance, number_of_random_bases_for_labels, projected_dim);
  % [K_train_approx, ~, ~, random_weight_matrix] = getApproxKernel(X, X, data_rbf_variance, number_of_random_bases_for_data, projected_dim);
  % [K_test_approx, ~, ~, ~] = getApproxKernel(X, X_test, data_rbf_variance, number_of_random_bases_for_data, projected_dim, random_weight_matrix);
  % % projected_X = psi * H * K_train_approx * K_train_approx;
  % % projected_X_test = psi * H * K_train_approx * K_test_approx;
  % % projected_X = psi * H * K_train_approx * K_train_approx;
  % % projected_X_test = psi * H * K_train_approx * K_test_approx;
  % projected_X = psi * H * K_train_approx;
  % projected_X_test = psi * H * K_test_approx;
  % % projected_X = psi * K_train_approx;
  % % projected_X_test = psi * K_test_approx;
  % output.accuracy_kspca_approx_direct = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);
  % output.duration_kspca_approx_direct = toc(time_start);

  % projected_X_kspca_direct = projected_X;
  % projected_X_test_kspca_direct = projected_X_test;



  % figure,

  % subplot(2,2,1)
  % plotPerClassTrainAndTestSamples(projected_X_spca_eigen, Y, projected_X_test_spca_eigen, Y_test);
  % title('spca eigen')

  % subplot(2,2,2)
  % plotPerClassTrainAndTestSamples(projected_X_kspca_eigen, Y, projected_X_test_kspca_eigen, Y_test);
  % title('kspca eigen')

  % subplot(2,2,3)
  % plotPerClassTrainAndTestSamples(projected_X_spca_direct, Y, projected_X_test_spca_direct, Y_test);
  % title('spca direct')

  % subplot(2,2,4)
  % plotPerClassTrainAndTestSamples(projected_X_kspca_direct, Y, projected_X_test_kspca_direct, Y_test);
  % title('kspca direct')

  % suptitle(dataset)

  % keyboard











  % -----------------------------------------------------------------------------
  % Compare Hierarchies
  % -----------------------------------------------------------------------------


  % output.test_accuracy_bp_trained_0 = getTestAccuracyFromMLP(X_0, Y, X_test_0, Y_test, [number_of_mlp_nodes]);
  % output.test_accuracy_bp_trained_1 = getTestAccuracyFromMLP(X_0, Y, X_test_0, Y_test, [tmpp, number_of_mlp_nodes]);
  % output.test_accuracy_bp_trained_2 = getTestAccuracyFromMLP(X_0, Y, X_test_0, Y_test, [tmpp, tmpp, number_of_mlp_nodes]);
  % output.test_accuracy_bp_trained_3 = getTestAccuracyFromMLP(X_0, Y, X_test_0, Y_test, [tmpp, tmpp, tmpp, number_of_mlp_nodes]);
  % output.test_accuracy_bp_trained_4 = getTestAccuracyFromMLP(X_0, Y, X_test_0, Y_test, [tmpp, tmpp, tmpp, tmpp, number_of_mlp_nodes]);







  % fh_evaluation = @getTestAccuracyFrom1NN;
  fh_evaluation = @getTestAccuracyFromLinearLeastSquares;


  % D = size(X, 1); % number_of_random_basis;
  D = 2; % number_of_random_basis_per_layer OR number_of_hidden_nodes_per_layer
  s = data_rbf_variance;

  % ----------------------------------------------------------------------------
  nonlin = @cos;
  % ----------------------------------------------------------------------------

  X_0 = X;
  X_test_0 = X_test;

  tmp_matrix = randn(D, size(X_0, 1)) / s;
  X_1 = sqrt(1/D) * nonlin(tmp_matrix * X_0);
  X_test_1 = sqrt(1/D) * nonlin(tmp_matrix * X_test_0);

  tmp_matrix = randn(D, size(X_1, 1)) / s;
  X_2 = sqrt(1/D) * nonlin(tmp_matrix * X_1);
  X_test_2 = sqrt(1/D) * nonlin(tmp_matrix * X_test_1);

  tmp_matrix = randn(D, size(X_2, 1)) / s;
  X_3 = sqrt(1/D) * nonlin(tmp_matrix * X_2);
  X_test_3 = sqrt(1/D) * nonlin(tmp_matrix * X_test_2);

  tmp_matrix = randn(D, size(X_3, 1)) / s;
  X_4 = sqrt(1/D) * nonlin(tmp_matrix * X_3);
  X_test_4 = sqrt(1/D) * nonlin(tmp_matrix * X_test_3);

  figure,

  subplot(2,5,1), plotPerClassTrainAndTestSamples(X_0, Y, X_test_0, Y_test), title('After 0 layers (cos)'),
  subplot(2,5,2), plotPerClassTrainAndTestSamples(X_1, Y, X_test_1, Y_test), title('After 1 layers (cos)'),
  subplot(2,5,3), plotPerClassTrainAndTestSamples(X_2, Y, X_test_2, Y_test), title('After 2 layers (cos)'),
  subplot(2,5,4), plotPerClassTrainAndTestSamples(X_3, Y, X_test_3, Y_test), title('After 3 layers (cos)'),
  subplot(2,5,5), plotPerClassTrainAndTestSamples(X_4, Y, X_test_4, Y_test), title('After 4 layers (cos)'),



  % [K_test_approx, psi_train, psi_test, ~] = getApproxKernel(X_0, X_test_0, s, D, -1);

  % X_1 = psi_train; % * X_0;
  % X_test_1 = psi_test; % * X_test_0;

  % [K_test_approx, psi_train, psi_test, ~] = getApproxKernel(X_1, X_test_1, s, D, -1);

  % X_2 = psi_train; % * X_1;
  % X_test_2 = psi_test; % * X_test_1;

  % [K_test_approx, psi_train, psi_test, ~] = getApproxKernel(X_2, X_test_2, s, D, -1);

  % X_3 = psi_train; % * X_2;
  % X_test_3 = psi_test; % * X_test_2;

  % [K_test_approx, psi_train, psi_test, ~] = getApproxKernel(X_3, X_test_3, s, D, -1);

  % X_4 = psi_train; % * X_3;
  % X_test_4 = psi_test; % * X_test_3;



  % projected_dim = 100;
  % [K_train_approx, psi_train, ~, random_weight_matrix] = getApproxKernel(X_0, X_0, data_rbf_variance, number_of_random_bases_for_data, projected_dim);
  % [K_test_approx, psi_train, psi_test, ~] = getApproxKernel(X_0, X_test_0, data_rbf_variance, number_of_random_bases_for_data, projected_dim, random_weight_matrix);

  % X_1 = psi_train; % * X_0;
  % X_test_1 = psi_test'; % * X_test_0;

  % [K_train_approx, psi_train, ~, random_weight_matrix] = getApproxKernel(X_1, X_1, data_rbf_variance, number_of_random_bases_for_data, projected_dim);
  % [K_test_approx, psi_train, psi_test, ~] = getApproxKernel(X_1, X_test_1, data_rbf_variance, number_of_random_bases_for_data, projected_dim, random_weight_matrix);

  % X_2 = psi_train; % * X_1;
  % X_test_2 = psi_test'; % * X_test_1;

  % [K_train_approx, psi_train, ~, random_weight_matrix] = getApproxKernel(X_2, X_2, data_rbf_variance, number_of_random_bases_for_data, projected_dim);
  % [K_test_approx, psi_train, psi_test, ~] = getApproxKernel(X_2, X_test_2, data_rbf_variance, number_of_random_bases_for_data, projected_dim, random_weight_matrix);

  % X_3 = psi_train; % * X_2;
  % X_test_3 = psi_test'; % * X_test_2;

  % [K_train_approx, psi_train, ~, random_weight_matrix] = getApproxKernel(X_3, X_3, data_rbf_variance, number_of_random_bases_for_data, projected_dim);
  % [K_test_approx, psi_train, psi_test, ~] = getApproxKernel(X_3, X_test_3, data_rbf_variance, number_of_random_bases_for_data, projected_dim, random_weight_matrix);

  % X_4 = psi_train; % * X_3;
  % X_test_4 = psi_test'; % * X_test_3;

  output.test_accuracy_proposed_0 = fh_evaluation(X_0, Y, X_test_0, Y_test);
  output.test_accuracy_proposed_1 = fh_evaluation(X_1, Y, X_test_1, Y_test);
  output.test_accuracy_proposed_2 = fh_evaluation(X_2, Y, X_test_2, Y_test);
  output.test_accuracy_proposed_3 = fh_evaluation(X_3, Y, X_test_3, Y_test);
  output.test_accuracy_proposed_4 = fh_evaluation(X_4, Y, X_test_4, Y_test);



  % ----------------------------------------------------------------------------
  nonlin = @relu;
  % ----------------------------------------------------------------------------

  X_0 = X;
  X_test_0 = X_test;

  tmp_matrix = randn(D, size(X_0, 1)) / s;
  X_1 = sqrt(1/D) * nonlin(tmp_matrix * X_0);
  X_test_1 = sqrt(1/D) * nonlin(tmp_matrix * X_test_0);

  tmp_matrix = randn(D, size(X_1, 1)) / s;
  X_2 = sqrt(1/D) * nonlin(tmp_matrix * X_1);
  X_test_2 = sqrt(1/D) * nonlin(tmp_matrix * X_test_1);

  tmp_matrix = randn(D, size(X_2, 1)) / s;
  X_3 = sqrt(1/D) * nonlin(tmp_matrix * X_2);
  X_test_3 = sqrt(1/D) * nonlin(tmp_matrix * X_test_2);

  tmp_matrix = randn(D, size(X_3, 1)) / s;
  X_4 = sqrt(1/D) * nonlin(tmp_matrix * X_3);
  X_test_4 = sqrt(1/D) * nonlin(tmp_matrix * X_test_3);

  output.test_accuracy_rp_0 = fh_evaluation(X_0, Y, X_test_0, Y_test);
  output.test_accuracy_rp_1 = fh_evaluation(X_1, Y, X_test_1, Y_test);
  output.test_accuracy_rp_2 = fh_evaluation(X_2, Y, X_test_2, Y_test);
  output.test_accuracy_rp_3 = fh_evaluation(X_3, Y, X_test_3, Y_test);
  output.test_accuracy_rp_4 = fh_evaluation(X_4, Y, X_test_4, Y_test);

  subplot(2,5,6), plotPerClassTrainAndTestSamples(X_0, Y, X_test_0, Y_test), title('After 0 layers (relu)'),
  subplot(2,5,7), plotPerClassTrainAndTestSamples(X_1, Y, X_test_1, Y_test), title('After 1 layers (relu)'),
  subplot(2,5,8), plotPerClassTrainAndTestSamples(X_2, Y, X_test_2, Y_test), title('After 2 layers (relu)'),
  subplot(2,5,9), plotPerClassTrainAndTestSamples(X_3, Y, X_test_3, Y_test), title('After 3 layers (relu)'),
  subplot(2,5,10), plotPerClassTrainAndTestSamples(X_4, Y, X_test_4, Y_test), title('After 4 layers (relu)'),

  suptitle(dataset)

  keyboard


















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
function [L_approx, psi_data_1, psi_data_2, random_weight_matrix] = getApproxKernel(data_1, data_2, rbf_variance, number_of_random_bases, projected_dim, w);
  % data consists of 1 sample per column
% -------------------------------------------------------------------------
  assert(size(data_1, 1) == size(data_2, 1));
  d = size(data_1, 1);
  D = number_of_random_bases;
  s = rbf_variance;
  k = projected_dim;

  if nargin == 6
    w = w; % when random weight matrix passed in, use it instead of generating new random matrix: e.g., for constructing K_train & K_test
  else
    w = randn(D, d) / s; % make sure the w is shared between the 2 lines below! do not create w in <each> line below separately.
  end
  random_weight_matrix = w;

  psi_data_1 = sqrt(1 / D) * cos(w * data_1);
  psi_data_2 = sqrt(1 / D) * cos(w * data_2);

  L_approx = psi_data_1' * psi_data_2;

  % if size(L_approx, 1) == size(L_approx, 2)
  %   [a,b] = eig(L_approx);
  %   b = sort(diag(b),'descend');
  %   b(1:10)
  % end

  % [U D V] = svd(L_approx);
  % psi_data_1 = (U(:,1:k) * D(1:k,1:k).^0.5)';
  % psi_data_2 = (D(1:k,1:k).^0.5 * V(:,1:k)')';



% -------------------------------------------------------------------------
function test_accuracy = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test)
% -------------------------------------------------------------------------
  fprintf('Computing 1-NN in %d-dims \t', size(projected_X,1));
  nearest_neighbor_model = fitcknn(projected_X', Y', 'NumNeighbors', 1);
  test_predictions = predict(nearest_neighbor_model, projected_X_test');
  test_labels = Y_test';
  test_accuracy = sum(test_predictions == test_labels) / length(test_labels);



% -------------------------------------------------------------------------
function test_accuracy = getTestAccuracyFromLinearLeastSquares(projected_X, Y, projected_X_test, Y_test)
% -------------------------------------------------------------------------
  fprintf('Computing Lin-LSQ in %d-dims \t', size(projected_X,1));

  projected_X = double(projected_X);
  projected_X_test = double(projected_X_test);
  Y = double(Y);
  Y_test = double(Y_test);

  % convert labels from {1,2} to {-1,1}
  Y = (-1) .^ Y;
  Y_test = (-1) .^ Y_test;

  % Solving for X in << Ax  = b >>
  % I have added an additional column of ones to the data matrix in order to
  % allow for a shift of the separator, thus making it a little more versatile.
  % If you don't do this, you force the separator to pass through the origin,
  % which will more often than not result in worse classification results.
  A_train = [projected_X' ones(size(projected_X,2), 1)];
  b_train = Y';
  x = lsqlin(A_train, b_train);

  A_test = [projected_X_test' ones(size(projected_X_test,2), 1)];
  b_test = Y_test';
  test_predictions = sign(A_test * x);

  assert(length(b_test) == length(test_predictions));
  test_accuracy = sum(b_test == test_predictions) / length(b_test);


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
  x(x<0) = 0;
  % x = x;











  % data_1 = data_1 + randn(1, size(data_1, 2)) / 10;
  % data_2 = data_2 + randn(1, size(data_2, 2)) / 10;

  % % WRONG!!!
  % %       --> gamma should be 1/(rbf_variance ^ 2) without a 1/2!
  % %       --> also, no need for b.
  % % gamma = 1 / (2 * rbf_variance ^ 2);
  % % w = randn(number_of_random_bases, d);
  % % b = 2 * pi * rand(number_of_random_bases, 1);
  % % tmp_1 = gamma * w * data_1 + b * ones(1, size(data_1, 2));
  % % tmp_2 = gamma * w * data_2 + b * ones(1, size(data_2, 2));

  % w = randn(D, d) / s; % w = normrnd(0, 1 / rbf_variance, [number_of_random_bases, d]);
  % projected_data_1 = cos(w * data_1);
  % projected_data_2 = cos(w * data_2);

  % % % TODO: do we need the sin as well???
  % % % projected_data_1 = [cos(w * data_1); sin(w * data_1)];
  % % % projected_data_2 = [cos(w * data_2); sin(w * data_2)];

  % psi_data_1 = sqrt(1 / D) * projected_data_1;
  % psi_data_2 = sqrt(1 / D) * projected_data_2;



% -------------------------------------------------------------------------
function plotPerClassTrainAndTestSamples(X_train, Y_train, X_test, Y_test);
% -------------------------------------------------------------------------

  indices_train_class_1 = Y_train == 1;
  indices_train_class_2 = Y_train == 2;
  indices_test_class_1 = Y_test == 1;
  indices_test_class_2 = Y_test == 2;

  X_train_class_1 = X_train(:, indices_train_class_1);
  X_train_class_2 = X_train(:, indices_train_class_2);
  X_test_class_1 = X_test(:, indices_test_class_1);
  X_test_class_2 = X_test(:, indices_test_class_2);

  hold on,
  scatter(X_train_class_1(1,:), X_train_class_1(2,:),'bs', 'filled');
  scatter(X_train_class_2(1,:), X_train_class_2(2,:),'rd', 'filled');
  scatter(X_test_class_1(1,:), X_test_class_1(2,:),'bs');
  scatter(X_test_class_2(1,:), X_test_class_2(2,:),'rd');
  hold off
















































