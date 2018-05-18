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

  fprintf('\t[INFO] Loading data... ');
  % should_load_saved_imdb = false;
  should_load_saved_imdb = true;
  if should_load_saved_imdb
    tmp_opts.dataset = dataset;
    if ~isSyntheticImdb(dataset) % NOT 'xor', 'rings', 'spirals'
      tmp_opts.posneg_balance = 'balanced-250';
    end
    imdb = loadSavedImdb(tmp_opts, false);
  else
    imdb = constructMultiClassImdbs(dataset, false);
    % in case the dataset is large, shrink it.
    if strcmp(dataset, 'usps')
      imdb = createImdbWithBalance(dataset, imdb, 100, 25, false, false);
    elseif strcmp(dataset, 'mnist-784')
      imdb = createImdbWithBalance(dataset, imdb, 100, 25, false, false);
      % imdb = createImdbWithBalance(dataset, imdb, 250, 100, false, false);
      % imdb = createImdbWithBalance(dataset, imdb, 2500, 250, false, false);
      % subset_indices = ...
      %   imdb.images.labels == 1 | ...
      %   imdb.images.labels == 3 | ...
      %   imdb.images.labels == 4 | ...
      %   imdb.images.labels == 7 | ...
      %   imdb.images.labels == 8;
      % imdb.images.data = imdb.images.data(:,:,:,subset_indices);
      % imdb.images.labels = imdb.images.labels(subset_indices);
      % imdb.images.set = imdb.images.set(subset_indices);
    elseif strcmp(dataset, 'uci-spam')
      imdb = createImdbWithBalance(dataset, imdb, 1000, 250, false, false);
    end
  end
  fprintf('done.\t');

  if strcmp(dataset, 'imagenet-tiny')
    % imdb = createImdbWithBalance(dataset, imdb, 10, 5, false, false);
    subset_indices = imdb.images.labels <= 10;
    imdb.images.data = imdb.images.data(:,:,:,subset_indices);
    imdb.images.labels = imdb.images.labels(subset_indices);
    imdb.images.set = imdb.images.set(subset_indices);
  end

  vectorized_imdb = getVectorizedImdb(imdb);
  all_data = vectorized_imdb.images.data';
  all_labels = vectorized_imdb.images.labels;

  should_normalize = 1;
  if should_normalize % from Elnaz's code
    X = all_data;
    [d,n] = size(X);
    X = (X - repmat(min(X')', 1, n)) ./ (repmat(max(X')', 1, n) - repmat(min(X')', 1, n));
    nan_ind = find(isnan(X)==1);
    X(nan_ind) = 0;
    all_data = X;;
  end

  indices_train = imdb.images.set == 1;
  indices_test = imdb.images.set == 3;
  X = all_data(:,indices_train);
  Y = all_labels(indices_train);
  X_test = all_data(:,indices_test);
  Y_test = all_labels(indices_test);

  Y = Y(:)'; % 1 by n_train
  Y_test = Y_test(:)'; % 1 by n_test


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
  label_rbf_variance = 1e-10; % extremely small variance because we are approximating delta kernel
  number_of_random_bases_for_data = 1000;
  number_of_random_bases_for_labels = projected_dim;

  if strcmp(dataset, 'usps')
    data_rbf_variance = 1.5e0;
  elseif strcmp(dataset, 'mnist-784')
    data_rbf_variance = 10e+0;
  elseif strcmp(dataset, 'imagenet-tiny')
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

  Y_plus_noise = Y + randn(1, size(Y, 2)) / 10e+5; % for UCI-Ion
  % Y_plus_noise = Y + randn(1, size(Y, 2)) / 10e+6;
  % Y_plus_noise = Y + randn(1, size(Y, 2)) / 10e+7;
  % Y_plus_noise = Y + randn(1, size(Y, 2)) / 10e+8;
  % Y_plus_noise = Y + randn(1, size(Y, 2)) / 10e+6; % for MNIST
  % Y_plus_noise = Y + randn(1, size(Y, 2)) / 10e+10; % for RINGS
  % Y_plus_noise = Y + randn(1, size(Y, 2)) / 10e+8; % for XOR
  % Y_plus_noise = Y + randn(1, size(Y, 2)) / 10e+9; % for XOR
  % Y_plus_noise = Y + randn(1, size(Y, 2)) / 10e+10; % for XOR
  % Y_plus_noise = Y;


  % -----------------------------------------------------------------------------
  % Compare projections
  % -----------------------------------------------------------------------------

  fh_evaluation = @getTestAccuracyFrom1NN;
  % fh_evaluation = @getTestAccuracyFromLinearLeastSquares;

  fh_getApproxKernel = @getApproximateRBFKernel;
  % fh_getApproxKernel = @getApproxKernelFastFood;


  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % SPCA-eigen
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  time_start = tic;
  L_actual = getActualKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance);
  tmp = X * H * L_actual * H * X';
  [U D V] = svd(tmp);
  output.duration_spca_eigen = toc(time_start);
  U = U(:,1:projected_dim);
  % U = U(:,1:projected_dim) * D(1:projected_dim,1:projected_dim).^0.5;

  projected_X = U' * X;
  projected_X_test = U' * X_test;

  output.accuracy_spca_eigen = fh_evaluation(projected_X, Y, projected_X_test, Y_test);
  projected_X_spca_eigen = projected_X;
  projected_X_test_spca_eigen = projected_X_test;


  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % SPCA-direct
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  oversample = false;

  if ~oversample

    time_start = tic;
    [~, psi, ~, ~] = fh_getApproxKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance, number_of_random_bases_for_labels);
    output.duration_spca_direct = toc(time_start);
    U = X * H * psi';

    projected_X = U' * X;
    projected_X_test = U' * X_test;

  else

    time_start = tic;
    [~, psi, ~, ~] = fh_getApproxKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance, number_of_random_bases_for_labels * 100);
    output.duration_spca_direct = toc(time_start);
    U = X * H * psi';

    projected_X = U' * X;
    projected_X_test = U' * X_test;

    [U,S,V] = svd(projected_X);
    projected_X = U(:,1:projected_dim)' * projected_X;
    projected_X_test = U(:,1:projected_dim)' * projected_X_test;

  end

  output.accuracy_spca_direct = fh_evaluation(projected_X, Y, projected_X_test, Y_test);
  projected_X_spca_direct = projected_X;
  projected_X_test_spca_direct = projected_X_test;


  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % % KSPCA-eigen
  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  % time_start = tic;
  % L_actual = getActualKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance);
  % K_train_actual = getActualKernel(X, X, data_rbf_variance);
  % K_test_actual = getActualKernel(X, X_test, data_rbf_variance);
  % tmp = H * L_actual * H * K_train_actual';
  % [U D V] = svd(tmp);
  % output.duration_kspca_eigen = toc(time_start);
  % U = U(:,1:projected_dim);
  % % U = U(:,1:projected_dim) * D(1:projected_dim,1:projected_dim).^0.5;

  % projected_X = U' * K_train_actual;
  % projected_X_test = U' * K_test_actual;

  % output.accuracy_kspca_eigen = fh_evaluation(projected_X, Y, projected_X_test, Y_test);
  % projected_X_kspca_eigen = projected_X;
  % projected_X_test_kspca_eigen = projected_X_test;


  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % KSPCA-direct
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  oversample = false;

  if ~oversample

    time_start = tic;
    [~, psi, ~, ~] = fh_getApproxKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance, number_of_random_bases_for_labels);
    [K_train_approx, ~, ~, params] = fh_getApproxKernel(X, X, data_rbf_variance, number_of_random_bases_for_data);
    [K_test_approx, ~, ~, ~] = fh_getApproxKernel(X, X_test, data_rbf_variance, number_of_random_bases_for_data, params);
    output.duration_kspca_direct = toc(time_start);

    projected_X = psi * H * K_train_approx;
    projected_X_test = psi * H * K_test_approx;

  else

    time_start = tic;
    [~, psi, ~, ~] = fh_getApproxKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance, number_of_random_bases_for_labels * 100);
    [K_train_approx, ~, ~, params] = fh_getApproxKernel(X, X, data_rbf_variance, number_of_random_bases_for_data);
    [K_test_approx, ~, ~, ~] = fh_getApproxKernel(X, X_test, data_rbf_variance, number_of_random_bases_for_data, params);
    output.duration_kspca_direct = toc(time_start);

    projected_X = psi * H * K_train_approx;
    projected_X_test = psi * H * K_test_approx;

    [U,S,V] = svd(projected_X);
    projected_X = U(:,1:projected_dim)' * projected_X;
    projected_X_test = U(:,1:projected_dim)' * projected_X_test;

  end

  output.accuracy_kspca_direct = fh_evaluation(projected_X, Y, projected_X_test, Y_test);
  projected_X_kspca_direct = projected_X;
  projected_X_test_kspca_direct = projected_X_test;














  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % % SPCA-aeigen
  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  % time_start = tic;
  % L_actual = getActualKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance);
  % tmp = X * H * L_actual * H * X';
  % [U D V] = approxSVD(tmp, projected_dim);
  % output.duration_spca_aeigen = toc(time_start);
  % U = U(:,1:projected_dim);

  % projected_X = U' * X;
  % projected_X_test = U' * X_test;

  % output.accuracy_spca_aeigen = fh_evaluation(projected_X, Y, projected_X_test, Y_test);
  % projected_X_spca_aeigen = projected_X;
  % projected_X_test_spca_aeigen = projected_X_test;


  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % % KSPCA-aeigen
  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

  % time_start = tic;
  % L_actual = getActualKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance);
  % K_train_actual = getActualKernel(X, X, data_rbf_variance);
  % K_test_actual = getActualKernel(X, X_test, data_rbf_variance);
  % tmp = H * L_actual * H * K_train_actual';
  % [U D V] = approxSVD(tmp, projected_dim);
  % output.duration_kspca_aeigen = toc(time_start);
  % U = U(:,1:projected_dim);

  % projected_X = U' * K_train_actual;
  % projected_X_test = U' * K_test_actual;

  % output.accuracy_kspca_aeigen = fh_evaluation(projected_X, Y, projected_X_test, Y_test);
  % projected_X_kspca_aeigen = projected_X;
  % projected_X_test_kspca_aeigen = projected_X_test;


  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % % PCA-direct
  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % Y_plus_noise = 1:length(Y); % so no information whatsoever

  % time_start = tic;
  % [~, psi, ~, ~] = fh_getApproxKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance, number_of_random_bases_for_labels);
  % U = X * H * psi';
  % output.duration_pca_direct = toc(time_start);

  % projected_X = U' * X;
  % projected_X_test = U' * X_test;

  % output.accuracy_pca_direct = fh_evaluation(projected_X, Y, projected_X_test, Y_test);
  % projected_X_pca_direct = projected_X;
  % projected_X_test_pca_direct = projected_X_test;


  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % % Random Projection
  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % time_start = tic;
  % tmp_D = number_of_random_bases_for_labels;
  % w = 1 / sqrt(tmp_D) * randn(tmp_D, size(X,1));
  % output.duration_random_projection = toc(time_start);

  % projected_X = w * X;
  % projected_X_test = w * X_test;

  % output.accuracy_random_projection = fh_evaluation(projected_X, Y, projected_X_test, Y_test);
  % projected_X_random_projection = projected_X;
  % projected_X_test_random_projection = projected_X_test;



  % -----------------------------------------------------------------------------
  % t-SNE
  % -----------------------------------------------------------------------------

  % initial_dims = 30;
  % if initial_dims >= d
  %   initial_dims = d;
  % end

  % perplexity_list = [5, 15, 30, 50];

  % figure,
  % for i = 1:4
  %   subplot(1,4,i),
  %   mappedX = tsne(X', [], 2, initial_dims, perplexity_list(i));
  %   gscatter(mappedX(:,1), mappedX(:,2), Y);
  % end

  % projected_X_all_spca_eigen_to_2D = tsne(double([projected_X_spca_eigen'; projected_X_test_spca_eigen']), [], 2, 10, 50)';
  % projected_X_all_spca_direct_to_2D = tsne(double([projected_X_spca_direct'; projected_X_test_spca_direct']), [], 2, 10, 50)';
  % projected_X_all_kspca_eigen_to_2D = tsne(double([projected_X_kspca_eigen'; projected_X_test_kspca_eigen']), [], 2, 10, 50)';
  % projected_X_all_kspca_direct_to_2D = tsne(double([projected_X_kspca_direct'; projected_X_test_kspca_direct']), [], 2, 10, 50)';

  % num_train = size(projected_X_spca_eigen, 2);
  % num_test = size(projected_X_test_spca_eigen, 2);
  % projected_X_spca_eigen = projected_X_all_spca_eigen_to_2D(:,1:num_train);
  % projected_X_test_spca_eigen = projected_X_all_spca_eigen_to_2D(:,num_train+1:end);
  % projected_X_spca_direct = projected_X_all_spca_direct_to_2D(:,1:num_train);
  % projected_X_test_spca_direct = projected_X_all_spca_direct_to_2D(:,num_train+1:end);
  % projected_X_kspca_eigen = projected_X_all_kspca_eigen_to_2D(:,1:num_train);
  % projected_X_test_kspca_eigen = projected_X_all_kspca_eigen_to_2D(:,num_train+1:end);
  % projected_X_kspca_direct = projected_X_all_kspca_direct_to_2D(:,1:num_train);
  % projected_X_test_kspca_direct = projected_X_all_kspca_direct_to_2D(:,num_train+1:end);



  % figure,

  % subplot(2,3,1)
  % plotPerClassTrainAndTestSamples(projected_X_spca_eigen, Y, projected_X_test_spca_eigen, Y_test);
  % title(sprintf('spca - acc: %.3f', output.accuracy_spca_eigen)),

  % subplot(2,3,2)
  % plotPerClassTrainAndTestSamples(projected_X_spca_aeigen, Y, projected_X_test_spca_aeigen, Y_test);
  % title(sprintf('spca-x - acc: %.3f', output.accuracy_spca_aeigen)),

  % subplot(2,3,3)
  % plotPerClassTrainAndTestSamples(projected_X_spca_direct, Y, projected_X_test_spca_direct, Y_test);
  % title(sprintf('srp - acc: %.3f', output.accuracy_spca_direct)),

  % subplot(2,3,4)
  % plotPerClassTrainAndTestSamples(projected_X_kspca_eigen, Y, projected_X_test_kspca_eigen, Y_test);
  % title(sprintf('kspca - acc: %.3f', output.accuracy_kspca_eigen)),

  % subplot(2,3,5)
  % plotPerClassTrainAndTestSamples(projected_X_kspca_aeigen, Y, projected_X_test_kspca_aeigen, Y_test);
  % title(sprintf('kspca-x - acc: %.3f', output.accuracy_kspca_aeigen)),

  % subplot(2,3,6)
  % plotPerClassTrainAndTestSamples(projected_X_kspca_direct, Y, projected_X_test_kspca_direct, Y_test);
  % title(sprintf('ksrp - acc: %.3f', output.accuracy_kspca_direct)),

  % % subplot(3,2,5)
  % % plotPerClassTrainAndTestSamples(projected_X_pca_direct, Y, projected_X_test_pca_direct, Y_test);
  % % title(sprintf('pca direct - acc: %.3f', accuracy_pca_direct)),

  % % subplot(3,2,6)
  % % plotPerClassTrainAndTestSamples(projected_X_random_projection, Y, projected_X_test_random_projection, Y_test);
  % % title(sprintf('random projection - acc: %.3f', accuracy_random_projection)),

  % % suptitle(dataset)

  % keyboard



  % figure,

  % subplot(2,2,1)
  % plotPerClassTrainAndTestSamples(projected_X_spca_eigen, Y, projected_X_test_spca_eigen, Y_test);
  % title('SPCA'),

  % subplot(2,2,2)
  % plotPerClassTrainAndTestSamples(projected_X_spca_direct, Y, projected_X_test_spca_direct, Y_test);
  % title('SRP'),

  % subplot(2,2,3)
  % plotPerClassTrainAndTestSamples(projected_X_kspca_eigen, Y, projected_X_test_kspca_eigen, Y_test);
  % title('KSPCA'),

  % subplot(2,2,4)
  % legend_cell_array = plotPerClassTrainAndTestSamples(projected_X_kspca_direct, Y, projected_X_test_kspca_direct, Y_test);
  % title('KSRP'),

  % % hL = legend([line1,line2,line3,line4],{'Data Axes 1','Data Axes 2','Data Axes 3','Data Axes 4'});
  % hL = legend(legend_cell_array);
  % % Programatically move the Legend
  % new_position = [0.43 0.43 0.2 0.2];
  % new_units = 'normalized';
  % set(hL,'Position', new_position,'Units', new_units);






  output
  keyboard

  % figure,

  % subplot(1,2,1)
  % plotPerClassTrainAndTestSamples(projected_X_kspca_eigen, Y, projected_X_test_kspca_eigen, Y_test);
  % title('KSPCA'),

  % subplot(1,2,2)
  % legend_cell_array = plotPerClassTrainAndTestSamples(projected_X_kspca_direct, Y, projected_X_test_kspca_direct, Y_test);
  % title('KSRP'),

  % % hL = legend([line1,line2,line3,line4],{'Data Axes 1','Data Axes 2','Data Axes 3','Data Axes 4'});
  % hL = legend(legend_cell_array);
  % % Programatically move the Legend
  % new_position = [0.43 0.43 0.2 0.2];
  % new_units = 'normalized';
  % set(hL,'Position', new_position,'Units', new_units);





  % % % saveas(gcf, sprintf('visualization_%s', dataset), 'epsc')

  % % keyboard



























% % -------------------------------------------------------------------------
% function L_actual = getActualKernel(data_1, data_2, rbf_variance)
%   % data_* consists of 1 sample per column
% % -------------------------------------------------------------------------
%   assert(size(data_1, 1) == size(data_2, 1));
%   tmp = pdist2(data_1',data_2','euclidean') .^ 2;
%   L_actual = exp(- tmp / (2 * rbf_variance ^ 2));


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
function [L_approx, psi_data_1, psi_data_2, params] = getApproxKernelFastFood(data_1, data_2, rbf_variance, number_of_random_bases, params);
  % data consists of 1 sample per column
% -------------------------------------------------------------------------
  assert(size(data_1, 1) == size(data_2, 1));
  d = size(data_1, 1);
  D = number_of_random_bases;
  s = rbf_variance;

  try
    % test whether we can use Spiral package
    fwht_spiral([1; 1]);
    use_spiral = 1;
  catch
      % display('Cannot perform Walsh-Hadamard transform using Spiral WHT package.');
      % display('Use Matlab function fwht instead, which is slow for large-scale data.')
      use_spiral = 0;
  end

  if nargin == 5
    params = params;
  else
    params = FastfoodPara(D, d);
  end

  psi_data_1 = FastfoodForKernel(data_1, params, s, use_spiral);
  psi_data_2 = FastfoodForKernel(data_2, params, s, use_spiral);
  L_approx = psi_data_1' * psi_data_2;






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



% % -------------------------------------------------------------------------
% function [projected_X, projected_X_test] = getProposedNNProjections(X, X_test, data_rbf_variance, number_of_random_bases_for_data, psi, H, kernel_type);
% % -------------------------------------------------------------------------
%   if strcmp(kernel_type, 'actual');
%     K_train_actual = getActualKernel(X, X, data_rbf_variance);
%     K_test_actual = getActualKernel(X, X_test, data_rbf_variance);
%     X = psi * H * K_train_actual;
%     X_test = psi * H * K_test_actual;
%   elseif strcmp(kernel_type, 'approx');
%     [K_train_approx, ~, ~, random_weight_matrix] = getApproxKernelRKS(X, X, data_rbf_variance, number_of_random_bases_for_data, -1);
%     [K_test_approx, ~, ~, ~] = getApproxKernelRKS(X, X_test, data_rbf_variance, number_of_random_bases_for_data, -1, random_weight_matrix);
%     X = psi * H * K_train_approx;
%     X_test = psi * H * K_test_approx;
%   end
%   projected_X = X;
%   projected_X_test = X_test;













