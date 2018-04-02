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



  % dataset = 'xor-10D-350-train-150-test';
  % % dataset = 'usps';
  % % dataset = 'mnist-784';
  % % dataset = 'uci-sonar';


  % if strcmp(dataset, 'xor-10D-350-train-150-test')
  %   projected_dim_list = [1:1:10];
  % elseif strcmp(dataset, 'usps')
  %   % projected_dim_list = [1:1:256];
  %   projected_dim_list = [1:15, 2.^[4:1:8]];
  % elseif strcmp(dataset, 'mnist-784')
  %   projected_dim_list = 2.^[0:1:9];
  % elseif strcmp(dataset, 'uci-sonar')
  %   projected_dim_list = [1:9,10:25:55,60];
  % end

  % repeat_count = 30;
  % average_per_dim_delta_pca_rpca = zeros(repeat_count, length(projected_dim_list));
  % average_per_dim_delta_pca_rpca_2 = zeros(repeat_count, length(projected_dim_list));
  % average_per_dim_delta_pca_rp = zeros(repeat_count, length(projected_dim_list));

  % for ii = 1:repeat_count

  %   fprintf('[INFO] trial #%d\n', ii);

  %   if strcmp(dataset, 'xor-10D-350-train-150-test')
  %     tmp_opts.dataset = dataset;
  %     imdb = loadSavedImdb(tmp_opts, false);
  %   elseif strcmp(dataset, 'usps')
  %     imdb = constructMultiClassImdbs(dataset, false);
  %     imdb = createImdbWithBalance(dataset, imdb, 25, 25, false, false);
  %   elseif strcmp(dataset, 'mnist-784')
  %     imdb = constructMultiClassImdbs(dataset, false);
  %     imdb = createImdbWithBalance(dataset, imdb, 25, 25, false, false);
  %     % imdb = createImdbWithBalance(dataset, imdb, 500, 500, false, false);
  %   elseif strcmp(dataset, 'uci-sonar')
  %     imdb = constructMultiClassImdbs(dataset, false);
  %   end

  %   per_dim_delta_pca_rpca = [];
  %   per_dim_delta_pca_rpca_2 = [];
  %   per_dim_delta_pca_rp = [];

  %   for projected_dim = projected_dim_list
  %     % projected_dim = 2;

  %     %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %     % Meta
  %     %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %     vectorized_imdb = getVectorizedImdb(imdb);
  %     indices_train = imdb.images.set == 1;
  %     indices_test = imdb.images.set == 3;
  %     X = vectorized_imdb.images.data(indices_train,:)';
  %     % X = randn(2,3);
  %     Y = vectorized_imdb.images.labels(indices_train);
  %     % Y = [1,2,3];
  %     X_test = vectorized_imdb.images.data(indices_test,:)';
  %     Y_test = vectorized_imdb.images.labels(indices_test);


  %     %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %     % Compute PCA (on training data only)
  %     %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %     tmp_X = X';
  %     tmp_X = bsxfun(@minus, tmp_X, mean(tmp_X,1));           % zero-center
  %     C = (tmp_X'*tmp_X)./(size(tmp_X,1)-1);                  %' cov(X)

  %     [V D] = eig(C);
  %     [D order] = sort(diag(D), 'descend');                   % sort cols high to low
  %     V = V(:,order);
  %     % [U D V] = svd(tmp_X');
  %     % D_1 = D(1:2,1:2);

  %     % n = length(Y);
  %     % H = eye(n) - 1 / n * (ones(n,n));
  %     % tmp = X * H * H * X';
  %     % [U D V] = svd(tmp);
  %     % D_2 = D;

  %     % (diag(D_1)').^2
  %     % diag(D_2)'
  %     % assert(numel(find((diag(D_1)').^2 - diag(D_2)' > 10e-3)) == 0)

  %     % (X * H)
  %     % (X * H)'
  %     % (U * D.^0.5)
  %     % (U * D.^0.5)'
  %     % (X * H) * (X * H)'
  %     % (U * D.^0.5) * (U * D.^0.5)'
  %     % % X * H = U * D.^0.5 up to a rotation
  %     % % H * X'
  %     % % D.^0.5 * U'
  %     % D_inv_half = pinv(diag([diag(D); 0])).^0.5;
  %     % U_1 = X * H * D_inv_half %  = U % X * H * D.^-0.5 = U
  %     % U_1=U_1(:,1:2)

  %     % U_2 = D_inv_half * H * X'
  %     % (U * D.^0.5)'

  %     % U' * U
  %     % U_1' * U_1

  %     % keyboard

  %     U_pca_exact = V(:,1:projected_dim);
  %     % U_pca_exact = U(:,1:projected_dim);
  %     % U_pca_exact = real(V(:,1:projected_dim) * diag(D(1:projected_dim).^0.5));
  %     projected_X_pca_exact = U_pca_exact' *  X;
  %     projected_X_test_pca_exact = U_pca_exact' *  X_test;

  %     tmp = squareform(pdist(projected_X_pca_exact'));
  %     distance_matrix_pca_exact = tmp / max(tmp(:));


  %     % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %     % % Random PCA
  %     % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %     % n = length(Y);
  %     % H = eye(n) - 1 / n * (ones(n,n));
  %     % label_rbf_variance = 10e-10;
  %     % number_of_random_bases_for_labels = 100;
  %     % [L_approx_w_many_bases, psi_w_many_bases, ~, ~] = getApproxKernel(1:n, 1:n, label_rbf_variance, number_of_random_bases_for_labels, -1);

  %     % tmp_projected_dim = projected_dim;
  %     % tmp_psi = psi_w_many_bases';
  %     % tmp_psi = bsxfun(@minus, tmp_psi, mean(tmp_psi,1));           % zero-center
  %     % C = (tmp_psi'*tmp_psi)./(size(tmp_psi,1)-1);                  %' cov(X)
  %     % [V D] = eig(C);
  %     % [D order] = sort(diag(D), 'descend');                         % sort cols high to low
  %     % V = V(:,order);
  %     % psi_w_few_bases = V(:,1:tmp_projected_dim)' * psi_w_many_bases;

  %     % % C = X * H * (psi_w_few_bases' * psi_w_few_bases) * H * X'./(size(tmp_X,1)-1);
  %     % % [V D] = eig(C);
  %     % % [D order] = sort(diag(D), 'descend');                   % sort cols high to low
  %     % % V = real(V(:,order));

  %     % % U_pca_random = V(:,1:projected_dim);
  %     % U_pca_random = X * H * psi_w_few_bases';
  %     % projected_X_pca_random = U_pca_random' * X;
  %     % projected_X_test_pca_random = U_pca_random' * X_test;

  %     % tmp = squareform(pdist(projected_X_pca_random'));
  %     % distance_matrix_pca_random = tmp / max(tmp(:));

  %     % % fprintf('Norm Diff PCA-RPCA: %.3f\n', norm(distance_matrix_pca_exact - distance_matrix_pca_random, 'fro'));



  %     % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %     % % Random PCA
  %     % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %     % n = length(Y);
  %     % H = eye(n) - 1 / n * (ones(n,n));
  %     % label_rbf_variance = 10e-10;
  %     % number_of_random_bases_for_labels = projected_dim;
  %     % [L_approx, psi, ~, ~] = getApproxKernel(1:n, 1:n, label_rbf_variance, number_of_random_bases_for_labels, -1);

  %     % C = X * H * L_approx * H * X'./(size(tmp_X,1)-1);
  %     % % C = X * H * eye(size(H,1)) * H * X'./(size(tmp_X,1)-1);

  %     % [V D] = eig(C);
  %     % [D order] = sort(diag(D), 'descend');                   % sort cols high to low
  %     % keyboard
  %     % V = real(V(:,order));

  %     % U_pca_random = V(:,1:projected_dim);
  %     % projected_X_pca_random = U_pca_random' * X;
  %     % projected_X_test_pca_random = U_pca_random' * X_test;

  %     % tmp = squareform(pdist(projected_X_pca_random'));
  %     % distance_matrix_pca_random = tmp / max(tmp(:));

  %     % % fprintf('Norm Diff PCA-RPCA: %.3f\n', norm(distance_matrix_pca_exact - distance_matrix_pca_random, 'fro'));



  %     %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %     % Random PCA
  %     %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %     n = length(Y);
  %     H = eye(n) - 1 / n * (ones(n,n));
  %     label_rbf_variance = 10e-10;
  %     number_of_random_bases_for_labels = projected_dim;
  %     [L_approx, psi, ~, ~] = getApproxKernel(1:n, 1:n, label_rbf_variance, number_of_random_bases_for_labels, -1);

  %     % tmp = X * H * H * X';
  %     % [U D V] = svd(tmp);
  %     % inverse_D_to_the_half = diag(diag(D(1:projected_dim,1:projected_dim)).^(-.5)); % = inv(D(1:projected_dim,1:projected_dim).^(.5));
  %     % % inverse_D_to_the_half = eye(projected_dim);
  %     % % keyboard

  %     % U_pca_random = X * H * psi' * inverse_D_to_the_half;
  %     % U_pca_random = normc(X * H * psi');
  %     U_pca_random = X * H * psi';
  %     projected_X_pca_random = U_pca_random' * X;
  %     projected_X_test_pca_random = U_pca_random' * X_test;

  %     tmp = squareform(pdist(projected_X_pca_random'));
  %     distance_matrix_pca_random = tmp / max(tmp(:));




  %     %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %     % Random PCA
  %     %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %     % n = length(Y);
  %     % H = eye(n) - 1 / n * (ones(n,n));
  %     % label_rbf_variance = 10e-10;
  %     % number_of_random_bases_for_labels = projected_dim;
  %     % [L_approx, psi, ~, ~] = getApproxKernel(1:n, 1:n, label_rbf_variance, number_of_random_bases_for_labels, -1);

  %     tmp = X * H * H * X';
  %     [U D V] = svd(tmp);
  %     D_2 = D;
  %     keyboard
  %     inverse_D_to_the_half = diag(diag(D(1:projected_dim,1:projected_dim)).^(-.5)); % = inv(D(1:projected_dim,1:projected_dim).^(.5));
  %     % inverse_D_to_the_half = eye(projected_dim);
  %     % keyboard

  %     U_pca_random = X * H * psi' * inverse_D_to_the_half;
  %     % U_pca_random = normc(X * H * psi');
  %     % U_pca_random = X * H * psi';
  %     projected_X_pca_random = U_pca_random' * X;
  %     projected_X_test_pca_random = U_pca_random' * X_test;

  %     tmp = squareform(pdist(projected_X_pca_random'));
  %     distance_matrix_pca_random_2 = tmp / max(tmp(:));







  %     %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %     % Random Projection
  %     %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %     w = 1 / sqrt(projected_dim) * randn(projected_dim, size(X,1));
  %     projected_X = w * X;
  %     projected_X_test = w * X_test;

  %     projected_X_random_projection = projected_X;
  %     projected_X_test_random_projection = projected_X_test;

  %     tmp = squareform(pdist(projected_X_random_projection'));
  %     distance_matrix_random_projection = tmp / max(tmp(:));



  %     %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %     % Analysis
  %     %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  %     % figure,

  %     % subplot(1,3,1),
  %     % imshow(distance_matrix_pca_exact),

  %     % subplot(1,3,2),
  %     % imshow(distance_matrix_pca_random),

  %     % subplot(1,3,3),
  %     % imshow(distance_matrix_random_projection),

  %     per_dim_delta_pca_rpca(end+1) = norm(distance_matrix_pca_exact - distance_matrix_pca_random, 'fro');
  %     per_dim_delta_pca_rpca_2(end+1) = norm(distance_matrix_pca_exact - distance_matrix_pca_random_2, 'fro');
  %     per_dim_delta_pca_rp(end+1) = norm(distance_matrix_pca_exact - distance_matrix_random_projection, 'fro');

  %     % fprintf('[k = %d] |D_{PCA} - D_{RPCA}|_F = %.3f\n', projected_dim, per_dim_delta_pca_rpca(end));
  %     % fprintf('[k = %d] |D_{PCA} - D_{RP}|_F = %.3f\n', projected_dim, per_dim_delta_pca_rp(end));

  %   end
  %   average_per_dim_delta_pca_rpca(ii,:) = per_dim_delta_pca_rpca;
  %   average_per_dim_delta_pca_rpca_2(ii,:) = per_dim_delta_pca_rpca_2;
  %   average_per_dim_delta_pca_rp(ii,:) = per_dim_delta_pca_rp;
  % end


  % legend_cell_array = {};
  % figure,
  % grid on,
  % hold on,
  % plot(projected_dim_list, mean(average_per_dim_delta_pca_rpca), 'LineWidth', 2); legend_cell_array = [legend_cell_array, '|D_{PCA} - D_{RPCA}|_F'];
  % plot(projected_dim_list, mean(average_per_dim_delta_pca_rpca_2), 'LineWidth', 2); legend_cell_array = [legend_cell_array, '|D_{PCA} - D_{RPCA_2}|_F'];
  % plot(projected_dim_list, mean(average_per_dim_delta_pca_rp), 'LineWidth', 2); legend_cell_array = [legend_cell_array, '|D_{PCA} - D_{RP}|_F'];
  % hold off,

  % set(gca, 'YLim', [0, get(gca, 'YLim') * [0; 1]])

  % xlabel('Projected Dimension');
  % ylabel('Delta in Frob Norm');
  % title(sprintf('%s - Comparison of RPCA vs RP (avg of %d)', dataset, repeat_count));
  % legend(legend_cell_array, 'Location','northeast');













  % keyboard
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


  % if strcmp(dataset, 'xor-10D-350-train-150-test') || ...
  %    strcmp(dataset, 'rings-10D-350-train-150-test') || ...
  %    strcmp(dataset, 'spirals-10D-350-train-150-test')
  %   tmp_opts.dataset = dataset;
  %   imdb = loadSavedImdb(tmp_opts, false);
  % % elseif strcmp(dataset, 'mnist')
  % %   tmp_opts.dataset = 'mnist-784-multi-class-subsampled';
  % %   tmp_opts.posneg_balance = 'balanced-250';
  % %   imdb = loadSavedImdb(tmp_opts, false);
  % else
    imdb = constructMultiClassImdbs(dataset, false);
    if strcmp(dataset, 'usps')
      imdb = createImdbWithBalance(dataset, imdb, 25, 25, false, false);
    elseif strcmp(dataset, 'mnist-784')
      % imdb = createImdbWithBalance(dataset, imdb, 25, 25, false, false);
      % imdb = createImdbWithBalance(dataset, imdb, 100, 100, false, false);
      % imdb = createImdbWithBalance(dataset, imdb, 500, 500, false, false);
      imdb = createImdbWithBalance(dataset, imdb, 1000, 200, false, false);
      % imdb = createImdbWithBalance(dataset, imdb, 2500, 500, false, false);
    elseif strcmp(dataset, 'uci-spam')
      imdb = createImdbWithBalance(dataset, imdb, 1000, 250, false, false);
    end
  % end
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
  X(X > 1) = 1; % HACKY???
  X_test(X_test > 1) = 1; % HACKY???

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
  elseif strcmp(dataset, 'mnist-784')
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


  % -----------------------------------------------------------------------------
  % Compare projections
  % -----------------------------------------------------------------------------

  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % SPCA-eigen
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  time_start = tic;
  L_actual = getActualKernel(Y, Y, label_rbf_variance);
  tmp = X * H * L_actual * H * X';
  [U D V] = svd(tmp);
  U = U(:,1:projected_dim);
  projected_X = U' * X;
  projected_X_test = U' * X_test;
  output.accuracy_spca_eigen = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);
  output.duration_spca_eigen = toc(time_start);

  projected_X_spca_eigen = projected_X;
  projected_X_test_spca_eigen = projected_X_test;


  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % KSPCA-eigen
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % Y_plus_noise = Y;
  Y_plus_noise = Y + randn(1, size(Y, 2)) / 10e+5;
  time_start = tic;
  L_actual = getActualKernel(Y, Y, label_rbf_variance);
  K_train_actual = getActualKernel(X, X, data_rbf_variance);
  K_test_actual = getActualKernel(X, X_test, data_rbf_variance);
  tmp = H * L_actual * H * K_train_actual';
  [U D V] = svd(tmp); % TODO: is it OK to use SVD? or should I use eigendec which is broken??
  U = U(:,1:projected_dim);
  projected_X = U' * K_train_actual;
  projected_X_test = U' * K_test_actual;
  output.accuracy_kspca_eigen = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);
  output.duration_kspca_eigen = toc(time_start);

  projected_X_kspca_eigen = projected_X;
  projected_X_test_kspca_eigen = projected_X_test;


  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % SPCA-direct
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % Y_plus_noise = Y;
  Y_plus_noise = Y + randn(1, size(Y, 2)) / 10e+5;

  time_start = tic;
  [L_approx, psi, ~, ~] = getApproxKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance, number_of_random_bases_for_labels, projected_dim);
  U = X * H * psi';
  % U = X * psi';
  projected_X = U' * X;
  projected_X_test = U' * X_test;
  output.accuracy_spca_direct = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);
  output.duration_spca_direct = toc(time_start);

  projected_X_spca_direct = projected_X;
  projected_X_test_spca_direct = projected_X_test;


  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % KSPCA-direct
  %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  Y_plus_noise = Y;
  % Y_plus_noise = Y + randn(1, size(Y, 2)) / 10e+5;

  time_start = tic;
  [L_approx, psi, ~, ~] = getApproxKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance, number_of_random_bases_for_labels, projected_dim);
  [K_train_approx, ~, ~, random_weight_matrix] = getApproxKernel(X, X, data_rbf_variance, number_of_random_bases_for_data, projected_dim);
  [K_test_approx, ~, ~, ~] = getApproxKernel(X, X_test, data_rbf_variance, number_of_random_bases_for_data, projected_dim, random_weight_matrix);
  % projected_X = psi * H * K_train_approx * K_train_approx;
  % projected_X_test = psi * H * K_train_approx * K_test_approx;
  % projected_X = psi * H * K_train_approx * K_train_approx;
  % projected_X_test = psi * H * K_train_approx * K_test_approx;
  projected_X = psi * H * K_train_approx;
  projected_X_test = psi * H * K_test_approx;
  % projected_X = psi * K_train_approx;
  % projected_X_test = psi * K_test_approx;
  output.accuracy_kspca_direct = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);
  output.duration_kspca_direct = toc(time_start);

  projected_X_kspca_direct = projected_X;
  projected_X_test_kspca_direct = projected_X_test;



  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % % PCA-direct
  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % Y_plus_noise = 1:length(Y); % so no information whatsoever

  % time_start = tic;
  % [L_approx, psi, ~, ~] = getApproxKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance, number_of_random_bases_for_labels, projected_dim);
  % U = X * H * psi';
  % projected_X = U' * X;
  % projected_X_test = U' * X_test;
  % output.accuracy_pca_direct = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);
  % output.duration_pca_direct = toc(time_start);

  % projected_X_pca_direct = projected_X;
  % projected_X_test_pca_direct = projected_X_test;


  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % % Random Projection
  % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  % tmp_D = number_of_random_bases_for_labels;
  % w = 1 / sqrt(tmp_D) * randn(tmp_D, size(X,1));
  % projected_X = w * X;
  % projected_X_test = w * X_test;

  % output.accuracy_random_projection = getTestAccuracyFrom1NN(projected_X, Y, projected_X_test, Y_test);
  % output.duration_random_projection = toc(time_start);

  % projected_X_random_projection = projected_X;
  % projected_X_test_random_projection = projected_X_test;



  % figure,

  % subplot(3,2,1)
  % plotPerClassTrainAndTestSamples(projected_X_spca_eigen, Y, projected_X_test_spca_eigen, Y_test);
  % title('spca eigen')

  % subplot(3,2,2)
  % plotPerClassTrainAndTestSamples(projected_X_kspca_eigen, Y, projected_X_test_kspca_eigen, Y_test);
  % title('kspca eigen')

  % subplot(3,2,3)
  % plotPerClassTrainAndTestSamples(projected_X_spca_direct, Y, projected_X_test_spca_direct, Y_test);
  % title('spca direct')

  % subplot(3,2,4)
  % plotPerClassTrainAndTestSamples(projected_X_kspca_direct, Y, projected_X_test_kspca_direct, Y_test);
  % title('kspca direct')

  % subplot(3,2,5)
  % plotPerClassTrainAndTestSamples(projected_X_pca_direct, Y, projected_X_test_pca_direct, Y_test);
  % title('pca direct')

  % subplot(3,2,6)
  % plotPerClassTrainAndTestSamples(projected_X_random_projection, Y, projected_X_test_random_projection, Y_test);
  % title('random projection')

  % suptitle(dataset)

  % keyboard





















  % -----------------------------------------------------------------------------
  % Compare Hierarchies
  % -----------------------------------------------------------------------------







  % fh_evaluation = @getTestAccuracyFrom1NN;
  % fh_evaluation = @getTestAccuracyFromLinearLeastSquares;


  % % D = size(X, 1); % number_of_random_basis;
  % D = 2; % number_of_random_basis_per_layer OR number_of_hidden_nodes_per_layer
  % s = data_rbf_variance;

  % % ----------------------------------------------------------------------------
  % nonlin = @cos;
  % % ----------------------------------------------------------------------------

  % X_0 = X;
  % X_test_0 = X_test;

  % tmp_matrix = randn(D, size(X_0, 1)) / s;
  % X_1 = sqrt(1/D) * nonlin(tmp_matrix * X_0);
  % X_test_1 = sqrt(1/D) * nonlin(tmp_matrix * X_test_0);

  % tmp_matrix = randn(D, size(X_1, 1)) / s;
  % X_2 = sqrt(1/D) * nonlin(tmp_matrix * X_1);
  % X_test_2 = sqrt(1/D) * nonlin(tmp_matrix * X_test_1);

  % tmp_matrix = randn(D, size(X_2, 1)) / s;
  % X_3 = sqrt(1/D) * nonlin(tmp_matrix * X_2);
  % X_test_3 = sqrt(1/D) * nonlin(tmp_matrix * X_test_2);

  % tmp_matrix = randn(D, size(X_3, 1)) / s;
  % X_4 = sqrt(1/D) * nonlin(tmp_matrix * X_3);
  % X_test_4 = sqrt(1/D) * nonlin(tmp_matrix * X_test_3);



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

  % output.accuracy_proposed_0 = fh_evaluation(X_0, Y, X_test_0, Y_test);
  % output.accuracy_proposed_1 = fh_evaluation(X_1, Y, X_test_1, Y_test);
  % output.accuracy_proposed_2 = fh_evaluation(X_2, Y, X_test_2, Y_test);
  % output.accuracy_proposed_3 = fh_evaluation(X_3, Y, X_test_3, Y_test);
  % output.accuracy_proposed_4 = fh_evaluation(X_4, Y, X_test_4, Y_test);

  % % figure,

  % % subplot(2,5,1), plotPerClassTrainAndTestSamples(X_0, Y, X_test_0, Y_test), title('After 0 layers (cos)'),
  % % subplot(2,5,2), plotPerClassTrainAndTestSamples(X_1, Y, X_test_1, Y_test), title('After 1 layers (cos)'),
  % % subplot(2,5,3), plotPerClassTrainAndTestSamples(X_2, Y, X_test_2, Y_test), title('After 2 layers (cos)'),
  % % subplot(2,5,4), plotPerClassTrainAndTestSamples(X_3, Y, X_test_3, Y_test), title('After 3 layers (cos)'),
  % % subplot(2,5,5), plotPerClassTrainAndTestSamples(X_4, Y, X_test_4, Y_test), title('After 4 layers (cos)'),




  % % ----------------------------------------------------------------------------
  % nonlin = @relu;
  % % ----------------------------------------------------------------------------

  % X_0 = X;
  % X_test_0 = X_test;

  % tmp_matrix = randn(D, size(X_0, 1)) / s;
  % X_1 = sqrt(1/D) * nonlin(tmp_matrix * X_0);
  % X_test_1 = sqrt(1/D) * nonlin(tmp_matrix * X_test_0);

  % tmp_matrix = randn(D, size(X_1, 1)) / s;
  % X_2 = sqrt(1/D) * nonlin(tmp_matrix * X_1);
  % X_test_2 = sqrt(1/D) * nonlin(tmp_matrix * X_test_1);

  % tmp_matrix = randn(D, size(X_2, 1)) / s;
  % X_3 = sqrt(1/D) * nonlin(tmp_matrix * X_2);
  % X_test_3 = sqrt(1/D) * nonlin(tmp_matrix * X_test_2);

  % tmp_matrix = randn(D, size(X_3, 1)) / s;
  % X_4 = sqrt(1/D) * nonlin(tmp_matrix * X_3);
  % X_test_4 = sqrt(1/D) * nonlin(tmp_matrix * X_test_3);

  % output.accuracy_rp_0 = fh_evaluation(X_0, Y, X_test_0, Y_test);
  % output.accuracy_rp_1 = fh_evaluation(X_1, Y, X_test_1, Y_test);
  % output.accuracy_rp_2 = fh_evaluation(X_2, Y, X_test_2, Y_test);
  % output.accuracy_rp_3 = fh_evaluation(X_3, Y, X_test_3, Y_test);
  % output.accuracy_rp_4 = fh_evaluation(X_4, Y, X_test_4, Y_test);

  % % subplot(2,5,6), plotPerClassTrainAndTestSamples(X_0, Y, X_test_0, Y_test), title('After 0 layers (relu)'),
  % % subplot(2,5,7), plotPerClassTrainAndTestSamples(X_1, Y, X_test_1, Y_test), title('After 1 layers (relu)'),
  % % subplot(2,5,8), plotPerClassTrainAndTestSamples(X_2, Y, X_test_2, Y_test), title('After 2 layers (relu)'),
  % % subplot(2,5,9), plotPerClassTrainAndTestSamples(X_3, Y, X_test_3, Y_test), title('After 3 layers (relu)'),
  % % subplot(2,5,10), plotPerClassTrainAndTestSamples(X_4, Y, X_test_4, Y_test), title('After 4 layers (relu)'),

  % % suptitle(dataset)

  % % keyboard





















  % should_plot_visu = false;
  % if should_plot_visu; figure, end;

  % X_original = X;
  % X_test_original = X_test;


  % % D = 10; % number_of_random_basis_per_layer OR number_of_hidden_nodes_per_layer
  % D = projected_dim; % number_of_random_basis_per_layer OR number_of_hidden_nodes_per_layer

  % Y_plus_noise = Y;
  % % Y_plus_noise = Y + randn(1, size(Y, 2)) / 1;
  % % Y_plus_noise = Y + randn(1, size(Y, 2)) / 10e+2;
  % Y_plus_noise = Y + randn(1, size(Y, 2)) / 10e+5;
  % % Y_plus_noise = Y + randn(1, size(Y, 2)) / 10e+10;

  % % [L_approx, psi, ~, ~] = getApproxKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance, D, -1);
  % % fprintf('\n\t Rank L_approx: %d\n', rank(L_approx));
  % % keyboard

  % % % D = 16;
  % % % figure
  % % [L_approx, psi, ~, ~] = getApproxKernel(Y, Y, label_rbf_variance, D, -1);
  % % % rank(L_approx),
  % % subplot(1,2,1),
  % % imshow(L_approx),

  % % [L_approx, psi, ~, ~] = getApproxKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance, D, -1);
  % % % rank(L_approx),
  % % subplot(1,2,2),
  % % imshow(L_approx),


  % time_start = tic;
  % [L_approx, psi, ~, ~] = getApproxKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance, D, -1);
  % X = X;
  % X_test = X_test;
  % % fprintf('\t [Layer 0] \t rank(X): %d \t rank(X_test): %d\n', rank(X), rank(X_test));
  % output.accuracy_proposed_0 = getTestAccuracyFromMLP(X, Y, X_test, Y_test, [D]);
  % output.duration_proposed_0 = toc(time_start);
  % if should_plot_visu; subplot(1,5,1), plotPerClassTrainAndTestSamples(X, Y, X_test, Y_test), title('After 0 layers (proposed)'), end;

  % time_start = tic;
  % [L_approx, psi, ~, ~] = getApproxKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance, D, -1);
  % [X, X_test] = getProposedNNProjections(X, X_test, data_rbf_variance, number_of_random_bases_for_data, psi, H, 'approx');
  % % fprintf('\t [Layer 1] \t rank(X): %d \t rank(X_test): %d\n', rank(X), rank(X_test));
  % output.accuracy_proposed_1 = getTestAccuracyFromMLP(X, Y, X_test, Y_test, [D]);
  % output.duration_proposed_1 = toc(time_start);
  % if should_plot_visu; subplot(1,5,2), plotPerClassTrainAndTestSamples(X, Y, X_test, Y_test), title('After 1 layers (proposed)'), end;

  % time_start = tic;
  % [L_approx, psi, ~, ~] = getApproxKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance, D, -1);
  % [X, X_test] = getProposedNNProjections(X, X_test, data_rbf_variance, number_of_random_bases_for_data, psi, H, 'approx');
  % % fprintf('\t [Layer 2] \t rank(X): %d \t rank(X_test): %d\n', rank(X), rank(X_test));
  % output.accuracy_proposed_2 = getTestAccuracyFromMLP(X, Y, X_test, Y_test, [D]);
  % output.duration_proposed_2 = toc(time_start);
  % if should_plot_visu; subplot(1,5,3), plotPerClassTrainAndTestSamples(X, Y, X_test, Y_test), title('After 2 layers (proposed)'), end;

  % % time_start = tic;
  % % [L_approx, psi, ~, ~] = getApproxKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance, D, -1);
  % % [X, X_test] = getProposedNNProjections(X, X_test, data_rbf_variance, number_of_random_bases_for_data, psi, H, 'approx');
  % % % fprintf('\t [Layer 3] \t rank(X): %d \t rank(X_test): %d\n', rank(X), rank(X_test));
  % % output.accuracy_proposed_3 = getTestAccuracyFromMLP(X, Y, X_test, Y_test, [D]);
  % % output.duration_proposed_3 = toc(time_start);
  % % if should_plot_visu; subplot(1,5,4), plotPerClassTrainAndTestSamples(X, Y, X_test, Y_test), title('After 3 layers (proposed)'), end;

  % % time_start = tic;
  % % [L_approx, psi, ~, ~] = getApproxKernel(Y_plus_noise, Y_plus_noise, label_rbf_variance, D, -1);
  % % [X, X_test] = getProposedNNProjections(X, X_test, data_rbf_variance, number_of_random_bases_for_data, psi, H, 'approx');
  % % % fprintf('\t [Layer 4] \t rank(X): %d \t rank(X_test): %d\n', rank(X), rank(X_test));
  % % output.accuracy_proposed_4 = getTestAccuracyFromMLP(X, Y, X_test, Y_test, [D]);
  % % output.duration_proposed_4 = toc(time_start);
  % % if should_plot_visu; subplot(1,5,5), plotPerClassTrainAndTestSamples(X, Y, X_test, Y_test), title('After 4 layers (proposed)'), end;

  % if should_plot_visu; keyboard, end;





  % X_0 = X_original;
  % X_test_0 = X_test_original;

  % time_start = tic;
  % output.accuracy_backprop_0 = getTestAccuracyFromMLP(X_0, Y, X_test_0, Y_test, [D]);
  % output.duration_backprop_0 = toc(time_start);

  % time_start = tic;
  % output.accuracy_backprop_1 = getTestAccuracyFromMLP(X_0, Y, X_test_0, Y_test, [D, D]);
  % output.duration_backprop_1 = toc(time_start);

  % time_start = tic;
  % output.accuracy_backprop_2 = getTestAccuracyFromMLP(X_0, Y, X_test_0, Y_test, [D, D, D]);
  % output.duration_backprop_2 = toc(time_start);

  % % time_start = tic;
  % % output.accuracy_backprop_3 = getTestAccuracyFromMLP(X_0, Y, X_test_0, Y_test, [D, D, D, D]);
  % % output.duration_backprop_3 = toc(time_start);

  % % time_start = tic;
  % % output.accuracy_backprop_4 = getTestAccuracyFromMLP(X_0, Y, X_test_0, Y_test, [D, D, D, D, D]);
  % % output.duration_backprop_4 = toc(time_start);











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

  % k = projected_dim;
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
function [projected_X, projected_X_test] = getProposedNNProjections(X, X_test, data_rbf_variance, number_of_random_bases_for_data, psi, H, kernel_type);
% -------------------------------------------------------------------------
  if strcmp(kernel_type, 'actual');
    K_train_actual = getActualKernel(X, X, data_rbf_variance);
    K_test_actual = getActualKernel(X, X_test, data_rbf_variance);
    X = psi * H * K_train_actual;
    X_test = psi * H * K_test_actual;
  elseif strcmp(kernel_type, 'approx');
    [K_train_approx, ~, ~, random_weight_matrix] = getApproxKernel(X, X, data_rbf_variance, number_of_random_bases_for_data, -1);
    [K_test_approx, ~, ~, ~] = getApproxKernel(X, X_test, data_rbf_variance, number_of_random_bases_for_data, -1, random_weight_matrix);
    X = psi * H * K_train_approx;
    X_test = psi * H * K_test_approx;
  end
  projected_X = X;
  projected_X_test = X_test;














































