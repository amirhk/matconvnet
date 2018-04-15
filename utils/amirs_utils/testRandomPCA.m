% -------------------------------------------------------------------------
function testRandomPCA()
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



  dataset = 'xor-10D-350-train-150-test';
  % dataset = 'usps';
  % dataset = 'mnist-784';
  % dataset = 'uci-sonar';
  % dataset = 'synthetic-small';


  if strcmp(dataset, 'xor-10D-350-train-150-test')
    projected_dim_list = [1:1:10];
  elseif strcmp(dataset, 'usps')
    % projected_dim_list = [1:1:256];
    projected_dim_list = [1:15, 2.^[4:1:8]];
  elseif strcmp(dataset, 'mnist-784')
    projected_dim_list = 2.^[0:1:9];
  elseif strcmp(dataset, 'uci-sonar')
    projected_dim_list = [1:9,10:25:55,60];
  elseif strcmp(dataset, 'synthetic-small')
    projected_dim_list = [1,2];
  end

  repeat_count = 30;
  average_per_dim_delta_pca_rpca = zeros(repeat_count, length(projected_dim_list));
  average_per_dim_delta_pca_rpca_2 = zeros(repeat_count, length(projected_dim_list));
  average_per_dim_delta_pca_rp = zeros(repeat_count, length(projected_dim_list));

  for ii = 1:repeat_count

    fprintf('[INFO] trial #%d\n', ii);

    if strcmp(dataset, 'xor-10D-350-train-150-test')
      tmp_opts.dataset = dataset;
      imdb = loadSavedImdb(tmp_opts, false);
    elseif strcmp(dataset, 'usps')
      imdb = constructMultiClassImdbs(dataset, false);
      imdb = createImdbWithBalance(dataset, imdb, 25, 25, false, false);
    elseif strcmp(dataset, 'mnist-784')
      imdb = constructMultiClassImdbs(dataset, false);
      imdb = createImdbWithBalance(dataset, imdb, 25, 25, false, false);
      % imdb = createImdbWithBalance(dataset, imdb, 500, 500, false, false);
    elseif strcmp(dataset, 'uci-sonar')
      imdb = constructMultiClassImdbs(dataset, false);
    end

    %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    % Meta
    %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    if ~strcmp(dataset, 'synthetic-small')
      vectorized_imdb = getVectorizedImdb(imdb);
      indices_train = imdb.images.set == 1;
      indices_test = imdb.images.set == 3;
      X = vectorized_imdb.images.data(indices_train,:)';
      Y = vectorized_imdb.images.labels(indices_train);
      X_test = vectorized_imdb.images.data(indices_test,:)';
      Y_test = vectorized_imdb.images.labels(indices_test);
    else
      X = [randn(2,3), randn(2,3) + 5];
      Y = [1,1,1,2,2,2];
      X_test = [randn(2,3), randn(2,3) + 5];
      Y_test = [1,1,1,2,2,2];
    end


    per_dim_delta_pca_rpca = [];
    per_dim_delta_pca_rpca_2 = [];
    per_dim_delta_pca_rp = [];

    for projected_dim = projected_dim_list

      %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      % Compute PCA (on training data only)
      %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      % X, X_centered are << d by n >> matrices
      X_centered = bsxfun(@minus, X', mean(X',1));           % zero-center per feature
      C = (X_centered'*X_centered)./(size(X_centered,1)-1);  %' cov(X)

      [V D] = eig(C);
      [D order] = sort(diag(D), 'descend');                   % sort cols high to low
      V = V(:,order);

      % assert(numel(find(X_centered - X * H > 10e-3)) == 0)

      % [U D V] = svd(X_centered);
      % D_1 = D(1:min(size(D)),1:min(size(D)));

      % n = length(Y);
      % H = eye(n) - 1 / n * (ones(n,n));
      % tmp = X * H * H * X';
      % [U D V] = svd(tmp);
      % D_2 = D;

      % D_1.^2
      % D_2
      % assert(numel(find(D_1.^2 - D_2 > 10e-3)) == 0)

      % keyboard

      % % (X * H)
      % % (X * H)'
      % % (U * D.^0.5)
      % % (U * D.^0.5)'
      % (X * H) * (X * H)'
      % (U * D.^0.5) * (U * D.^0.5)'
      % % X * H = U * D.^0.5 up to a rotation
      % % H * X'
      % % D.^0.5 * U'
      % D_inv_half = pinv(diag([diag(D); 0; 0; 0; 0])).^0.5;
      % U_1 = X * H * D_inv_half %  = U % X * H * D.^-0.5 = U
      % U_1=U_1(:,1:min(size(D)))

      % U_2 = D_inv_half * H * X'
      % (U * D.^0.5)'

      % U' * U
      % U_1' * U_1

      % keyboard

      U_pca_exact = V(:,1:projected_dim);
      % U_pca_exact = U(:,1:projected_dim);
      % U_pca_exact = real(V(:,1:projected_dim) * diag(D(1:projected_dim).^0.5));
      projected_X_pca_exact = U_pca_exact' *  X;
      projected_X_test_pca_exact = U_pca_exact' *  X_test;

      tmp = squareform(pdist(projected_X_pca_exact'));
      distance_matrix_pca_exact = tmp / max(tmp(:));


      % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      % % Random PCA
      % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      % n = length(Y);
      % H = eye(n) - 1 / n * (ones(n,n));
      % label_rbf_variance = 10e-10;
      % number_of_random_bases_for_labels = 100;
      % [L_approx_w_many_bases, psi_w_many_bases, ~, ~] = getApproxKernel(1:n, 1:n, label_rbf_variance, number_of_random_bases_for_labels, -1);

      % tmp_projected_dim = projected_dim;
      % tmp_psi = psi_w_many_bases';
      % tmp_psi = bsxfun(@minus, tmp_psi, mean(tmp_psi,1));           % zero-center
      % C = (tmp_psi'*tmp_psi)./(size(tmp_psi,1)-1);                  %' cov(X)
      % [V D] = eig(C);
      % [D order] = sort(diag(D), 'descend');                         % sort cols high to low
      % V = V(:,order);
      % psi_w_few_bases = V(:,1:tmp_projected_dim)' * psi_w_many_bases;

      % % C = X * H * (psi_w_few_bases' * psi_w_few_bases) * H * X'./(size(tmp_X,1)-1);
      % % [V D] = eig(C);
      % % [D order] = sort(diag(D), 'descend');                   % sort cols high to low
      % % V = real(V(:,order));

      % % U_pca_random = V(:,1:projected_dim);
      % U_pca_random = X * H * psi_w_few_bases';
      % projected_X_pca_random = U_pca_random' * X;
      % projected_X_test_pca_random = U_pca_random' * X_test;

      % tmp = squareform(pdist(projected_X_pca_random'));
      % distance_matrix_pca_random = tmp / max(tmp(:));

      % % fprintf('Norm Diff PCA-RPCA: %.3f\n', norm(distance_matrix_pca_exact - distance_matrix_pca_random, 'fro'));



      % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      % % Random PCA
      % %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      % n = length(Y);
      % H = eye(n) - 1 / n * (ones(n,n));
      % label_rbf_variance = 10e-10;
      % number_of_random_bases_for_labels = projected_dim;
      % [L_approx, psi, ~, ~] = getApproxKernel(1:n, 1:n, label_rbf_variance, number_of_random_bases_for_labels, -1);

      % C = X * H * L_approx * H * X'./(size(tmp_X,1)-1);
      % % C = X * H * eye(size(H,1)) * H * X'./(size(tmp_X,1)-1);

      % [V D] = eig(C);
      % [D order] = sort(diag(D), 'descend');                   % sort cols high to low
      % keyboard
      % V = real(V(:,order));

      % U_pca_random = V(:,1:projected_dim);
      % projected_X_pca_random = U_pca_random' * X;
      % projected_X_test_pca_random = U_pca_random' * X_test;

      % tmp = squareform(pdist(projected_X_pca_random'));
      % distance_matrix_pca_random = tmp / max(tmp(:));

      % % fprintf('Norm Diff PCA-RPCA: %.3f\n', norm(distance_matrix_pca_exact - distance_matrix_pca_random, 'fro'));



      %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      % Random PCA
      %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      n = length(Y);
      H = eye(n) - 1 / n * (ones(n,n));
      label_rbf_variance = 10e-10;
      number_of_random_bases_for_labels = projected_dim;
      [L_approx, psi, ~, ~] = getApproxKernel(1:n, 1:n, label_rbf_variance, number_of_random_bases_for_labels, -1);

      % tmp = X * H * H * X';
      % [U D V] = svd(tmp);
      % inverse_D_to_the_half = diag(diag(D(1:projected_dim,1:projected_dim)).^(-.5)); % = inv(D(1:projected_dim,1:projected_dim).^(.5));
      % % inverse_D_to_the_half = eye(projected_dim);
      % % keyboard

      % U_pca_random = X * H * psi' * inverse_D_to_the_half;
      % U_pca_random = normc(X * H * psi');
      U_pca_random = X * H * psi';
      projected_X_pca_random = U_pca_random' * X;
      % projected_X_pca_random = U_pca_random';
      % projected_X_test_pca_random = U_pca_random' * X_test;

      tmp = squareform(pdist(projected_X_pca_random'));
      distance_matrix_pca_random = tmp / max(tmp(:));
      % keyboard




      %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      % Random PCA
      %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      % n = length(Y);
      % H = eye(n) - 1 / n * (ones(n,n));
      % label_rbf_variance = 10e-10;
      % number_of_random_bases_for_labels = projected_dim;
      % [L_approx, psi, ~, ~] = getApproxKernel(1:n, 1:n, label_rbf_variance, number_of_random_bases_for_labels, -1);

      tmp = X * H * H * X';
      [U D V] = svd(tmp);
      D_2 = D;
      % keyboard
      inverse_D_to_the_half = diag(diag(D(1:projected_dim,1:projected_dim)).^(-.5)); % = inv(D(1:projected_dim,1:projected_dim).^(.5));
      % inverse_D_to_the_half = eye(projected_dim);
      % keyboard

      U_pca_random = X * H * psi' * inverse_D_to_the_half;
      % U_pca_random = normc(X * H * psi');
      % U_pca_random = X * H * psi';
      projected_X_pca_random = U_pca_random' * X;
      projected_X_test_pca_random = U_pca_random' * X_test;

      tmp = squareform(pdist(projected_X_pca_random'));
      distance_matrix_pca_random_2 = tmp / max(tmp(:));







      %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      % Random Projection
      %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      w = 1 / sqrt(projected_dim) * randn(projected_dim, size(X,1));
      projected_X = w * X;
      projected_X_test = w * X_test;

      projected_X_random_projection = projected_X;
      projected_X_test_random_projection = projected_X_test;

      tmp = squareform(pdist(projected_X_random_projection'));
      distance_matrix_random_projection = tmp / max(tmp(:));



      %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      % Analysis
      %% -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
      % figure,

      % subplot(1,3,1),
      % imshow(distance_matrix_pca_exact),

      % subplot(1,3,2),
      % imshow(distance_matrix_pca_random),

      % subplot(1,3,3),
      % imshow(distance_matrix_random_projection),

      per_dim_delta_pca_rpca(end+1) = norm(distance_matrix_pca_exact - distance_matrix_pca_random, 'fro');
      per_dim_delta_pca_rpca_2(end+1) = norm(distance_matrix_pca_exact - distance_matrix_pca_random_2, 'fro');
      per_dim_delta_pca_rp(end+1) = norm(distance_matrix_pca_exact - distance_matrix_random_projection, 'fro');

      % fprintf('[k = %d] |D_{PCA} - D_{RPCA}|_F = %.3f\n', projected_dim, per_dim_delta_pca_rpca(end));
      % fprintf('[k = %d] |D_{PCA} - D_{RP}|_F = %.3f\n', projected_dim, per_dim_delta_pca_rp(end));

    end
    average_per_dim_delta_pca_rpca(ii,:) = per_dim_delta_pca_rpca;
    average_per_dim_delta_pca_rpca_2(ii,:) = per_dim_delta_pca_rpca_2;
    average_per_dim_delta_pca_rp(ii,:) = per_dim_delta_pca_rp;
  end


  legend_cell_array = {};
  figure,
  grid on,
  hold on,
  plot(projected_dim_list, mean(average_per_dim_delta_pca_rpca), 'LineWidth', 2); legend_cell_array = [legend_cell_array, '|D_{PCA} - D_{RPCA}|_F'];
  % plot(projected_dim_list, mean(average_per_dim_delta_pca_rpca_2), 'LineWidth', 2); legend_cell_array = [legend_cell_array, '|D_{PCA} - D_{RPCA_2}|_F'];
  plot(projected_dim_list, mean(average_per_dim_delta_pca_rp), 'LineWidth', 2); legend_cell_array = [legend_cell_array, '|D_{PCA} - D_{RP}|_F'];
  hold off,

  set(gca, 'YLim', [0, get(gca, 'YLim') * [0; 1]])

  xlabel('Projected Dimension');
  ylabel('Delta in Frob Norm');
  title(sprintf('%s - Comparison of RPCA vs RP (avg of %d)', dataset, repeat_count));
  legend(legend_cell_array, 'Location','northeast');


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
