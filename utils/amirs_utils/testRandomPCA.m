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

  % % n = 250;
  % % d = 100;

  % n = 100;
  % d = 25;

  % % n = 25;
  % % d = 10;

  n = 5;
  d = 3;

  X = randn(d, n) + 2; % NO STRUCTURE
  % X = [randn(d-2, n) + 2; randn(2, n) * 3]; % SOME STRUCTURE
  % X = [randn(d-2, n) + 2; randn(1, n) * 3; randn(1, n) * 10]; % MORE STRUCTURE
  % X = [randn(d-20, n) + 2; randn(10, n) * 3; randn(10, n) * 10]; % MORE STRUCTURE



  % % n = 250; % number of samples
  % % d = 100;  % total space dims
  % % p = 25;   % subspace dims

  % % n = 100; % number of samples
  % % d = 25;  % total space dims
  % % p = 25;   % subspace dims

  % n = 10; % number of samples
  % d = 5;  % total space dims
  % p = 5;   % subspace dims

  % % n = 5; % number of samples
  % % d = 3;  % total space dims
  % % p = 2;   % subspace dims

  % B = randn(d,p);
  % Z = randn(d,n);
  % E = randn(d,n) * .75; % w/ standard deviation of .75 (* sqrt(.75) if variance of .75)

  % X = B * inv(B' * B) * B' * Z + E;







  % % dataset = 'xor-10D-350-train-150-test';
  % % % dataset = 'uci-sonar';
  % % % dataset = 'uci-ion';
  % % tmp_opts.dataset = dataset;
  % % imdb = loadSavedImdb(tmp_opts, false);

  % dataset = 'mnist-784';
  % imdb = constructMultiClassImdbs(dataset, false);
  % imdb = createImdbWithBalance(dataset, imdb, 100, 25, false, false);

  % vectorized_imdb = getVectorizedImdb(imdb);
  % indices_train = imdb.images.set == 1;
  % indices_test = imdb.images.set == 3;
  % X = vectorized_imdb.images.data(indices_train,:)';
  % Y = vectorized_imdb.images.labels(indices_train);
  % X_test = vectorized_imdb.images.data(indices_test,:)';
  % Y_test = vectorized_imdb.images.labels(indices_test);
  % [d, n] = size(X);



  H = eye(n) - 1 / n * (ones(n,n));
  X = X * H;
  % mean(X') % IMP: means are NOT completely 0,... they're on the order of 1.0e-3



  %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
  % EVALUATION
  %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

  % evaluateSandbox(X);
  evaluatePDistDifference(X);








% -------------------------------------------------------------------------
function [X_pca, U_pca] = getPCAProjection(X, k)
  % X is << d by n >>
% -------------------------------------------------------------------------
  [U, S, V] = svd(X * X');
  U_pca = U(:,1:k);
  X_pca = U_pca' * X;
  % X_pca = U_pca * U_pca' * X;


% -------------------------------------------------------------------------
function [X_rpca, U_rpca] = getRPCAProjection(X, d, k)
  % X is << d by n >>
% -------------------------------------------------------------------------
  [~, phi, ~, ~] = getApproxKernelRKS(1:d, 1:d, 10e-10, k);
  U_rpca = phi';
  X_rpca = U_rpca' * X;
  % X_rpca = U_rpca * U_rpca' * X;


% -------------------------------------------------------------------------
function [X_rp, U_rp] = getRPProjection(X, k)
  % X is << d by n >>
% -------------------------------------------------------------------------
  W = 1 / sqrt(k) * randn(size(X, 1), k);
  U_rp = W;
  X_rp = U_rp' * X;
  % X_rp = U_rp * U_rp' * X;


% -------------------------------------------------------------------------
function projected_pdist = getNormalizedPDistFromProjectedData(projected_X)
  % projected_X is << k by n >>
% -------------------------------------------------------------------------
  % tmp = squareform(pdist(projected_X'));
  % % projected_pdist = tmp / max(tmp(:));
  % projected_pdist = tmp;
  % projected_pdist = L2_distance(projected_X,projected_X);
  projected_pdist = L2_distance(projected_X,projected_X) .^ 2;
  % projected_pdist = L2_distance(real(projected_X),real(projected_X));


% -------------------------------------------------------------------------
function [L_approx, psi_data_1, psi_data_2, params] = getApproxKernelRKS(data_1, data_2, rbf_variance, number_of_random_bases, params)
  % data_1 is << d by n_1 >>; data_2 is << d by n_2 >>
% -------------------------------------------------------------------------
  assert(size(data_1, 1) == size(data_2, 1));
  d = size(data_1, 1);
  D = number_of_random_bases;
  s = rbf_variance;

  if nargin == 5
    w = params.w; % when random weight matrix passed in, use it instead of generating new random matrix: e.g., for constructing K_train & K_test
  else
    w = randn(D, d) / s; % make sure the w is shared between the 2 lines below! do not create w in <each> line below separately.
  end
  params.w = w; % random_weight_matrix

  % psi_data_1 = sqrt(1 / D) * cos(w * data_1);
  % psi_data_2 = sqrt(1 / D) * cos(w * data_2);

  psi_data_1 = sqrt(2 / D) * cos(w * data_1);
  psi_data_2 = sqrt(2 / D) * cos(w * data_2);

  % psi_data_1 = sqrt(2 / D) * exp(i * w * data_1); % 2 or 1 ??????
  % psi_data_2 = sqrt(2 / D) * exp(i * w * data_2); % 2 or 1 ??????

  L_approx = psi_data_1' * psi_data_2;





















% -------------------------------------------------------------------------
function evaluateSandbox(X)
  % X is << d by n >>
% -------------------------------------------------------------------------
  [d, n] = size(X);


  % [d, n] = size(X);
  % k = 2;
  % [X_pca, ~] = getPCAProjection(X, k);
  % [X_rpca, ~] = getRPCAProjection(X, d, k);
  % [X_rp, ~] = getRPProjection(X, k);
  % figure,
  % subplot(1,3,1), scatter(X_pca(1,:), X_pca(2,:));
  % subplot(1,3,2), scatter(X_rpca(1,:), X_rpca(2,:));
  % subplot(1,3,3), scatter(X_rp(1,:), X_rp(2,:));



  % k = 40;

  % % PCA
  % [X_pca, U_pca] = getPCAProjection(X, k);
  % pdist_pca = getNormalizedPDistFromProjectedData(X_pca);

  % % RPCA
  % [X_rpca, U_rpca] = getRPCAProjection(X, d, k);
  % pdist_rpca = getNormalizedPDistFromProjectedData(X_rpca);

  % % RP
  % [X_rp, U_rp] = getRPProjection(X, k);
  % pdist_rp = getNormalizedPDistFromProjectedData(X_rp);

  % figure,
  % subplot(3,3,1), imagesc(pdist_pca), colorbar, title('pdist pca')
  % subplot(3,3,2), imagesc(pdist_rpca ./ pdist_pca), colorbar, title('pdist rpca / pdist pca')
  % subplot(3,3,3), imagesc(pdist_rp ./ pdist_pca), colorbar, title('pdist rp / pdist pca')
  % subplot(3,3,4), imagesc(U_pca), colorbar, title('U pca')
  % subplot(3,3,5), imagesc(U_rpca), colorbar, title('U rpca')
  % subplot(3,3,6), imagesc(U_rp), colorbar, title('U rp')

  % U_pca_row_normalized = U_pca ./ repmat(sqrt(sum(U_pca.^2, 2)), [1,size(U_pca,2)]);
  % U_rpca_row_normalized = U_rpca ./ repmat(sqrt(sum(U_rpca.^2, 2)), [1,size(U_rpca,2)]);
  % U_rp_row_normalized = U_rp ./ repmat(sqrt(sum(U_rp.^2, 2)), [1,size(U_rp,2)]);

  % subplot(3,3,7), imagesc(U_pca_row_normalized), colorbar, title('U pca - row normalized')
  % subplot(3,3,8), imagesc(U_rpca_row_normalized), colorbar, title('U rpca - row normalized')
  % subplot(3,3,9), imagesc(U_rp_row_normalized), colorbar, title('U rp - row normalized')

  % keyboard

  % % keyboard

  % % figure, imagesc(pdist_rp ./ pdist_pca), colorbar

  % % keyboard

  figure,

  subplot_counter = 1;
  for k = [10:5:d]
    [X_pca, U_pca] = getPCAProjection(X, k);
    [X_rpca, U_rpca] = getRPCAProjection(X, d, k);
    R = X_rpca / X_pca;
    subplot(3,3,subplot_counter),
    imshow(R' * R),
    title(sprintf('k = %d', k), 'FontSize', 14);
    subplot_counter = subplot_counter + 1;
  end
  suptitle('Comparing PCA and RPCA in rotation');

  figure,

  subplot_counter = 1;
  for k = [10:5:d]
    [X_pca, U_pca] = getPCAProjection(X, k);
    [X_rp, U_rp] = getRPProjection(X, k);
    R = X_rp / X_pca;
    subplot(3,3,subplot_counter),
    imshow(R' * R),
    title(sprintf('k = %d', k), 'FontSize', 14);
    subplot_counter = subplot_counter + 1;
  end
  suptitle('Comparing PCA and RP in rotation');



% -------------------------------------------------------------------------
function evaluatePDistDifference(X)
  % X is << d by n >>
% -------------------------------------------------------------------------
  [d, n] = size(X);

  all_dim_avg_delta_pca_original = [];
  all_dim_avg_delta_rpca_original = [];
  all_dim_avg_delta_rp_original = [];

  if n <= 50
    projected_dim_list = 1:min(d,100);
    repeat_count = 100;
  elseif n <= 250
    projected_dim_list = 1:2:min(d,100);
    repeat_count = 30;
  else
    projected_dim_list = 1:5:min(d,100);
    repeat_count = 10;
  end

  pdist_original = getNormalizedPDistFromProjectedData(X);

  for k = [projected_dim_list]

    fprintf('[INFO] Running %d tests for k = %d\n', repeat_count, k);

    one_dim_all_delta_pca_original = [];
    one_dim_all_delta_rpca_original = [];
    one_dim_all_delta_rp_original = [];

    for ii = 1:repeat_count

      % PCA
      [X_pca, U_pca] = getPCAProjection(X, k);
      pdist_pca = getNormalizedPDistFromProjectedData(X_pca);

      % RPCA
      [X_rpca, U_rpca] = getRPCAProjection(X, d, k);
      pdist_rpca = getNormalizedPDistFromProjectedData(X_rpca);

      % RP
      [X_rp, U_rp] = getRPProjection(X, k);
      pdist_rp = getNormalizedPDistFromProjectedData(X_rp);

      one_dim_all_delta_pca_original(end+1) = norm(pdist_pca - pdist_original, 'fro');
      one_dim_all_delta_rpca_original(end+1) = norm(pdist_rpca - pdist_original, 'fro');
      one_dim_all_delta_rp_original(end+1) = norm(pdist_rp - pdist_original, 'fro');

      % if k == 2

      %   % if norm(pdist_pca - pdist_original, 'fro') > norm(pdist_rpca - pdist_original, 'fro')

      %   %   k
      %   %   ii
      %   %   norm(pdist_pca - pdist_original, 'fro')
      %   %   norm(pdist_rpca - pdist_original, 'fro')
      %   %   figure,
      %   %   subplot(1,4,1), scatter3(X(1,:), X(2,:), X(3,:)), title('X original'),
      %   %   subplot(1,4,2), scatter3(X_pca(1,:), X_pca(2,:), ones(1,size(X_pca, 2))), title('X pca'),
      %   %   subplot(1,4,3), scatter3(X_rpca(1,:), X_rpca(2,:), ones(1,size(X_rpca, 2))), title('X rpca'),
      %   %   subplot(1,4,4), scatter3(X_rp(1,:), X_rp(2,:), ones(1,size(X_rp, 2))), title('X rp'),

      %   %   keyboard

      %   % else
      %   if norm(pdist_pca - pdist_original, 'fro') > norm(pdist_rp - pdist_original, 'fro')

      %     k
      %     ii
      %     norm(pdist_pca - pdist_original, 'fro')
      %     norm(pdist_rp - pdist_original, 'fro')
      %     figure,
      %     subplot(1,4,1), scatter3(X(1,:), X(2,:), X(3,:)), title('X original'),
      %     subplot(1,4,2), scatter3(X_pca(1,:), X_pca(2,:), ones(1,size(X_pca, 2))), title('X pca'),
      %     subplot(1,4,3), scatter3(X_rpca(1,:), X_rpca(2,:), ones(1,size(X_rpca, 2))), title('X rpca'),
      %     subplot(1,4,4), scatter3(X_rp(1,:), X_rp(2,:), ones(1,size(X_rp, 2))), title('X rp'),

      %     keyboard

      %   end

      % end

    end

    all_dim_avg_delta_pca_original(end+1) = mean(one_dim_all_delta_pca_original);
    all_dim_avg_delta_rpca_original(end+1) = mean(one_dim_all_delta_rpca_original);
    all_dim_avg_delta_rp_original(end+1) = mean(one_dim_all_delta_rp_original);

  end

  legend_cell_array = {};
  figure,
  grid on,
  hold on,
  plot(projected_dim_list, all_dim_avg_delta_pca_original, 'LineWidth', 2); legend_cell_array = [legend_cell_array, '|D_{PCA} - D_{ORIGINAL}|_F'];
  plot(projected_dim_list, all_dim_avg_delta_rpca_original, 'LineWidth', 2); legend_cell_array = [legend_cell_array, '|D_{RPCA} - D_{ORIGINAL}|_F'];
  plot(projected_dim_list, all_dim_avg_delta_rp_original, 'LineWidth', 2); legend_cell_array = [legend_cell_array, '|D_{RP} - D_{ORIGINAL}|_F'];
  hold off,

  set(gca, 'YLim', [0, get(gca, 'YLim') * [0; 1]]);

  xlabel('Projected Dimension', 'FontSize', 14);
  ylabel('Delta in Frob Norm', 'FontSize', 14);
  title(sprintf('Comparison of RPCA vs RP (avg of %d)', repeat_count), 'FontSize', 14);
  legend(legend_cell_array, 'Location','northeast', 'FontSize', 14);


















