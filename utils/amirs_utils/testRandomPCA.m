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

  % n = 250;
  % d = 100;

  % % n = 100;
  % % d = 25;

  % % % n = 25;
  % % % d = 10;

  % % n = 5;
  % % d = 3;

  % X = randn(d, n) + 2; % NO STRUCTURE
  % % X = [randn(d-2, n) + 2; randn(2, n) * 3]; % SOME STRUCTURE
  % % X = [randn(d-2, n) + 2; randn(1, n) * 3; randn(1, n) * 10]; % MORE STRUCTURE
  % % X = [randn(d-20, n) + 2; randn(10, n) * 3; randn(10, n) * 10]; % MORE STRUCTURE



  % n = 100; % number of samples
  % d = 1000;  % total space dims
  % p = 25;   % subspace dims

  % % n = 100; % number of samples
  % % d = 25;  % total space dims
  % % p = 10;   % subspace dims

  % % n = 10; % number of samples
  % % d = 5;  % total space dims
  % % p = 5;   % subspace dims

  % % n = 5; % number of samples
  % % d = 3;  % total space dims
  % % p = 2;   % subspace dims

  % B = randn(d,p);
  % Z = randn(d,n);
  % E = randn(d,n) * .75; % w/ standard deviation of .75 (* sqrt(.75) if variance of .75)

  % X = B * inv(B' * B) * B' * Z + E;







  % % dataset = 'xor-10D-350-train-150-test';
  % dataset = 'uci-sonar';
  % % dataset = 'uci-ion';
  % tmp_opts.dataset = dataset;
  % imdb = loadSavedImdb(tmp_opts, false);

  dataset = 'mnist-784';
  imdb = constructMultiClassImdbs(dataset, false);
  imdb = createImdbWithBalance(dataset, imdb, 250, 25, false, false);

  vectorized_imdb = getVectorizedImdb(imdb);
  indices_train = imdb.images.set == 1;
  indices_test = imdb.images.set == 3;
  X = vectorized_imdb.images.data(indices_train,:)';
  Y = vectorized_imdb.images.labels(indices_train);
  X_test = vectorized_imdb.images.data(indices_test,:)';
  Y_test = vectorized_imdb.images.labels(indices_test);
  [d, n] = size(X);



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
  % [U, S, V] = svd(X * X');
  [U, S, V] = svd(X);
  U_pca = U(:,1:k);
  X_pca = U_pca' * X;


% -------------------------------------------------------------------------
function [X_rpca, U_rpca] = getRPCAProjection1(X, k)
  % X is << d by n >>
% -------------------------------------------------------------------------
  d = size(X, 1);
  [~, phi, ~, ~] = getApproximateRBFKernel(1:d, 1:d, 10e-10, k);
  U_rpca = phi';
  X_rpca = U_rpca' * X;


% -------------------------------------------------------------------------
function [X_rpca, U_rpca] = getRPCAProjection2(X, k)
  % X is << d by n >>
% -------------------------------------------------------------------------
  [U_rpca, X_rpca, ~] = getApproximateLinearKernel(X, k);


% -------------------------------------------------------------------------
function [X_rp, U_rp] = getRPProjection(X, k)
  % X is << d by n >>
% -------------------------------------------------------------------------
  % U_rp = 1 / sqrt(k) * randn(size(X, 1), k);
  U_rp = 1 / sqrt(k) * randn(size(X, 1), k);
  X_rp = U_rp' * X;


% -------------------------------------------------------------------------
function pdist = getNormalizedPDistFromData(X)
  % X is << k by n >>
% -------------------------------------------------------------------------
  % tmp = squareform(pdist(X'));
  % % pdist = tmp / max(tmp(:));
  % pdist = tmp;
  % pdist = L2_distance(X,X);
  % pdist = L2_distance(X,X) .^ 2;
  % pdist = L2_distance(real(X),real(X));
  pdist = X' * X;










% -------------------------------------------------------------------------
function evaluateSandbox(X)
  % X is << d by n >>
% -------------------------------------------------------------------------
  [d, n] = size(X);

  figure,

  subplot_counter = 1;
  for k = [10:5:d]
    [X_pca, U_pca] = getPCAProjection(X, k);
    [X_rpca, U_rpca] = getRPCAProjection(X, k);
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
  all_dim_avg_delta_rpca_1_original = [];
  all_dim_avg_delta_rpca_2_original = [];
  all_dim_avg_delta_rp_original = [];

  if n <= 50
    projected_dim_list = 1:min(d,100);
    repeat_count = 100;
  elseif n <= 250
    projected_dim_list = 1:2:min(d,100);
    repeat_count = 100;
  else
    projected_dim_list = 1:5:min(d,100);
    repeat_count = 30;
  end

  pdist_original = getNormalizedPDistFromData(X);

  for k = [projected_dim_list]

    fprintf('[INFO] Running %d tests for k = %d\n', repeat_count, k);

    one_dim_all_delta_pca_original = [];
    one_dim_all_delta_rpca_1_original = [];
    one_dim_all_delta_rpca_2_original = [];
    one_dim_all_delta_rp_original = [];

    for ii = 1:repeat_count

      % PCA
      [X_pca, ~] = getPCAProjection(X, k);
      pdist_pca = getNormalizedPDistFromData(X_pca);

      % RPCA 1
      [X_rpca, ~] = getRPCAProjection1(X, k);
      pdist_rpca_1 = getNormalizedPDistFromData(X_rpca);

      % RPCA 2
      [X_rpca, ~] = getRPCAProjection2(X, k);
      pdist_rpca_2 = getNormalizedPDistFromData(X_rpca);

      % RP
      [X_rp, ~] = getRPProjection(X, k);
      pdist_rp = getNormalizedPDistFromData(X_rp);

      one_dim_all_delta_pca_original(end+1) = norm(pdist_original - pdist_pca, 'fro');
      one_dim_all_delta_rpca_1_original(end+1) = norm(pdist_original - pdist_rpca_1, 'fro');
      one_dim_all_delta_rpca_2_original(end+1) = norm(pdist_original - pdist_rpca_2, 'fro');
      one_dim_all_delta_rp_original(end+1) = norm(pdist_original - pdist_rp, 'fro');

    end

    all_dim_avg_delta_pca_original(end+1) = mean(one_dim_all_delta_pca_original);
    all_dim_avg_delta_rpca_1_original(end+1) = mean(one_dim_all_delta_rpca_1_original);
    all_dim_avg_delta_rpca_2_original(end+1) = mean(one_dim_all_delta_rpca_2_original);
    all_dim_avg_delta_rp_original(end+1) = mean(one_dim_all_delta_rp_original);

  end

  legend_cell_array = {};
  figure,
  grid on,
  hold on,
  plot(projected_dim_list, all_dim_avg_delta_pca_original, 'LineWidth', 2); legend_cell_array = [legend_cell_array, '|D_{PCA} - D_{ORIGINAL}|_F'];
  plot(projected_dim_list, all_dim_avg_delta_rpca_1_original, 'LineWidth', 2); legend_cell_array = [legend_cell_array, '|D_{RPCA 1} - D_{ORIGINAL}|_F'];
  plot(projected_dim_list, all_dim_avg_delta_rpca_2_original, 'LineWidth', 2); legend_cell_array = [legend_cell_array, '|D_{RPCA 2} - D_{ORIGINAL}|_F'];
  plot(projected_dim_list, all_dim_avg_delta_rp_original, 'LineWidth', 2); legend_cell_array = [legend_cell_array, '|D_{RP} - D_{ORIGINAL}|_F'];
  hold off,

  set(gca, 'YLim', [0, get(gca, 'YLim') * [0; 1]]);

  xlabel('Projected Dimension', 'FontSize', 14);
  ylabel('|X^T X - Z^T Z|_F', 'FontSize', 14);
  title(sprintf('Comparison of RPCA vs RP (avg of %d)', repeat_count), 'FontSize', 14);
  legend(legend_cell_array, 'Location','northeast', 'FontSize', 14);


















