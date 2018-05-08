n = 3;
d = 2;

X = randn(d,n) + 2;
H = eye(n) - 1 / n * (ones(n,n));

Q = (X * H) * (X * H)';
M = (X * H);

[U S V] = svd(Q);
N = U * S .^ .5;

% U_un_trunc = [U, zeros(d,n-d)];
% S_un_trunc = diag([diag(S); zeros(n-d,1)]);
% V_un_trunc = [V, zeros(d,n-d)];

% tmp_1 = U * S * V';
% tmp_2 = U_un_trunc * S_un_trunc * V_un_trunc';
% assert(numel(find(abs(tmp_1 - tmp_2) > 10e-3)) == 0);

assert(numel(find(abs(M * M' - N * N') > 10e-3)) == 0);

disp('norm of cols of M & N should NOT be equal')
sqrt(sum(M.^2, 1))
sqrt(sum(N.^2, 1))


disp('norm of cols of M^T & N^T should be equal')
sqrt(sum(M.^2, 2))
sqrt(sum(N.^2, 2))

sqrt(sum((N * R').^2, 1))
sqrt(sum((N * R').^2, 2))



R = M' / N';
R' * R

R = M / M;
R' *  R








































% M * M' = N * N'; % this means that M' and N' are rotations of one another, but
                   % M and N are not rotations of one another.

%% -----------------------------------------------------------------------------
% CLAIM: M' and N' (NOT M & N) are rotated versions of one another when M * M' = N * N'
R = M'/N';  % this holds even when M and N are not same dimensions...
tmp = R'*R; % this holds even when M and N are not same dimensions...
tmp = round(tmp * 10e3) / 10e3;
assert(isequal(diag(diag(tmp)), tmp)); % TODO: R'*R should be diagonal and only contain 1's starting from upper left corner



%% -----------------------------------------------------------------------------
% CLAIM: because, M' and N' are rotated versions of one another, they should
% have the same eigenvalues
[U_M, D_M, V_M] = svd(M'); % same goes for svd(M) because: A and its transpose A' have the same eigenvalues.
[U_N, D_N, V_N] = svd(N'); % same goes for svd(N) because: A and its transpose A' have the same eigenvalues.
assert(numel(find(abs(diag(D_M) - diag(D_N)) > 10e-3)) == 0);



%% -----------------------------------------------------------------------------
% FALSE CLAIM: M can be turned into U (from N = U * S ^ .5) by normalizing its columns
% and ordering from largest norm to smallest norm

% assert norm of columns of U all equal to 1
sqrt(sum(U.^2, 1))


% assert norm of columns of N equals to sqrt(diag(S)) in order
sqrt(sum(N.^2, 1))
(diag(S).^0.5)'

% Because M' and N' are rotations of one another, they have the same eigenvalues
% and the norm of the columns of M' and N' are equal (likewise norms of M & N's rows)
% but the singular values of N are contained in the norm of the columns of N (not N')

disp('norm of rows of M & N should be equal')
sqrt(sum(M.^2, 2))
sqrt(sum(N.^2, 2))

disp('norm of cols of M & N are not equal')
sqrt(sum(M.^2, 1))
sqrt(sum(N.^2, 1))






% R = M / N;
% R * R'
% R' * R


% R = N / M;
% R * R'
% R' * R


% R = M' / N';
% R * R'
% R' * R


% R = N' / M';
% R * R'
% R' * R







































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
