% Copyright Barshan, Ghodsi 2009
% Paper: Supervised principal component analysis: Visualization, classification and
% regression on subspaces and submanifolds.
clear
clc
close all

% load sonar;
% load param_dim_sonar;

imdb = setupExperimentsUsingProjectedImbds('uci-ion', 'whatever', false, false);
vectorized_imdb = getVectorizedImdb(imdb);
X = vectorized_imdb.images.data';
Y = vectorized_imdb.images.labels;

[p,n] = size(X);
should_normalize = input('Is normalizetion required?(1/0)');
if should_normalize==1
    X = (X - repmat(min(X')', 1, n)) ./ (repmat(max(X')', 1, n) - repmat(min(X')', 1, n));
    nan_ind = find(isnan(X)==1);
    X(nan_ind) = 0;
end

ratio = 0.7;
number_of_training_samples = round(n * ratio);
indicies = randperm(n);
X = X(:,indicies);
Y = Y(indicies);

X_train = X(:, 1:number_of_training_samples);
X_test = X(:, number_of_training_samples + 1 : end);
Y_train = Y(1:number_of_training_samples);
Y_test = Y(number_of_training_samples + 1 : end);

projected_dim = 6;
param.k_type_y = 'delta_cls';
param.k_param_y = 1;
param.k_type_x = 'rbf';
param.k_param_x = 0.1; % param_dim_sonar(projected_dim);
[Z_train_SPCA U] = SPCA(X_train, Y_train, projected_dim, param);
[Z_train_KSPCA Beta] = KSPCA(X_train, Y_train, projected_dim, param);

% Testing Data Kernel Computation
K_test = repmat(0, size(X_train, 2), size(X_test, 2));
for i = 1 : size(X_train, 2)
    for j = 1 : size(X_test, 2)
        K_test(i,j) = kernel(param.k_type_x, X_train(:,i), X_test(:,j), param.k_param_x, []);
    end
end

Z_test_SPCA = U' * X_test;
Z_test_KSPCA = Beta' * K_test;


