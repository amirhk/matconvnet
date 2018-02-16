function [projected_data_train, projected_data_test, projection_matrix] = KSPCA(data_train, labels_train, data_test, projected_dim, param)
%Copyright Barshan, Ghodsi 2009
%Paper: Supervised principal component analysis: Visualization, classification and
%regression on subspaces and submanifolds.
% Z = SPCA(X,Y,d,param)
% Input:
%       X:  explanatory variable (pxn)
%       Y:  response variables (lxn)
%       d:  number of projection dimensions
%       param:
%             param.k_type_y : kernel type of the response variable
%             param.k_param_y : kernel parameter of the response variable
%             param.k_type_x : kernel type of the explanatory variable
%             param.k_param_x : kernel parameter of the explanatory variable


% Output:
%       Z:  dimension reduced data (dxn)
%       Beta:  U = Phi(X)xBeta where U is the orthogonal projection matrix (pxd)
if size(data_train,2) ~= size(labels_train,2)
    error('data_train and labels_train must be the same length')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing Kernel Function of Labels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[l,n] = size(labels_train);
L = repmat(0, n, n);

% making L full rank for classification
if strcmp(param.k_type_y, 'delta_cls')
     L = L + eye(n);
     param.k_type_y = 'delta';
end
for i = 1:n
    for j = 1:n
        L(i,j) = L(i,j) + kernel(param.k_type_y, labels_train(:,i), labels_train(:,j), param.k_param_y, []);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing Kernel Function of Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
actual = 0;

if actual == 1
  K_train = repmat(0, n, n);
  for i = 1:n
      for j = 1:n
          K_train(i,j) = kernel(param.k_type_x, data_train(:,i), data_train(:,j), param.k_param_x, []);
      end
  end
  % data_test Kernel Computation
  K_test = repmat(0, size(data_train, 2), size(data_test, 2));
  for i = 1 : size(data_train, 2)
      for j = 1 : size(data_test, 2)
          K_test(i,j) = kernel(param.k_type_x, data_train(:,i), data_test(:,j), param.k_param_x, []);
      end
  end
else
  % initial_dim = 34;
  % vectorized_data_train = data_train;
  % vectorized_data_test = data_test;
  % % number_of_random_lines = initial_dim * 10;
  % number_of_random_lines = 100;

  % keyboard

  % rbf_width = 0.1;
  % gamma = 1 / (2 * rbf_width ^ 2);

  % % random projection lines are in every row
  % random_projection_matrix = mvnrnd(zeros(initial_dim, 1), 2 * gamma * eye(initial_dim), number_of_random_lines);
  % random_offset_vector = rand(number_of_random_lines, 1) * 2 * pi;

  % % cos(w'x + b)
  % % data samples are in every column
  % % random_offset_vector is added to every column
  % % cos() is applied elementwise
  % vectorized_projected_data_train = cos(random_projection_matrix * vectorized_data_train + random_offset_vector);
  % vectorized_projected_data_test = cos(random_projection_matrix * vectorized_data_test + random_offset_vector);
  % K_train = vectorized_projected_data_train' * vectorized_projected_data_train;
  % K_test = vectorized_projected_data_train' * vectorized_projected_data_test;

  X_train = data_train;
  X_test = data_test;

  original_dim = size(data_train, 1); % 34;
  D = 1000; % number of basis for approximation

  gamma = 1 / (2*param.k_param_x^2);
  w = randn(D, original_dim);
  b = 2 * pi * rand(D, 1);

  start_time_train = cputime;
  Z_train = cos(gamma * w * X_train + b * ones(1,size(data_train,2)));
  end_time_train = cputime;
  training_time_approx = end_time_train - start_time_train;

  start_time_test = cputime;
  Z_test = cos(gamma * w * X_test + b * ones(1,size(data_test,2)));
  end_time_test = cputime;
  testing_time_approx = end_time_test - start_time_test;

  K_train_approx = Z_train' * Z_train;
  K_test_approx = Z_train' * Z_test;

  K_train = K_train_approx;
  K_test = K_test_approx;
end

% size(K_train)
% size(K_test)
% keyboard



% TODO: write the approximation code here and compare with K_train...?
% keyboard


H = eye(n) - 1 / n * (ones(n,n));
tmp = H * L * H * K_train';
[Beta D] = eigendec(tmp, projected_dim, 'LM');
% Z = Beta' * K_train;





projection_matrix = Beta;
projected_data_train = projection_matrix' * K_train;
projected_data_test = projection_matrix' * K_test;
