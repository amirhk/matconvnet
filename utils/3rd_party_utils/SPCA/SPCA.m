function [projected_data_train, projected_data_test, projection_matrix] = SPCA(data_train, labels_train, data_test, projected_dim, param)
% Copyright Barshan, Ghodsi 2009
% Paper: Supervised principal component analysis: Visualization, classification and
% regression on subspaces and submanifolds.

% [Z U] = SPCA(X,Y,d,param)
% Input:
%       X:  explanatory variable (pxn)
%       Y:  response variables (lxn)
%       d:  dimension of effective subspaces
%       param:
%             param.k_type_y : kernel type of the response variable
%             param.k_param_y : kernel parameter of the response variable

% Output:
%       Z:  dimension reduced data (dxn)
%       U:  orthogonal projection matrix (pxd)

if size(data_train,2) ~= size(labels_train,2)
    error('data_train and labels_train must be the same length')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing Kernel Function of Labels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[l, n] = size(labels_train);
L = repmat(0, n, n);

% Making L full rank for classification
if strcmp(param.k_type_y, 'delta_cls')
     % L = L + eye(n);
     param.k_type_y = 'delta';
end

actual = 1;

% for i = 1 : n
%     for j = 1 : n
%         L(i,j) = L(i,j) + kernel(param.k_type_y, labels_train(:,i), labels_train(:,j), param.k_param_y, []);
%     end
% end
% L_actual = L;

% number_of_basis = 10;
% data_dim = 1; % IMPORTANT size(labels_train, 1);
% number_of_labels = size(labels_train, 2);

% rbf_variance = 10e-15 ; % extremely small variance because we are approximating delta kernel
% gamma = 1 / (2 * rbf_variance ^ 2);
% w = randn(number_of_basis, data_dim);
% b = 2 * pi * rand(number_of_basis, 1);

% projected_labels = cos(gamma * w * labels_train + b * ones(1, number_of_labels));
% psi = sqrt(2 / number_of_basis) * projected_labels;
% L_approx = psi' * psi;

% figure,

% subplot(3,2,1),
% imshow(L_actual),
% title('L actual'),
% subplot(3,2,2),
% imshow(L_approx),
% title(sprintf('L approx w/ 10 bases; norm diff: %.3f', norm(L_actual - L_approx))),


% number_of_basis = 100;
% data_dim = 1; % IMPORTANT size(labels_train, 1);
% number_of_labels = size(labels_train, 2);

% rbf_variance = 10e-15 ; % extremely small variance because we are approximating delta kernel
% gamma = 1 / (2 * rbf_variance ^ 2);
% w = randn(number_of_basis, data_dim);
% b = 2 * pi * rand(number_of_basis, 1);

% projected_labels = cos(gamma * w * labels_train + b * ones(1, number_of_labels));
% psi = sqrt(2 / number_of_basis) * projected_labels;
% L_approx = psi' * psi;



% subplot(3,2,3),
% imshow(L_actual),
% title('L actual'),
% subplot(3,2,4),
% imshow(L_approx),
% title(sprintf('L approx w/ 100 bases; norm diff: %.3f', norm(L_actual - L_approx))),


% number_of_basis = 1000;
% data_dim = 1; % IMPORTANT size(labels_train, 1);
% number_of_labels = size(labels_train, 2);

% rbf_variance = 10e-15 ; % extremely small variance because we are approximating delta kernel
% gamma = 1 / (2 * rbf_variance ^ 2);
% w = randn(number_of_basis, data_dim);
% b = 2 * pi * rand(number_of_basis, 1);

% projected_labels = cos(gamma * w * labels_train + b * ones(1, number_of_labels));
% psi = sqrt(2 / number_of_basis) * projected_labels;
% L_approx = psi' * psi;


% subplot(3,2,5),
% imshow(L_actual),
% title('L actual'),
% subplot(3,2,6),
% imshow(L_approx),
% title(sprintf('L approx w/ 1000 bases; norm diff: %.3f', norm(L_actual - L_approx))),


% keyboard


if actual == 1

  % -----------------------------------------------------------------------------
  % Actual Target (!) Kernel
  % -----------------------------------------------------------------------------
  for i = 1 : n
      for j = 1 : n
          L(i,j) = L(i,j) + kernel(param.k_type_y, labels_train(:,i), labels_train(:,j), param.k_param_y, []);
      end
  end

  H = eye(n) - 1 / n * (ones(n,n));
  [p, n] = size(data_train);
  if n > p
      tmp = data_train * H * L * H * data_train';
      [U D] = eigendec(tmp, projected_dim, 'LM');
  else
     [u, s, v] = svd(L);
     phi_Y = s ^ .5 * v';
     tmp = phi_Y * H * data_train' * data_train * H * phi_Y';
     [V D] = eigendec(tmp, projected_dim, 'LM');
     U = data_train * H * phi_Y' * V * inv(diag(D) ^ .5);
  end

  % keyboard
  % TODO: compare L with decomposed y * y', where y = a * mpower(b, 0.5), and [a,b,c] = svd(L) - DONE, MATCHED!
  % TODO: compare L with decomposed y' * y, where y = approx kernel from below - DONE, MAKE SURE TO NORMALIZE psi (*sqrt(2 / number_of_basis)) AND TO ADD eye(number_of_labels)

else

  % -----------------------------------------------------------------------------
  % Approx Target (!) Kernel
  % -----------------------------------------------------------------------------

  number_of_basis = projected_dim;
  data_dim = 1; % IMPORTANT size(labels_train, 1);
  number_of_labels = size(labels_train, 2);

  rbf_variance = 10e-15 ; % extremely small variance because we are approximating delta kernel
  gamma = 1 / (2 * rbf_variance ^ 2);
  w = randn(number_of_basis, data_dim);
  b = 2 * pi * rand(number_of_basis, 1);

  projected_labels = cos(gamma * w * labels_train + b * ones(1, number_of_labels));
  psi = sqrt(2 / number_of_basis) * projected_labels;

  add_eye = 0;
  if add_eye
    L_approx = psi' * psi + eye(number_of_labels); % DUMB SHIT WE HAVE TO DO...

    [a,b,c] = svd(L_approx);

    psi_correct = a * mpower(b, 0.5);
    psi_correct = psi_correct(:,1:projected_dim);
    psi = psi_correct'; % SVD is [a * mpower(b,0.5)] * [mpower(b,0.5) * c'], which means the first part then requires a transpose.
  end

  % X = data_train;
  % H = eye(n) - 1 / n * (ones(n,n));
  % U_approx = X * H * psi';
  % U = U_approx;
  % % keyboard

  % % L_approx = psi' * psi;
  % % L = L_approx; %  + eye(n); % INTERESTINGLY, eye(n) is not really required here... because L_approx has a bunch of random values... and so it's not close to singular unlike (the above 'acutal' case)
  % % tmp = data_train * H * L * H * data_train';
  % % [U D] = eigendec(tmp, projected_dim, 'LM');

  % X = data_train;
  % H = eye(n) - 1 / n * (ones(n,n));
  % U_approx = X * H * psi';
  % U = U_approx;
  % U_scaled = U;

  L_approx = psi' * psi;
  L = L_approx; %  + eye(n); % INTERESTINGLY, eye(n) is not really required here... because L_approx has a bunch of random values... and so it's not close to singular unlike (the above 'acutal' case)
  H = eye(n) - 1 / n * (ones(n,n));
  tmp = data_train * H * L * H * data_train';
  [U D] = eigendec(tmp, projected_dim, 'LM');
  % U_unscaled = U;


  % U_scaled(1:5,1:5)
  % U_unscaled(1:5,1:5)

  % keyboard

end

% figure, imshow(U)
% % size(U)
% keyboard

projection_matrix = U;
projected_data_train = projection_matrix' * data_train;
projected_data_test = projection_matrix' * data_test;



































