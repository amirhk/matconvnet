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
     L = L + eye(n);
     param.k_type_y = 'delta';
end

actual = 1;

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
else

  % -----------------------------------------------------------------------------
  % Approx Target (!) Kernel
  % -----------------------------------------------------------------------------

  fprintf('projected_dim: %d', projected_dim);
  number_of_bases = projected_dim;
  data_dim = 1; % IMPORTANT size(labels_train, 1);
  number_of_labels = size(labels_train, 2);

  rbf_variance = 10e-15 ; % extremely small variance because we are approximating delta kernel
  gamma = 1 / (2 * rbf_variance ^ 2);
  w = randn(number_of_bases, data_dim);
  b = 2 * pi * rand(number_of_bases, 1);

  projected_labels = cos(gamma * w * labels_train + b * ones(1, number_of_labels));
  psi = sqrt(2 / number_of_bases) * projected_labels;

  % % INTERESTINGLY, eye(n) is not really required here... because L_approx has a
  % % bunch of random values... and so it's not close to singular (unlike the
  % % above 'acutal' case)
  % add_eye = 0;
  % if add_eye
  %   L_approx = psi' * psi + eye(number_of_labels); % DUMB SHIT WE HAVE TO DO...

  %   [a,b,c] = svd(L_approx);

  %   psi_correct = a * mpower(b, 0.5);
  %   psi_correct = psi_correct(:,1:projected_dim);
  %   psi = psi_correct'; % SVD is [a * mpower(b,0.5)] * [mpower(b,0.5) * c'], which means the first part then requires a transpose.
  % end

  X = data_train;
  H = eye(n) - 1 / n * (ones(n,n));
  U_approx = X * H * psi';

  L_approx = psi' * psi;

  assert(numel(find(X * H * psi' * psi * H * X' - X * H * L_approx * H * X' > 10e-3)) == 0);
  % assert(numel(find(X * H * psi' * psi * H * X' - X * H * L_approx * H * X' > 10e-4)) == 0);

  % L_actual = repmat(0, n, n);
  % for i = 1 : n
  %     for j = 1 : n
  %         L_actual(i,j) = L_actual(i,j) + kernel(param.k_type_y, labels_train(:,i), labels_train(:,j), param.k_param_y, []);
  %     end
  % end

  % figure,
  % subplot(1,2,1),
  % imshow(L_actual, []),
  % subplot(1,2,2),
  % imshow(L_approx, []),

  tmp = U_approx * U_approx';
  [U D] = eigendec(tmp, projected_dim, 'LM');
  U_actual = U;


  % Goal, starting with U_approx, want to get to U_actual...
  % My claim is that because tmp_1 * tmp_1' == tmp_2 * tmp_2', we can conclude
  % that tmp_1 and tmp_2 are rotations of one another. After that, we should
  % left-multiply tmp_2 by inv(diag(D) ^ 0.5) to get tmp_1 = U_approx.

  tmp_1 = U_approx;
  tmp_2 = U_actual * diag(D) ^ 0.5;

  % Step 1) Confirm tmp_1 * tmp_1' == tmp_2 * tmp_2'
  assert(numel(find(tmp_1 * tmp_1' - tmp_2 * tmp_2' > 10e-3)) == 0);

  % Step 2) Get rotational relationship between tmp_1 and tmp_2
  R = tmp_2' / tmp_1';
  assert(numel(find(R * R' - eye(projected_dim) > 10e-10)) == 0);
  assert(numel(find(R * tmp_1' - tmp_2' > 10e-10)) == 0);

  % Step 3) ...
  U_actual - R * U_approx' * inv(diag(D) ^ 0.5);



  % keyboard

  % a = normc(U_actual);
  % b = normc(U_approx);

  % a' * a
  % b' * b

  % keyboard


  % R = U_actual' / U_approx';
  % % find(R * U_approx' - U_actual' > 10e-5)
  % U = R * U_approx';
  % U = U';

  % R = U_actual' / U_approx';
  % U = R * U_approx';
  % keyboard

  % U = U_actual;
  U = U_approx;

end

keyboard

projection_matrix = U;
projected_data_train = projection_matrix' * data_train;
projected_data_test = projection_matrix' * data_test;



































