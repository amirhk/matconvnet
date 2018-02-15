% -------------------------------------------------------------------------
function projected_imdb = getSPCAEmbedding(imdb, projected_dim)
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

  [train_imdb, test_imdb] = splitImdb(imdb);

  vectorized_train_imdb = getVectorizedImdb(train_imdb);
  vectorized_test_imdb = getVectorizedImdb(test_imdb);

  spca_projection_matrix = computeSPCAMatrix(vectorized_train_imdb, projected_dim); % projection matrix calculated only on the training data
  projected_imdb = projectImdb(imdb, spca_projection_matrix);                       % projecing both training and testing data
  % keyboard


% -------------------------------------------------------------------------
function projected_imdb = projectImdb(imdb, spca_projection_matrix)
% -------------------------------------------------------------------------
  vectorized_imdb = getVectorizedImdb(imdb);
  vectorized_data = vectorized_imdb.images.data';
  projected_vectorized_data = spca_projection_matrix * vectorized_data;
  projected_vectorized_imdb = vectorized_imdb;
  projected_vectorized_imdb.images.data = single(projected_vectorized_data');

  projected_dim = size(projected_vectorized_imdb.images.data, 2);
  number_of_samples = size(projected_vectorized_imdb.images.data, 1);
  projected_imdb = get4DImdb(projected_vectorized_imdb, projected_dim, 1, 1, number_of_samples);


% -------------------------------------------------------------------------
function spca_projection_matrix = computeSPCAMatrix(imdb, projected_dim)
% -------------------------------------------------------------------------
  number_of_samples = size(imdb.images.data, 1);
  number_of_dimensions = size(imdb.images.data, 2);

  X = double(imdb.images.data');
  Y = double(getOneHotLabels(imdb.images.labels));

  % param.k_type_y = 'delta_cls';
  % param.k_param_y = 1;
  % [Ztr_SPCA U] = SPCA(X, Y, projected_dim, param);
  % spca_projection_matrix = U;

  e = ones(number_of_samples, 1);
  H = eye(number_of_samples) - (1 / number_of_samples) * e * e';

  L = Y' * Y;
  Q = X * H * L * H * X';
  % Q(1:4,1:4)
  % keyboard

  % [V,D] = eigs(double(Q), projected_dim);
  % projected_dim
  % keyboard
  [V,d] = eigendec(Q, projected_dim, 'LM');
  % is_v_real = 0;
  % while ~is_v_real
  %   [V,D] = eigs(Q, projected_dim);
  %   is_v_real = isreal(V)
  % end
  % fprintf('\nsize Q: %d x %d, is V real: %d\n', size(Q,1), size(Q,2), isreal(V))
  % keyboard
  spca_projection_matrix = V';



% -------------------------------------------------------------------------
function one_hot_labels = getOneHotLabels(digit_labels)
% -------------------------------------------------------------------------
  I = eye(max(digit_labels));
  one_hot_labels = I(digit_labels,:)';
