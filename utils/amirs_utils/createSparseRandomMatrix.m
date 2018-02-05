% -------------------------------------------------------------------------
function sparse_random_matrix = createSparseRandomMatrix(dim_y, dim_x)
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

  % sparsity = 1;
  % sparsity = 3;
  sparsity = 30;



  % sparsity = 10;
  % sparsity = 100;
  % sparsity = 1000;
  % sparsity = 10000;
  % sparsity = 100000;

  tmp = rand(dim_y, dim_x);
  tmp(intersect(find(tmp <  1 / (2 * sparsity)),     find(tmp >= 0))) = -1;
  tmp(intersect(find(tmp <  1 - 1 / (2 * sparsity)), find(tmp >= 1 / (2 * sparsity)))) = 0;
  tmp(intersect(find(tmp <= 1),                      find(tmp >= 1 - 1 / (2 * sparsity)))) = +1;
  tmp = tmp * sqrt(sparsity);

  sparse_random_matrix = tmp;
  % sparse_random_matrix = zeros(dim_y, dim_x);
  % figure, imshow(sparse_random_matrix)
