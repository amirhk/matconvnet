% -------------------------------------------------------------------------
function [projection_matrix, projected_data, approximate_kernel] = getApproximateLinearKernel(data, number_of_basis)
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

  % http://proceedings.mlr.press/v22/kar12/kar12.pdf

  X = data;
  [d,n] = size(X);
  k = number_of_basis;

  p = 2; % value fixed from paper
  fh_p = @(p,n) 1./p.^(n+1);
  tmp = fh_p(p, 0:100);
  tmp = tmp / sum(tmp); % now we have a discrete probability distribution whose sum is 1.
  % tabulate(discretesample(tmp, 100) - 1) % sanity check, the first few elements should have the highest count!

  % Maybe use this for higher order polynomials as well.
  % Drawback is that we have to calculate the derivative for each k.
  % syms f(x)
  % f(x) = x;
  % df = diff(f,x,N)
  % a_N = df(0);

  % Z = zeros(k,n);
  % for i = 1 : k
  %   N = discretesample(tmp, 1) - 1; % -1 because we want a non-negative integer, which allows for 0. discretesample's first index is 1, ... so we simply subtract by 1.
  %   if N == 1
  %     a_N = 1; % for linear kernels
  %   else
  %     a_N = 0;
  %   end
  %   W = sign(randn(d, N)); % choosing N vectors w_1, ..., w_N \in {-1,+1}^d
  %   Z_i = sqrt(a_N * p ^ (N + 1)) * prod(W' * data, 1);
  %   assert(isequal(size(Z_i), [1,n]));
  %   Z(i,:) = Z_i;
  % end
  % Z = 1/sqrt(k) * Z;
  % K_approx = Z' * Z;

  % projected_data = Z;
  % approximate_kernel = K_approx;




  N = discretesample(tmp, k) - 1; % -1 because we want a non-negative integer, which allows for 0. discretesample's first index is 1, ... so we simply subtract by 1.
  M = zeros(k, d);
  M(N == 1,:) = 1; % mask: mark those rows with N = 1 as ones
  W = 2/sqrt(k) * sign(randn(k, d)); % 2/sqrt(k) = sqrt(1 * p ^ 2) * 1/sqrt(k)
  Psi = (M .* W);

  % Psi = 1/sqrt(k) * sign(randn(k, d));
  % Psi = 1/sqrt(k) * randn(k, d);

  Z = Psi * X;
  K_approx = Z' * Z;

  projection_matrix = Psi;
  projected_data = Z;
  approximate_kernel = K_approx;

