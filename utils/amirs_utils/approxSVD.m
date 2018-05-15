% -------------------------------------------------------------------------
function [U_approx, S_approx, V_approx] = approxSVD(X, k)
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

  % implementation from slide 9 of https://amath.colorado.edu/faculty/martinss/Talks/2010_mmds_martinsson.pdf

  A = X;

  [d,n] = size(A);

  Omega = randn(n,k);
  Y = A * Omega;
  [Q, R] = qr(Y); % Q now has orthonormal columns % NOT Q = randn(d,k);
  B = Q' * A;

  [U,S,V] = svd(B);

  U_approx = U;
  S_approx = S;
  V_approx = V;




% % n = 250;
% % d = 100;
% % A = randn(d,n);
% % A = (A * A')^2 * A;

% A = X;
% [d,n] = size(A);
% % mu = mean(A, 2);
% % A = bsxfun(@minus, A, mu);

% [U_1,S_1,V_1] = svd(A);

% k = 10;
% Omega = randn(n,k);
% Y = A * Omega;
% [Q, R] = qr(Y); % Q now has orthonormal columns
% % Q = randn(d,k);
% B = Q' * A;

% [U_2,S_2,V_2] = svd(B);

% norm(U_1 * S_1 * V_1' - A, 'fro')
% norm(Q * U_2 * S_2 * V_2' - A, 'fro')
% norm(Q * Q' * A - A, 'fro')
