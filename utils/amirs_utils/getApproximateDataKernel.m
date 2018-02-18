% -------------------------------------------------------------------------
function [projected_data, approximate_kernel] = getApproximateRBFDataKernel(data, number_of_basis, rbf_variance)
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


  % http://www.argmin.net/2017/12/05/kitchen-sinks/
  % TODO: must use same basis for both data_train and data_test

  data_dim = size(data, 1);
  number_of_samples = size(data, 2);

  gamma = 1 / (2 * rbf_variance ^ 2);
  w = randn(number_of_basis, data_dim);
  b = 2 * pi * rand(number_of_basis, 1);

  projected_data = cos(gamma * w * data + b * ones(1, number_of_samples));

  projected_data = projected_data' * projected_data;
