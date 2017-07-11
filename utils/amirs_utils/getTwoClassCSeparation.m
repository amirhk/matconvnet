% -------------------------------------------------------------------------
function c_separation = getTwoClassCSeparation(imdb)
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

  assert(numel(unique(imdb.images.labels)) == 2);

  vectorized_imdb = getVectorizedImdb(imdb);
  vectorized_data_1 = vectorized_imdb.images.data(imdb.images.labels == 1,:);
  vectorized_data_2 = vectorized_imdb.images.data(imdb.images.labels == 2,:);

  mean_1 = mean(vectorized_data_1);
  mean_2 = mean(vectorized_data_2);

  cov_1 = cov(vectorized_data_1);
  cov_2 = cov(vectorized_data_2);

  c_separation = norm(mean_1 - mean_2) / sqrt(max(trace(cov_1), trace(cov_2)));

  % c_granularity = 0.1;
  % c = c_granularity;
  % flag = true
  % while 1
  %   % || mean_1 - mean_2 || >= c sqrt{ max{ trace(cov_1), trace(cov_2) } }
  %   if
  %     break
  %   end
  %   c = c + c_granularity;
  % end
